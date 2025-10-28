#include "alphazero/trainer.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <future>
#include <torch/serialize/archive.h>

namespace alphazero {

namespace {
torch::optim::AdamOptions make_adam_options(const TrainingConfig& cfg) {
  torch::optim::AdamOptions opt(cfg.learning_rate);
  opt.weight_decay(cfg.weight_decay);
  return opt;
}

// TODO: pass separate mcts config in ctor
MCTSConfig make_mcts_cfg(const TrainingConfig& cfg) {
  MCTSConfig m;
  m.num_simulations   = cfg.num_simulations;
  m.cpuct             = cfg.cpuct;
  m.dirichlet_alpha   = cfg.dirichlet_alpha;
  m.dirichlet_epsilon = cfg.dirichlet_epsilon;
  m.gamma             = cfg.gamma;
  return m;
}
} // namespace

AlphaZeroTrainer::AlphaZeroTrainer(PolicyNetwork network,
                                   eurobot::EurobotWorld world,
                                   const TrainingConfig& cfg)
  : cfg_(cfg),
    policy_(std::move(network)),
    world_(std::move(world)),
    replay_(cfg_.replay_capacity),
    mcts_cfg_(make_mcts_cfg(cfg)), 
    mcts_(mcts_cfg_, policy_),
    optimizer_(policy_->parameters(), make_adam_options(cfg_)) {

  TORCH_CHECK(policy_, "AlphaZeroTrainer: policy network must be non-null");
  action_dim_ = static_cast<int64_t>(world_.action_space.size());

  if (cfg_.enable_tensorboard && !cfg_.log_dir.empty()) {
    try {
      tb_logger_ = std::make_unique<TensorboardLogger>(cfg_.log_dir, cfg_.run_name);
      std::cout << "TensorBoard events: " << tb_logger_->file_path() << std::endl;
    } catch (const std::exception& ex) {
      std::cerr << "[AlphaZeroTrainer] TensorBoard logger disabled: " << ex.what() << std::endl;
    }
  }
}

void AlphaZeroTrainer::set_device(const torch::Device& device) {
  device_ = device;
  policy_->to(device_);
}

void AlphaZeroTrainer::train() {
  try_resume();

  if (iteration_counter_ >= cfg_.num_iterations) {
    std::cout << "[AlphaZeroTrainer] Requested iterations already completed ("
              << iteration_counter_ << " >= " << cfg_.num_iterations << ")." << std::endl;
    maybe_save_checkpoint(/*force=*/true);
    return;
  }

  // ensure Torch is single-threaded inside workers
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  // choose worker count
  const int W = std::max(1u, std::thread::hardware_concurrency());
  // const int W = std::min(cfg_.games_per_iter, (int)std::max(1u, std::thread::hardware_concurrency()));
  workers_.reserve(W);
  for (int i = 0; i < W; ++i) {
    workers_.emplace_back(world_, policy_, mcts_cfg_);
  }


  for (int it = static_cast<int>(iteration_counter_); it < cfg_.num_iterations; ++it) {
    std::cout<<"Iteration num: "<<it<<std::endl;
    const int G = cfg_.games_per_iter;
    const int base = G / W, rem = G % W;

    std::vector<std::future<WorkerBatch>> futs;
    futs.reserve(W);

    const auto root_state = world_.get_state(); // snapshot once

    for (int i = 0; i < W; ++i) {
      const int episodes_for_this_worker = base + (i < rem ? 1 : 0);
      if (episodes_for_this_worker == 0) continue;

      futs.emplace_back(std::async(std::launch::async, [this, i, episodes_for_this_worker, root_state]{
        auto& wk = workers_[i];
        WorkerBatch wb;
        wb.traj.reserve(episodes_for_this_worker * 64);

        for (int e = 0; e < episodes_for_this_worker; ++e) {
          wk.world.set_state(root_state);              // TODO: swap with world.reset()
          auto traj = play_episode_collect(wk.world, wk.mcts);
          if (!traj.empty()) {
            ++wb.episodes;
            wb.blue_sum += traj.front().value;         
            wb.traj.insert(wb.traj.end(),
                          std::make_move_iterator(traj.begin()),
                          std::make_move_iterator(traj.end()));
          }
        }
        return wb;
      }));
    }

    int episodes_collected = 0;
    double blue_return_sum = 0.0;  

    for (auto& f : futs) {
      auto wb = f.get();
      episodes_collected += wb.episodes;
      blue_return_sum     += wb.blue_sum;
      for (auto &t : wb.traj) {
        replay_.add(t.obs.to(device_), t.pi.to(device_), t.value);
      }
    }

    if (tb_logger_ && episodes_collected > 0) {
      const double avg_return = blue_return_sum / (double)episodes_collected;
      tb_logger_->add_scalar("train/avg_return", iteration_counter_, avg_return);
      tb_logger_->add_scalar("rollout/blue_final_score_mean", iteration_counter_, avg_return * 160.0);
    }

    for (int s = 0; s < cfg_.training_steps; ++s) {
      optimize_step();
    }
    ++iteration_counter_;
    maybe_save_checkpoint();
  }
  maybe_save_checkpoint(/*force=*/true);
}

std::pair<float, float> AlphaZeroTrainer::play_episode() {
  policy_->eval();
  world_.reset();
  bool done = false;
  int step_idx = 0;

  std::vector<torch::Tensor> obs_list;
  std::vector<std::vector<float>> pi_list;

  while (!done) {
    // run MCTS from current state ()
    auto root = mcts_.search(world_);  // returns TreeNode by value (has children with ptrs)

    // visit-count policy
    std::vector<float> visits(action_dim_, 0.0f);
    int total_visits = 0;
    for (const auto& kv : root.children) {
      const int aidx = kv.first;
      const int n    = kv.second->vis_count;
      visits[aidx] = static_cast<float>(n);
      total_visits += n;
    }
    if (total_visits <= 0) {
      // fallback: legal mask → uniform over legal
      auto legal = world_.legal_mask();
      auto legal_cpu = legal.to(torch::kCPU);
      visits.assign(action_dim_, 0.0f);
      for (int64_t i = 0; i < legal_cpu.numel(); ++i)
        if (legal_cpu[i].item<bool>()) visits[(size_t)i] = 1.0f;
      total_visits = static_cast<int>(std::accumulate(visits.begin(), visits.end(), 0.0f));
      if (total_visits == 0) return {0.f, 0.f}; // degenerate
    }
    // normalize
    for (float& v : visits) v = v / std::max(1, total_visits);

    // record obs + policy target
    obs_list.push_back(world_.obs());   // 1D tensor
    pi_list.push_back(visits);

    // select action with temperature schedule
    const double T = select_temperature(step_idx);
    const int action_idx = sample_action_from_visits(visits, T);
    const auto& action = world_.action_space[action_idx];

    // step environment (Blue)
    done = world_.step_blue(action) || step_idx > cfg_.max_env_steps;

    ++step_idx;
  }

  std::cout << "Number of steps: "<< step_idx << std::endl;

  // terminal value (normalized blue score)
  const auto scores = world_.final_scores_norm(); // pair<float,float>
  const float ret = scores.first;

  // commit episode to replay
  for (size_t i = 0; i < obs_list.size(); ++i) {
    // obs: 1D float; pi: 1D float (A); value: scalar float
    auto piT = torch::from_blob(pi_list[i].data(), { (long)action_dim_ }, torch::kFloat32).clone();
    replay_.add(obs_list[i].detach().clone(), piT, ret);
  }
  return scores;
}

std::vector<Transition> AlphaZeroTrainer::play_episode_collect(eurobot::EurobotWorld& world, MCTS& mcts) {
  policy_->eval();
  world.reset();
  bool done = false;
  int step_idx = 0;

  std::vector<torch::Tensor> obs_list;
  std::vector<std::vector<float>> pi_list;

  while (!done) {
    auto root = mcts.search(world);

    std::vector<float> visits(action_dim_, 0.0f);
    int total_visits = 0;
    for (const auto& kv : root.children) {
      visits[(size_t)kv.first] = (float)kv.second->vis_count;
      total_visits += kv.second->vis_count;
    }
    if (total_visits == 0) break;
    for (float& v : visits) v /= std::max(1, total_visits);

    obs_list.push_back(world.obs());
    pi_list.push_back(visits);

    const double T = select_temperature(step_idx);
    const int action_idx = sample_action_from_visits(pi_list.back(), T);
    const auto& action = world.action_space[action_idx];
    done = world.step_blue(action) || step_idx > cfg_.max_env_steps;
    ++step_idx;
  }

  const auto scores = world.final_scores_norm();
  const float ret = scores.first;

  std::vector<Transition> traj;
  traj.reserve(obs_list.size());
  for (size_t i = 0; i < obs_list.size(); ++i) {
    auto piT = torch::from_blob(pi_list[i].data(), {(long)action_dim_}, torch::kFloat32).clone();
    traj.push_back({obs_list[i].detach().clone(), piT, ret});
  }
  return traj;
}


void AlphaZeroTrainer::optimize_step() {
  if (replay_.size() < static_cast<size_t>(cfg_.batch_size)) return;

  policy_->train();

  auto batch = replay_.sample(cfg_.batch_size, device_);
  // observations: [B, obs_dim], policies: [B, A], values: [B]
  TORCH_CHECK(batch.policies.size(1) == action_dim_, "policy target width != action space");

  // forward
  auto out = policy_->forward(batch.observations);   // {policy_logits: [B,A], value: [B] or [B,1]}
  auto logits = out.policy_logits;                   // [B, A]
  auto value  = out.value.view({-1});                // [B]

  // losses
  auto log_probs = torch::log_softmax(logits, /*dim=*/1);
  auto probs     = torch::softmax(logits,     /*dim=*/1);

  auto policy_loss = -(batch.policies * log_probs).sum(1).mean();
  auto value_loss  = torch::mse_loss(value, batch.values);

  auto entropy = -(probs * log_probs).sum(1).mean();

  auto loss = cfg_.policy_loss_weight * policy_loss
            + cfg_.value_loss_weight  * value_loss
            - cfg_.entropy_weight     * entropy;

  // step
  optimizer_.zero_grad();
  loss.backward();
  if (cfg_.max_grad_norm > 0.0) {
    torch::nn::utils::clip_grad_norm_(policy_->parameters(), cfg_.max_grad_norm);
  }
  optimizer_.step();

  if (tb_logger_) {
    const double loss_val = loss.detach().cpu().item<double>();
    const double policy_loss_val = policy_loss.detach().cpu().item<double>();
    const double value_loss_val  = value_loss.detach().cpu().item<double>();

    tb_logger_->add_scalar("train/loss", train_step_counter_, loss_val);
    tb_logger_->add_scalar("train/policy_loss", train_step_counter_, policy_loss_val);
    tb_logger_->add_scalar("train/value_loss", train_step_counter_, value_loss_val);
  }
  ++train_step_counter_;
}

void AlphaZeroTrainer::try_resume() {
  if (has_attempted_resume_) return;
  has_attempted_resume_ = true;
  if (!cfg_.resume_from_checkpoint) return;
  if (cfg_.checkpoint_path.empty()) {
    std::cerr << "[AlphaZeroTrainer] Resume requested but checkpoint_path is empty." << std::endl;
    return;
  }

  if (load_checkpoint(cfg_.checkpoint_path)) {
    resume_successful_ = true;
    last_checkpoint_iteration_ = iteration_counter_;
    std::cout << "[AlphaZeroTrainer] Resumed from " << cfg_.checkpoint_path
              << " (iteration=" << iteration_counter_
              << ", train_step=" << train_step_counter_ << ")." << std::endl;
  } else {
    std::cout << "[AlphaZeroTrainer] Resume requested but checkpoint '"
              << cfg_.checkpoint_path << "' not found or invalid." << std::endl;
  }
}

void AlphaZeroTrainer::maybe_save_checkpoint(bool force) {
  if (cfg_.checkpoint_path.empty()) return;
  if (!force) {
    if (cfg_.checkpoint_interval <= 0) return;
    if (iteration_counter_ == last_checkpoint_iteration_) return;
    if ((iteration_counter_ % cfg_.checkpoint_interval) != 0) return;
  }
  if (save_checkpoint(cfg_.checkpoint_path)) {
    last_checkpoint_iteration_ = iteration_counter_;
    std::cout << "[AlphaZeroTrainer] Checkpoint saved to "
              << cfg_.checkpoint_path << " (iteration=" << iteration_counter_ << ")." << std::endl;
  }
}

bool AlphaZeroTrainer::save_checkpoint(const std::string& path) const {
  if (path.empty()) return false;
  try {
    namespace fs = std::filesystem;
    const fs::path checkpoint_path(path);
    if (checkpoint_path.has_parent_path()) {
      fs::create_directories(checkpoint_path.parent_path());
    }

    torch::serialize::OutputArchive archive;

    torch::serialize::OutputArchive policy_archive;
    policy_->save(policy_archive);
    archive.write("policy", policy_archive);

    torch::serialize::OutputArchive optim_archive;
    optimizer_.save(optim_archive);
    archive.write("optimizer", optim_archive);

    std::vector<int64_t> counters_vec = {iteration_counter_, train_step_counter_};
    auto counters = torch::tensor(counters_vec, torch::TensorOptions().dtype(torch::kInt64));
    archive.write("counters", counters);

    torch::serialize::OutputArchive replay_archive;
    replay_.save(replay_archive);
    archive.write("replay", replay_archive);

    archive.save_to(path);
    return true;
  } catch (const c10::Error& e) {
    std::cerr << "[AlphaZeroTrainer] Failed to save checkpoint (" << path << "): "
              << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[AlphaZeroTrainer] Failed to save checkpoint (" << path << "): "
              << e.what() << std::endl;
  }
  return false;
}

bool AlphaZeroTrainer::load_checkpoint(const std::string& path) {
  if (path.empty()) return false;
  try {
    namespace fs = std::filesystem;
    if (!fs::exists(path)) {
      return false;
    }

    torch::serialize::InputArchive archive;
    archive.load_from(path);

    {
      torch::serialize::InputArchive policy_archive;
      archive.read("policy", policy_archive);
      policy_->load(policy_archive);
      policy_->to(device_);
    }

    {
      torch::serialize::InputArchive optim_archive;
      archive.read("optimizer", optim_archive);
      optimizer_.load(optim_archive);
    }

    try {
      torch::Tensor counters;
      archive.read("counters", counters);
      auto cpu = counters.to(torch::kCPU);
      if (cpu.numel() >= 2) {
        iteration_counter_ = cpu[0].item<int64_t>();
        train_step_counter_ = cpu[1].item<int64_t>();
      }
    } catch (const c10::Error&) {
      iteration_counter_ = 0;
      train_step_counter_ = 0;
    }

    try {
      torch::serialize::InputArchive replay_archive;
      archive.read("replay", replay_archive);
      replay_.load(replay_archive);
    } catch (const c10::Error& e) {
      std::cerr << "[AlphaZeroTrainer] Warning: replay buffer missing in checkpoint: "
                << e.what() << std::endl;
    }
    return true;
  } catch (const c10::Error& e) {
    std::cerr << "[AlphaZeroTrainer] Failed to load checkpoint (" << path << "): "
              << e.what() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[AlphaZeroTrainer] Failed to load checkpoint (" << path << "): "
              << e.what() << std::endl;
  }
  return false;
}

/* --------- helpers ---------- */

int AlphaZeroTrainer::sample_action_from_visits(const std::vector<float>& visits, double T) {
  // deterministic at T≈0
  if (T <= 1e-6) {
    return static_cast<int>(std::distance(
      visits.begin(),
      std::max_element(visits.begin(), visits.end())
    ));
  }
  // softmax over log(visits) / T
  std::vector<double> logits(visits.size());
  for (size_t i=0;i<visits.size();++i) logits[i] = std::log(std::max(1e-8f, visits[i])) / std::max(1e-6, T);
  const double mx = *std::max_element(logits.begin(), logits.end());
  double Z = 0.0;
  for (auto& x : logits) { x = std::exp(x - mx); Z += x; }
  for (auto& x : logits) x /= std::max(1e-12, Z);

  std::discrete_distribution<int> dist(logits.begin(), logits.end());
  static thread_local std::mt19937 rng{std::random_device{}()};
  return dist(rng);
}

double AlphaZeroTrainer::select_temperature(int step_idx) const {
  if (cfg_.temperature_decay_steps <= 0) return cfg_.temperature;
  return (step_idx < cfg_.temperature_decay_steps) ? cfg_.temperature : 1e-6;
}

} // namespace alphazero
