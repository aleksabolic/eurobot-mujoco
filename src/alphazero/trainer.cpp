#include "alphazero/trainer.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace alphazero {

namespace {
torch::optim::AdamOptions make_adam_options(const TrainingConfig& cfg) {
  torch::optim::AdamOptions opt(cfg.learning_rate);
  opt.weight_decay(cfg.weight_decay);
  return opt;
}
// TODO: move this to ctor
MCTSConfig make_mcts_cfg(const TrainingConfig& cfg) {
  MCTSConfig m;
  m.num_simulations   = cfg.num_simulations;
  m.cpuct             = cfg.cpuct;
  m.dirichlet_alpha   = cfg.dirichlet_alpha;
  m.dirichlet_epsilon = cfg.dirichlet_epsilon;
  return m;
}
} // namespace

AlphaZeroTrainer::AlphaZeroTrainer(PolicyNetwork network,
                                   eurobot::EurobotWorld world,
                                   const TrainingConfig& cfg)
  : cfg_(cfg),
    policy_(std::move(network)),
    world_(std::move(world)),
    action_helper_(world_),
    replay_(cfg_.replay_capacity),
    mcts_(make_mcts_cfg(cfg_), policy_, action_helper_),
    optimizer_(policy_->parameters(), make_adam_options(cfg_)) {

  TORCH_CHECK(policy_, "AlphaZeroTrainer: policy network must be non-null");
  action_dim_ = static_cast<int64_t>(action_helper_.actions().size());
}

void AlphaZeroTrainer::set_device(const torch::Device& device) {
  device_ = device;
  policy_->to(device_);
}

void AlphaZeroTrainer::train() {
  for (int it = 0; it < cfg_.num_iterations; ++it) {
    std::cout<< "Iteration num: " << it << std::endl;
    for (int g = 0; g < cfg_.games_per_iter; ++g) {
      play_episode();
    }
    for (int s = 0; s < cfg_.training_steps; ++s) {
      optimize_step();
    }
    // TODO: checkpoint (torch::save policy_->named_parameters(), etc.)
  }
}

void AlphaZeroTrainer::play_episode() {
  policy_->eval();

  world_.reset();                 // restart match
  bool done = false;
  int step_idx = 0;

  std::vector<torch::Tensor> obs_list;
  std::vector<std::vector<float>> pi_list;

  while (!done) {
    // snapshot state (so MCTS can restore per sim)
    const auto root_state = world_.get_state();

    // run MCTS from current state
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
      auto legal = action_helper_.legal_mask(world_);
      auto legal_cpu = legal.to(torch::kCPU);
      visits.assign(action_dim_, 0.0f);
      for (int64_t i = 0; i < legal_cpu.numel(); ++i)
        if (legal_cpu[i].item<bool>()) visits[(size_t)i] = 1.0f;
      total_visits = static_cast<int>(std::accumulate(visits.begin(), visits.end(), 0.0f));
      if (total_visits == 0) return; // degenerate
    }
    // normalize
    for (float& v : visits) v = v / std::max(1, total_visits);

    // record obs + policy target
    obs_list.push_back(world_.obs());   // 1D tensor
    pi_list.push_back(visits);

    // select action with temperature schedule
    const double T = select_temperature(step_idx);
    const int action_idx = sample_action_from_visits(visits, T);
    const auto& action = action_helper_.action_by_index(action_idx);

    // step environment (Blue)
    done = world_.step_blue(action);

    ++step_idx;
  }

  // terminal value (normalized blue score)
  const auto scores = world_.final_scores_norm(); // pair<float,float>
  const float ret = scores.first;

  // commit episode to replay
  for (size_t i = 0; i < obs_list.size(); ++i) {
    // obs: 1D float; pi: 1D float (A); value: scalar float
    auto piT = torch::from_blob(pi_list[i].data(), { (long)action_dim_ }, torch::kFloat32).clone();
    replay_.add(obs_list[i].detach().clone(), piT, ret);
  }
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
