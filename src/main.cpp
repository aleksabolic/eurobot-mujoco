#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "alphazero/trainer.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/policy_network.hpp"

namespace az = alphazero;
namespace eu = eurobot;

// ---------------------------- YAML helpers ----------------------------
template <typename T>
inline void assign_if(const YAML::Node& n, const char* key, T& out) {
  if (n && n[key]) out = n[key].as<std::decay_t<T>>();
}

template <typename T>
inline T get_or(const YAML::Node& n, const char* key, T def) {
  return (n && n[key]) ? n[key].as<T>() : def;
}

inline std::filesystem::path pick_config_path(int argc, char** argv) {
  namespace fs = std::filesystem;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--config" || a == "-c") {
      if (i + 1 >= argc)
        throw std::runtime_error("--config requires a file path");
      fs::path p = argv[i + 1];
      if (!fs::exists(p))
        throw std::runtime_error("Config file not found: " + p.string());
      return fs::canonical(p);
    }
    const std::string prefix = "--config=";
    if (a.rfind(prefix, 0) == 0) {
      fs::path p = a.substr(prefix.size());
      if (!fs::exists(p))
        throw std::runtime_error("Config file not found: " + p.string());
      return fs::canonical(p);
    }
  }

  throw std::runtime_error("Missing required argument: --config <path>");
}

// ---------------------------- domain builders ----------------------------
inline eu::RobotProfile load_profile(const YAML::Node& node) {
  if (!node) throw std::runtime_error("profile node missing");

  eu::RobotProfile p{};
  p.capacity        = node["capacity"].as<int>();
  p.max_action_qty  = node["max_action_qty"].as<int>();
  p.can_flip        = node["can_flip"].as<bool>();

  const float v_max = get_or<float>(node, "v_max", 0.2f);
  const float a_max = get_or<float>(node, "a_max", 1.0f);

  // trapezoid motion model + start/stop overhead
  p.travel_time = [v_max, a_max](double d) {
    const float d_min = (v_max * v_max) / a_max;
    double t_motion = (d <= d_min + 1e-12)
        ? 2.0 * std::sqrt(d / a_max)
        : 2.0 * (v_max / a_max) + (d - d_min) / v_max;
    return 0.4 + t_motion;
  };

  const YAML::Node h = node["handle_time"];
  const double pick  = get_or<double>(h, "PICK",  0.0);
  const double place = get_or<double>(h, "PLACE", 0.0);
  const double flip  = get_or<double>(h, "FLIP",  0.0);
  const double steal = get_or<double>(h, "STEAL", 0.0);

  p.handle_time = [pick, place, flip, steal](eu::Verb v, int q) {
    const double k = (v == eu::Verb::PICK)  ? pick  :
                     (v == eu::Verb::PLACE) ? place :
                     (v == eu::Verb::FLIP)  ? flip  :
                                              steal;
    return k * q;
  };
  return p;
}

inline eu::RewardConfig load_reward(const YAML::Node& root) {
  eu::RewardConfig r{};
  const YAML::Node n = root["reward"];
  assign_if(n, "pantry_bonus",         r.pantry_bonus);
  assign_if(n, "nest_bonus",           r.nest_bonus);
  assign_if(n, "interest_bonus",       r.interest_bonus);
  assign_if(n, "finish_in_nest_bonus", r.finish_in_nest_bonus);
  return r;
}

inline eu::EurobotWorld make_world(const YAML::Node& root) {
  const YAML::Node profiles = root["profiles"];
  if (!profiles || !profiles["blue"] || !profiles["yellow"])
    throw std::runtime_error("profiles.blue/yellow missing");

  auto blue   = load_profile(profiles["blue"]);
  auto yellow = load_profile(profiles["yellow"]);
  auto R      = load_reward(root);

  const YAML::Node w = root["world"];
  const bool allow_steal   = get_or<bool>(w, "allow_steal", true);
  const uint64_t seed      = get_or<uint64_t>(w, "seed", 42ULL);
  const bool record_hist   = get_or<bool>(w, "record_history", false);

  return eu::EurobotWorld(blue, yellow, R, allow_steal, seed, record_hist);
}

inline std::vector<int> load_hidden_sizes(const YAML::Node& root,
                                          std::vector<int> def = {256, 256}) {
  const YAML::Node n = root["network"]["hidden_sizes"];
  if (!n || !n.IsSequence() || n.size() == 0) return def;
  std::vector<int> hs;
  hs.reserve(n.size());
  for (const auto& x : n) hs.push_back(x.as<int>());
  return hs;
}

inline az::PolicyNetwork make_network(const YAML::Node& root,
                                      int64_t obs_dim,
                                      int64_t act_dim,
                                      const torch::Device& device) {
  az::PolicyNetworkOptions opt;
  opt.input_size   = obs_dim;
  opt.hidden_sizes = load_hidden_sizes(root);
  opt.action_size  = act_dim;

  az::PolicyNetwork net(std::move(opt));
  net->to(device);
  return net;
}

inline az::TrainingConfig load_training(const YAML::Node& root) {
  az::TrainingConfig cfg{};
  const YAML::Node node = root["training"];
  if (!node) return cfg;

  assign_if(node, "num_iterations",          cfg.num_iterations);
  assign_if(node, "games_per_iter",          cfg.games_per_iter);
  assign_if(node, "training_steps",          cfg.training_steps);
  assign_if(node, "batch_size",              cfg.batch_size);
  assign_if(node, "replay_capacity",         cfg.replay_capacity);
  assign_if(node, "num_simulations",         cfg.num_simulations);
  assign_if(node, "max_env_steps",           cfg.max_env_steps);
  assign_if(node, "cpuct",                   cfg.cpuct);
  assign_if(node, "dirichlet_alpha",         cfg.dirichlet_alpha);
  assign_if(node, "dirichlet_epsilon",       cfg.dirichlet_epsilon);
  assign_if(node, "gamma",                   cfg.gamma);
  assign_if(node, "learning_rate",           cfg.learning_rate);
  assign_if(node, "weight_decay",            cfg.weight_decay);
  assign_if(node, "max_grad_norm",           cfg.max_grad_norm);
  assign_if(node, "policy_loss_weight",      cfg.policy_loss_weight);
  assign_if(node, "value_loss_weight",       cfg.value_loss_weight);
  assign_if(node, "entropy_weight",          cfg.entropy_weight);
  assign_if(node, "temperature",             cfg.temperature);
  assign_if(node, "temperature_decay_steps", cfg.temperature_decay_steps);
  assign_if(node, "enable_tensorboard",      cfg.enable_tensorboard);
  assign_if(node, "log_dir",                 cfg.log_dir);
  assign_if(node, "run_name",                cfg.run_name);
  assign_if(node, "checkpoint_path",         cfg.checkpoint_path);
  assign_if(node, "checkpoint_interval",     cfg.checkpoint_interval);
  assign_if(node, "resume_from_checkpoint",  cfg.resume_from_checkpoint);

  if (cfg.checkpoint_path.empty() && !cfg.log_dir.empty()) {
    const std::string token = cfg.run_name.empty() ? "alphazero" : cfg.run_name;
    cfg.checkpoint_path = cfg.log_dir + "/" + token + "_checkpoint.pt";
  }
  return cfg;
}

int main(int argc, char** argv) {
  const torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  torch::manual_seed(42);

  const YAML::Node cfg_yaml = YAML::LoadFile(pick_config_path(argc, argv).string());
  eu::EurobotWorld world = make_world(cfg_yaml);

  const int64_t obs_dim    = world.obs().size(0);
  const int64_t action_dim = static_cast<int64_t>(world.action_space.size());

  az::PolicyNetwork net = make_network(cfg_yaml, obs_dim, action_dim, device);
  az::TrainingConfig train_cfg = load_training(cfg_yaml);

  az::AlphaZeroTrainer trainer(net, world, train_cfg);
  trainer.set_device(device);

  std::cout << "Device=" << (device.is_cuda() ? "CUDA" : "CPU")
            << " | obs=" << obs_dim
            << " | act=" << action_dim << '\n';

  trainer.train();
  torch::save(net, "runs/alphazero_policy.pt");
  return 0;
}
