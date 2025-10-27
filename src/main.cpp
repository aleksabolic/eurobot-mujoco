#include <filesystem>
#include <iostream>
#include <type_traits>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "alphazero/trainer.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/policy_network.hpp"

using namespace alphazero;
using namespace eurobot;

int main() {
  // device
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  torch::manual_seed(42);

  const auto resolve_config_path = []() {
    namespace fs = std::filesystem;
    fs::path cursor = fs::current_path();
    for (int i = 0; i < 4; ++i) {
      const fs::path candidate = cursor / "configs/world.yaml";
      if (fs::exists(candidate)) {
        return candidate;
      }
      if (!cursor.has_parent_path()) {
        break;
      }
      cursor = cursor.parent_path();
    }
    throw std::runtime_error("Unable to locate configs/world.yaml");
  };

  const YAML::Node config = YAML::LoadFile(resolve_config_path().string());

  const auto load_profile = [](const YAML::Node& node) -> Profile {
    if (!node) {
      throw std::runtime_error("Missing profile configuration");
    }

    Profile p;
    p.capacity = node["capacity"].as<int>();
    p.max_action_qty = node["max_action_qty"].as<int>();
    p.can_flip = node["can_flip"].as<bool>();

    const auto travel = node["travel_time"];
    const double slope = travel && travel["slope"] ? travel["slope"].as<double>() : 1.0;
    const double bias = travel && travel["bias"] ? travel["bias"].as<double>() : 0.0;
    p.travel_time = [slope, bias](double d) { return bias + slope * d; };

    const auto handle = node["handle_time"];
    const auto read_coeff = [&](const char* key, double fallback) {
      return handle && handle[key] ? handle[key].as<double>() : fallback;
    };
    const double pick = read_coeff("PICK", 0.0);
    const double place = read_coeff("PLACE", 0.0);
    const double flip = read_coeff("FLIP", 0.0);
    const double steal = read_coeff("STEAL", 0.0);
    p.handle_time = [pick, place, flip, steal](Verb v, int q) {
      const double coeff = [&]() {
        switch (v) {
          case Verb::PICK:  return pick;
          case Verb::PLACE: return place;
          case Verb::FLIP:  return flip;
          case Verb::STEAL: return steal;
        }
        return 0.0;
      }();
      return coeff * q;
    };

    return p;
  };

  const auto profiles = config["profiles"];
  if (!profiles || !profiles["blue"] || !profiles["yellow"]) {
    throw std::runtime_error("profiles.blue/yellow missing in configs/world.yaml");
  }

  Profile blue_prof = load_profile(profiles["blue"]);
  Profile yellow_prof = load_profile(profiles["yellow"]);

  RewardConfig R{};
  if (const auto reward = config["reward"]) {
    const auto read_reward = [&](const char* key, float& target) {
      if (reward[key]) {
        target = reward[key].as<float>();
      }
    };
    read_reward("pantry_bonus", R.pantry_bonus);
    read_reward("nest_bonus", R.nest_bonus);
    read_reward("interest_bonus", R.interest_bonus);
    read_reward("finish_in_nest_bonus", R.finish_in_nest_bonus);
  }

  bool allow_steal = true;
  uint64_t seed = 42;
  bool record_history = false;
  if (const auto world_node = config["world"]) {
    allow_steal = world_node["allow_steal"] ? world_node["allow_steal"].as<bool>() : allow_steal;
    seed = world_node["seed"] ? world_node["seed"].as<uint64_t>() : seed;
    record_history = world_node["record_history"] ? world_node["record_history"].as<bool>() : record_history;
  }

  EurobotWorld world(blue_prof, yellow_prof, R, allow_steal, seed, record_history);

  // sizes
  const int64_t obs_dim = world.obs().size(0);
  const int64_t action_dim = static_cast<int64_t>(world.action_space().size());

  std::vector<int> hidden_sizes = {256, 256};
  if (const auto network_node = config["network"]) {
    const auto hs = network_node["hidden_sizes"];
    if (hs && hs.IsSequence() && hs.size() > 0) {
      hidden_sizes.clear();
      hidden_sizes.reserve(hs.size());
      for (const auto& entry : hs) {
        hidden_sizes.push_back(entry.as<int>());
      }
    }
  }

  PolicyNetworkOptions network_options;
  network_options.input_size = obs_dim;
  network_options.hidden_sizes = hidden_sizes;
  network_options.action_size = action_dim;

  auto net = PolicyNetwork(std::move(network_options));
  net->to(device);

  TrainingConfig cfg;
  if (const auto training = config["training"]) {
    const auto set_scalar = [&](const char* key, auto& target) {
      if (training[key]) {
        using T = std::decay_t<decltype(target)>;
        target = training[key].as<T>();
      }
    };
    set_scalar("num_iterations", cfg.num_iterations);
    set_scalar("games_per_iter", cfg.games_per_iter);
    set_scalar("training_steps", cfg.training_steps);
    set_scalar("batch_size", cfg.batch_size);
    set_scalar("replay_capacity", cfg.replay_capacity);
    set_scalar("num_simulations", cfg.num_simulations);
    set_scalar("cpuct", cfg.cpuct);
    set_scalar("dirichlet_alpha", cfg.dirichlet_alpha);
    set_scalar("dirichlet_epsilon", cfg.dirichlet_epsilon);
    set_scalar("learning_rate", cfg.learning_rate);
    set_scalar("weight_decay", cfg.weight_decay);
    set_scalar("max_grad_norm", cfg.max_grad_norm);
    set_scalar("policy_loss_weight", cfg.policy_loss_weight);
    set_scalar("value_loss_weight", cfg.value_loss_weight);
    set_scalar("entropy_weight", cfg.entropy_weight);
    set_scalar("temperature", cfg.temperature);
    set_scalar("temperature_decay_steps", cfg.temperature_decay_steps);
    set_scalar("enable_tensorboard", cfg.enable_tensorboard);
    set_scalar("log_dir", cfg.log_dir);
    set_scalar("run_name", cfg.run_name);
    set_scalar("checkpoint_path", cfg.checkpoint_path);
    set_scalar("checkpoint_interval", cfg.checkpoint_interval);
    set_scalar("resume_from_checkpoint", cfg.resume_from_checkpoint);
  }

  if (cfg.checkpoint_path.empty() && !cfg.log_dir.empty()) {
    const std::string run_token = cfg.run_name.empty() ? "alphazero" : cfg.run_name;
    cfg.checkpoint_path = cfg.log_dir + "/" + run_token + "_checkpoint.pt";
  }

  // ---- Trainer ----
  AlphaZeroTrainer trainer(net, world, cfg);
  trainer.set_device(device);

  std::cout << "device: " << (device.is_cuda() ? "CUDA" : "CPU")
            << ", obs_dim=" << obs_dim
            << ", action_dim=" << action_dim << std::endl;

  trainer.train();

  // save checkpoint
  torch::save(net, "runs/alphazero_policy.pt");

  return 0;
}
