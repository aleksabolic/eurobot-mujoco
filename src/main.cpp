#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "alphazero/trainer.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/policy_network.hpp"
#include "alphazero/yellow_policies.hpp"

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

inline std::string to_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

inline eu::Verb parse_verb_field(const YAML::Node& node) {
  if (!node || !node.IsScalar())
    throw std::runtime_error("yellow_policy.script entries require a 'verb' scalar");
  const std::string value = to_lower_copy(node.as<std::string>());
  if (value == "pick")  return eu::Verb::PICK;
  if (value == "place") return eu::Verb::PLACE;
  if (value == "flip")  return eu::Verb::FLIP;
  if (value == "steal") return eu::Verb::STEAL;
  throw std::runtime_error("Unknown verb in yellow_policy.script: " + node.as<std::string>());
}

inline int parse_color_field(const YAML::Node& node, int default_value) {
  if (!node) return default_value;
  if (!node.IsScalar())
    throw std::runtime_error("yellow_policy.script color must be a scalar value");
  const std::string raw = node.as<std::string>();
  const std::string value = to_lower_copy(raw);
  if (value == "blue" || value == "b")   return static_cast<int>(eu::Col::BLUE);
  if (value == "yellow" || value == "y") return static_cast<int>(eu::Col::YELLOW);
  try {
    size_t idx = 0;
    const int parsed = std::stoi(raw, &idx);
    if (idx == raw.size()) return parsed;
  } catch (const std::exception&) {
    // fallthrough to throw
  }
  throw std::runtime_error("Unknown color in yellow_policy.script: " + raw);
}

inline int resolve_node_id(const YAML::Node& node, const eu::EurobotWorld& world) {
  if (!node)
    throw std::runtime_error("yellow_policy.script entry missing required 'node'");
  if (!node.IsScalar())
    throw std::runtime_error("yellow_policy.script node must be a string or integer");
  const std::string raw = node.as<std::string>();
  const std::string key = to_lower_copy(raw);
  const auto& nodes = world.nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::string name = to_lower_copy(nodes[i].name);
    if (name == key) return static_cast<int>(i);
  }
  try {
    size_t idx = 0;
    const int parsed = std::stoi(raw, &idx);
    if (idx == raw.size()) {
      if (parsed < 0 || parsed >= static_cast<int>(world.nodes().size()))
        throw std::runtime_error("yellow_policy.script node index out of range: " + raw);
      return parsed;
    }
  } catch (const std::exception&) {
    // fallthrough
  }
  throw std::runtime_error("Unknown node in yellow_policy.script: " + raw);
}

inline void validate_node_for_verb(eu::Verb verb, const eu::Node& node) {
  switch (verb) {
    case eu::Verb::PICK:
      if (node.kind != eu::NodeType::PICKUP)
        throw std::runtime_error("yellow_policy.script PICK must target a pickup node");
      break;
    case eu::Verb::STEAL:
      if (node.kind != eu::NodeType::PANTRY)
        throw std::runtime_error("yellow_policy.script STEAL must target a pantry node");
      break;
    case eu::Verb::PLACE:
      if (node.kind != eu::NodeType::PANTRY && node.kind != eu::NodeType::NEST)
        throw std::runtime_error("yellow_policy.script PLACE must target a pantry or nest");
      break;
    case eu::Verb::FLIP:
      // any node is fine; action happens in-place
      break;
  }
}

inline eu::Action parse_script_action(const YAML::Node& node,
                                      const eu::EurobotWorld& world) {
  if (!node || !node.IsMap())
    throw std::runtime_error("yellow_policy.script entries must be maps");
  eu::Action action{};
  const eu::Verb verb = parse_verb_field(node["verb"]);
  action.verb = static_cast<int>(verb);

  if (verb == eu::Verb::FLIP && !node["node"]) {
    if (world.yellow_nest_node_idx < 0)
      throw std::runtime_error("yellow nest is undefined, cannot infer node for FLIP");
    action.node = world.yellow_nest_node_idx;
  } else {
    action.node = resolve_node_id(node["node"], world);
  }
  if (action.node < 0 || action.node >= static_cast<int>(world.nodes().size()))
    throw std::runtime_error("yellow_policy.script node index out of bounds");
  validate_node_for_verb(verb, world.nodes().at(action.node));

  const int default_color =
      (verb == eu::Verb::FLIP) ? static_cast<int>(eu::Col::BLUE)
                               : static_cast<int>(eu::Col::YELLOW);
  action.color = parse_color_field(node["color"], default_color);

  const int qty = node["qty"] ? node["qty"].as<int>() : 1;
  if (qty <= 0)
    throw std::runtime_error("yellow_policy.script qty must be positive");
  action.qty = qty;

  return action;
}

inline void configure_yellow_policy(const YAML::Node& node, eu::EurobotWorld& world) {
  std::string type_value = "heuristic";
  if (node) {
    if (node.IsScalar()) {
      type_value = to_lower_copy(node.as<std::string>());
    } else if (node["type"]) {
      type_value = to_lower_copy(node["type"].as<std::string>());
    }
  }

  if (type_value == "none") {
    world.yellow_policy = {};
    return;
  }

  if (type_value == "heuristic") {
    eu::HeuristicYellowPolicyOptions opts{};
    if (node && node.IsMap()) {
      assign_if(node, "enable_steal", opts.enable_steal);
      assign_if(node, "enable_flip", opts.enable_flip);
      assign_if(node, "min_pickup_stock", opts.min_pickup_stock);
      assign_if(node, "min_steal_stock", opts.min_steal_stock);
      assign_if(node, "flip_threshold", opts.flip_threshold);
    }
    world.yellow_policy = eu::make_heuristic_yellow_policy(opts);
    return;
  }

  if (type_value == "static" || type_value == "static_script" || type_value == "script") {
    const YAML::Node seq = (node && node.IsMap()) ? node["script"] : YAML::Node{};
    if (!seq || !seq.IsSequence() || seq.size() == 0)
      throw std::runtime_error("yellow_policy.script must be a non-empty sequence");
    eu::StaticYellowPolicyOptions opts{};
    opts.loop = node && node.IsMap() ? get_or<bool>(node, "loop", true) : true;
    opts.script.reserve(seq.size());
    for (const auto& entry : seq) {
      opts.script.push_back(parse_script_action(entry, world));
    }
    world.yellow_policy = eu::make_static_yellow_policy(std::move(opts));
    return;
  }

  throw std::runtime_error("Unknown yellow_policy.type: " + type_value);
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
  p.max_flip_qty    = get_or<int>(node, "max_flip_qty", p.max_action_qty);
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

  eu::EurobotWorld world(blue, yellow, R, allow_steal, seed, record_hist);
  configure_yellow_policy(root["yellow_policy"], world);
  return world;
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
