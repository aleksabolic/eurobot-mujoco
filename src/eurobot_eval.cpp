#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/serialize/archive.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "alphazero/eurobot_render.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/policy_network.hpp"
#include "alphazero/yellow_policies.hpp"

namespace {

struct EvalOptions {
  std::filesystem::path config_path;
  std::string model_path;
  std::string video_path;
  int episodes = 1;
  double fps = 1.0;
  bool show_window = false;
};

void print_usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --model <path> [--config FILE] [--episodes N] [--fps FPS]"
            << " [--video PATH] [--show]\n";
}

EvalOptions parse_args(int argc, char** argv) {
  EvalOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      opts.model_path = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      opts.config_path = argv[++i];
    } else if (arg == "--episodes" && i + 1 < argc) {
      opts.episodes = std::max(1, std::atoi(argv[++i]));
    } else if (arg == "--fps" && i + 1 < argc) {
      opts.fps = std::max(1e-3, std::atof(argv[++i]));
    } else if (arg == "--video" && i + 1 < argc) {
      opts.video_path = argv[++i];
    } else if (arg == "--show") {
      opts.show_window = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.model_path.empty()) {
    print_usage(argv[0]);
    std::exit(1);
  }

  return opts;
}

std::filesystem::path resolve_config_path() {
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
}

std::filesystem::path pick_config_path(const EvalOptions& opts) {
  namespace fs = std::filesystem;
  if (opts.config_path.empty()) {
    return resolve_config_path();
  }
  if (!fs::exists(opts.config_path)) {
    throw std::runtime_error("Config file not found: " + opts.config_path.string());
  }
  return fs::canonical(opts.config_path);
}

template <typename T>
void assign_if(const YAML::Node& n, const char* key, T& out) {
  if (n && n[key]) {
    out = n[key].as<std::decay_t<T>>();
  }
}

template <typename T>
T get_or(const YAML::Node& n, const char* key, T def) {
  return (n && n[key]) ? n[key].as<T>() : def;
}

std::string to_lower_copy(std::string s) {
  std::transform(
      s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

eurobot::Verb parse_verb_field(const YAML::Node& node) {
  if (!node || !node.IsScalar()) {
    throw std::runtime_error("yellow_policy.script entries require a 'verb' scalar");
  }
  const std::string value = to_lower_copy(node.as<std::string>());
  if (value == "pick") return eurobot::Verb::PICK;
  if (value == "place") return eurobot::Verb::PLACE;
  if (value == "flip") return eurobot::Verb::FLIP;
  if (value == "steal") return eurobot::Verb::STEAL;
  throw std::runtime_error("Unknown verb in yellow_policy.script: " + node.as<std::string>());
}

int parse_color_field(const YAML::Node& node, int default_value) {
  if (!node) return default_value;
  if (!node.IsScalar()) {
    throw std::runtime_error("yellow_policy.script color must be a scalar");
  }
  const std::string raw = node.as<std::string>();
  const std::string value = to_lower_copy(raw);
  if (value == "blue" || value == "b") return static_cast<int>(eurobot::Col::BLUE);
  if (value == "yellow" || value == "y") return static_cast<int>(eurobot::Col::YELLOW);
  try {
    size_t idx = 0;
    const int parsed = std::stoi(raw, &idx);
    if (idx == raw.size()) return parsed;
  } catch (const std::exception&) {
    // ignored
  }
  throw std::runtime_error("Unknown color in yellow_policy.script: " + raw);
}

int resolve_node_id(const YAML::Node& node, const eurobot::EurobotWorld& world) {
  if (!node) {
    throw std::runtime_error("yellow_policy.script entry missing required 'node'");
  }
  if (!node.IsScalar()) {
    throw std::runtime_error("yellow_policy.script node must be a string or integer");
  }
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
      if (parsed < 0 || parsed >= static_cast<int>(nodes.size())) {
        throw std::runtime_error("yellow_policy.script node index out of range: " + raw);
      }
      return parsed;
    }
  } catch (const std::exception&) {
    // fallthrough
  }
  throw std::runtime_error("Unknown node in yellow_policy.script: " + raw);
}

void validate_node_for_verb(eurobot::Verb verb, const eurobot::Node& node) {
  switch (verb) {
    case eurobot::Verb::PICK:
      if (node.kind != eurobot::NodeType::PICKUP) {
        throw std::runtime_error("yellow_policy.script PICK must target a pickup node");
      }
      break;
    case eurobot::Verb::STEAL:
      if (node.kind != eurobot::NodeType::PANTRY) {
        throw std::runtime_error("yellow_policy.script STEAL must target a pantry node");
      }
      break;
    case eurobot::Verb::PLACE:
      if (node.kind != eurobot::NodeType::PANTRY && node.kind != eurobot::NodeType::NEST) {
        throw std::runtime_error("yellow_policy.script PLACE must target a pantry or nest");
      }
      break;
    case eurobot::Verb::FLIP:
      break;
  }
}

eurobot::Action parse_script_action(const YAML::Node& node, const eurobot::EurobotWorld& world) {
  if (!node || !node.IsMap()) {
    throw std::runtime_error("yellow_policy.script entries must be maps");
  }
  eurobot::Action action{};
  const eurobot::Verb verb = parse_verb_field(node["verb"]);
  action.verb = static_cast<int>(verb);

  if (verb == eurobot::Verb::FLIP && !node["node"]) {
    if (world.yellow_nest_node_idx < 0) {
      throw std::runtime_error("yellow nest undefined, cannot infer node for FLIP");
    }
    action.node = world.yellow_nest_node_idx;
  } else {
    action.node = resolve_node_id(node["node"], world);
  }
  if (action.node < 0 || action.node >= static_cast<int>(world.nodes().size())) {
    throw std::runtime_error("yellow_policy.script node index out of bounds");
  }
  validate_node_for_verb(verb, world.nodes().at(action.node));

  const int default_color =
      (verb == eurobot::Verb::FLIP) ? static_cast<int>(eurobot::Col::BLUE)
                                    : static_cast<int>(eurobot::Col::YELLOW);
  action.color = parse_color_field(node["color"], default_color);

  const int qty = node["qty"] ? node["qty"].as<int>() : 1;
  if (qty <= 0) {
    throw std::runtime_error("yellow_policy.script qty must be positive");
  }
  action.qty = qty;

  return action;
}

void configure_yellow_policy(const YAML::Node& node, eurobot::EurobotWorld& world) {
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
    eurobot::HeuristicYellowPolicyOptions opts{};
    if (node && node.IsMap()) {
      assign_if(node, "enable_steal", opts.enable_steal);
      assign_if(node, "enable_flip", opts.enable_flip);
      assign_if(node, "min_pickup_stock", opts.min_pickup_stock);
      assign_if(node, "min_steal_stock", opts.min_steal_stock);
      assign_if(node, "flip_threshold", opts.flip_threshold);
    }
    world.yellow_policy = eurobot::make_heuristic_yellow_policy(opts);
    return;
  }

  if (type_value == "static" || type_value == "static_script" || type_value == "script") {
    const YAML::Node seq = (node && node.IsMap()) ? node["script"] : YAML::Node{};
    if (!seq || !seq.IsSequence() || seq.size() == 0) {
      throw std::runtime_error("yellow_policy.script must be a non-empty sequence");
    }
    eurobot::StaticYellowPolicyOptions opts{};
    opts.loop = node && node.IsMap() ? get_or<bool>(node, "loop", true) : true;
    opts.script.reserve(seq.size());
    for (const auto& entry : seq) {
      opts.script.push_back(parse_script_action(entry, world));
    }
    world.yellow_policy = eurobot::make_static_yellow_policy(std::move(opts));
    return;
  }

  throw std::runtime_error("Unknown yellow_policy.type: " + type_value);
}

alphazero::PolicyNetworkOptions make_network_options(const YAML::Node& config, int64_t input_size, int64_t action_size) {
  alphazero::PolicyNetworkOptions opts;
  opts.input_size = input_size;
  opts.action_size = action_size;
  opts.hidden_sizes = {256, 256};
  if (const auto network_node = config["network"]) {
    const auto hs = network_node["hidden_sizes"];
    if (hs && hs.IsSequence() && hs.size() > 0) {
      opts.hidden_sizes.clear();
      opts.hidden_sizes.reserve(hs.size());
      for (const auto& entry : hs) {
        opts.hidden_sizes.push_back(entry.as<int>());
      }
    }
  }
  return opts;
}

alphazero::PolicyNetwork load_policy(const EvalOptions& opts,
                                     const YAML::Node& config,
                                     const torch::Device& device,
                                     int64_t obs_dim,
                                     int64_t action_dim) {
  auto net_opts = make_network_options(config, obs_dim, action_dim);
  auto net = alphazero::PolicyNetwork(std::move(net_opts));
  bool loaded = false;
  try {
    torch::load(net, opts.model_path);
    loaded = true;
  } catch (const c10::Error&) {
    // fall through to checkpoint loader
  }

  if (!loaded) {
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(opts.model_path);
      torch::serialize::InputArchive policy_archive;
      archive.read("policy", policy_archive);
      net->load(policy_archive);
      loaded = true;
    } catch (const c10::Error& e) {
      std::cerr << "Failed to load policy from checkpoint '" << opts.model_path
                << "': " << e.what() << std::endl;
      throw;
    }
  }

  net->to(device);
  net->eval();
  return net;
}

std::filesystem::path make_video_path(const std::filesystem::path& base, int episode_idx, int total_episodes) {
  if (base.empty()) {
    return {};
  }
  std::filesystem::path stem = base;
  std::string ext = stem.extension().string();
  if (ext.empty()) {
    ext = ".gif";
  }

  stem.replace_extension();  // drop extension
  if (total_episodes > 1) {
    stem += "_ep" + std::to_string(episode_idx + 1);
  }
  stem += ext;
  return stem;
}

bool is_gif_path(const std::filesystem::path& p) {
  auto ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".gif";
}

std::string quote_path(const std::filesystem::path& p) {
  return "\"" + p.string() + "\"";
}

struct GifRecorder {
  std::filesystem::path tmp_dir;
  double fps = 1.0;
  int frame_idx = 0;
  bool ready = false;

  explicit GifRecorder(double fps_val) : fps(std::max(1e-3, fps_val)) {}

  bool start() {
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "eurobot_gif_frames";
    for (int attempt = 0; attempt < 10; ++attempt) {
      auto candidate = base;
      candidate += "_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count() + attempt);
      std::error_code ec;
      if (fs::create_directories(candidate, ec) && !ec) {
        tmp_dir = candidate;
        ready = true;
        frame_idx = 0;
        return true;
      }
    }
    return false;
  }

  bool add_frame(const cv::Mat& frame) {
    if (!ready) return false;
    namespace fs = std::filesystem;
    std::ostringstream name;
    name << "frame_" << std::setfill('0') << std::setw(5) << frame_idx << ".png";
    auto path = tmp_dir / name.str();
    ++frame_idx;
    return cv::imwrite(path.string(), frame);
  }

  bool finalize(const std::filesystem::path& output_path) {
    namespace fs = std::filesystem;
    if (!ready) return false;
    if (frame_idx == 0) {
      fs::remove_all(tmp_dir);
      ready = false;
      return false;
    }

    const auto pattern = tmp_dir / "frame_%05d.png";
    const auto palette = tmp_dir / "palette.png";

    std::ostringstream fps_stream;
    fps_stream << std::setprecision(6) << fps;
    const std::string fps_arg = fps_stream.str();

    std::string cmd_palette = "ffmpeg -y -loglevel error -framerate " + fps_arg +
                              " -i " + quote_path(pattern) +
                              " -vf palettegen " + quote_path(palette);
    std::string cmd_gif = "ffmpeg -y -loglevel error -framerate " + fps_arg +
                          " -i " + quote_path(pattern) +
                          " -i " + quote_path(palette) +
                          " -lavfi paletteuse " + quote_path(output_path);

    const int ret1 = std::system(cmd_palette.c_str());
    const int ret2 = (ret1 == 0) ? std::system(cmd_gif.c_str()) : -1;

    fs::remove_all(tmp_dir);
    ready = false;
    return ret1 == 0 && ret2 == 0;
  }
};

eurobot::RobotProfile load_profile(const YAML::Node& node) {
  if (!node) {
    throw std::runtime_error("Missing profile configuration");
  }

  eurobot::RobotProfile p{};
  p.capacity = node["capacity"].as<int>();
  p.max_action_qty = node["max_action_qty"].as<int>();
  p.max_flip_qty = get_or<int>(node, "max_flip_qty", p.max_action_qty);
  p.can_flip = node["can_flip"].as<bool>();

  const float v_max = get_or<float>(node, "v_max", 0.2f);
  const float a_max = get_or<float>(node, "a_max", 1.0f);

  p.travel_time = [v_max, a_max](double d) {
    const float d_min = (v_max * v_max) / a_max;
    double t_motion =
        (d <= d_min + 1e-12) ? 2.0 * std::sqrt(d / a_max)
                             : 2.0 * (v_max / a_max) + (d - d_min) / v_max;
    return 0.4 + t_motion;
  };

  const YAML::Node handle = node["handle_time"];
  const double pick = get_or<double>(handle, "PICK", 0.0);
  const double place = get_or<double>(handle, "PLACE", 0.0);
  const double flip = get_or<double>(handle, "FLIP", 0.0);
  const double steal = get_or<double>(handle, "STEAL", 0.0);

  p.handle_time = [pick, place, flip, steal](eurobot::Verb v, int q) {
    const double coeff = (v == eurobot::Verb::PICK)
                             ? pick
                             : (v == eurobot::Verb::PLACE)
                                   ? place
                                   : (v == eurobot::Verb::FLIP) ? flip : steal;
    return coeff * q;
  };
  return p;
}

eurobot::RewardConfig load_reward(const YAML::Node& root) {
  eurobot::RewardConfig rewards{};
  const YAML::Node node = root["reward"];
  assign_if(node, "pantry_bonus", rewards.pantry_bonus);
  assign_if(node, "nest_bonus", rewards.nest_bonus);
  assign_if(node, "interest_bonus", rewards.interest_bonus);
  assign_if(node, "finish_in_nest_bonus", rewards.finish_in_nest_bonus);
  return rewards;
}

eurobot::EurobotWorld make_world(const YAML::Node& config) {
  const auto profiles = config["profiles"];
  if (!profiles || !profiles["blue"] || !profiles["yellow"]) {
    throw std::runtime_error("profiles.blue/yellow missing in configs/world.yaml");
  }

  eurobot::RobotProfile blue_prof = load_profile(profiles["blue"]);
  eurobot::RobotProfile yellow_prof = load_profile(profiles["yellow"]);
  eurobot::RewardConfig rewards = load_reward(config);

  const YAML::Node world_node = config["world"];
  const bool allow_steal = get_or<bool>(world_node, "allow_steal", true);
  const uint64_t seed = get_or<uint64_t>(
      world_node, "seed",
      static_cast<uint64_t>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  const bool record_history = get_or<bool>(world_node, "record_history", false);

  eurobot::EurobotWorld world(blue_prof, yellow_prof, rewards, allow_steal, seed, record_history);
  configure_yellow_policy(config["yellow_policy"], world);
  return world;
}

int select_action(const eurobot::EurobotWorld& world, const torch::Tensor& logits) {
  auto logit_mask = world.logit_mask().to(logits.device());
  auto masked_logits = logits + logit_mask;
  masked_logits = torch::where(
      torch::isfinite(masked_logits), masked_logits,
      torch::full_like(masked_logits, -1e9f));

  auto legal_mask = world.legal_mask().to(torch::kCPU);
  if (!legal_mask.any().item<bool>()) {
    throw std::runtime_error("World reports no legal actions for the current state");
  }

  auto action_idx = masked_logits.argmax().item<int64_t>();
  if (!legal_mask[action_idx].item<bool>()) {
    auto legal_indices = legal_mask.nonzero();
    action_idx = legal_indices[0].item<int64_t>();
  }
  return static_cast<int>(action_idx);
}

}  // namespace

int main(int argc, char** argv) {
  EvalOptions opts = parse_args(argc, argv);
  const std::filesystem::path config_path = pick_config_path(opts);

  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

  const YAML::Node config = YAML::LoadFile(config_path.string());
  eurobot::EurobotWorld world = make_world(config);

  const int64_t obs_dim = world.obs().size(0);
  const int64_t action_dim = static_cast<int64_t>(world.action_space.size());

  auto policy = load_policy(opts, config, device, obs_dim, action_dim);

  eurobot::EurobotCV2Renderer renderer(cv::Size(1000, 700));
  renderer.set_window_title("Eurobot Evaluation");

  for (int ep = 0; ep < opts.episodes; ++ep) {
    world.reset();

    std::filesystem::path video_path;
    if (!opts.video_path.empty()) {
      video_path = make_video_path(opts.video_path, ep, opts.episodes);
    }

    cv::VideoWriter writer;
    bool video_ready = false;

    std::optional<GifRecorder> gif_recorder;
    const bool write_gif = !video_path.empty() && is_gif_path(video_path);
    if (write_gif) {
      gif_recorder.emplace(opts.fps);
      if (!gif_recorder->start()) {
        std::cerr << "Warning: failed to create temp directory for GIF frames; GIF export disabled.\n";
        gif_recorder.reset();
      }
    }

    bool done = false;
    int step = 0;

    while (!done) {
      torch::NoGradGuard guard;
      auto obs = world.obs();
      auto output = policy->forward(obs);
      auto logits = output.policy_logits.squeeze(0);
      TORCH_CHECK(logits.dim() == 1, "policy logits must be 1-D");
      TORCH_CHECK(logits.size(0) == action_dim, "policy logits/action mismatch");

      const int action_idx = select_action(world, logits);
      const auto& action = world.action_space.at(static_cast<size_t>(action_idx));

      done = world.step_blue(action);
      ++step;

      std::optional<double> value_pred;
      if (output.value.numel() > 0) {
        value_pred = output.value.squeeze().item<double>();
      }

      auto frame = renderer.draw_snapshot(world, std::nullopt, {}, opts.show_window, value_pred);

      if (!video_path.empty()) {
        if (gif_recorder) {
          if (!gif_recorder->add_frame(frame)) {
            std::cerr << "Warning: failed to write GIF frame " << step << "\n";
          }
        } else {
          if (!video_ready) {
            const double fps = std::max(opts.fps, 1e-3);
            const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            video_ready = writer.open(video_path.string(), fourcc, fps,
                                      frame.size(), /*isColor=*/true);
            if (!video_ready) {
              std::cerr << "Warning: failed to open video writer at "
                        << video_path << std::endl;
            }
          }
          if (video_ready) {
            writer.write(frame);
          }
        }
      }
    }

    std::cout<< "Made: "<< step << " number of steps."<<std::endl;

    if (gif_recorder) {
      if (gif_recorder->finalize(video_path)) {
        std::cout << "Saved GIF: " << video_path << std::endl;
      } else {
        std::cerr << "Failed to finalize GIF at " << video_path << std::endl;
      }
    } else {
      if (writer.isOpened()) {
        writer.release();
        std::cout << "Saved video: " << video_path << std::endl;
      }
    }

    const auto scores = world.final_scores();
    const auto scores_norm = world.final_scores_norm();

    std::cout << "Episode " << (ep + 1)
              << ": blue_score=" << scores.first
              << " yellow_score=" << scores.second
              << " normalized_return=" << scores_norm.first
              << std::endl;
  }

  return 0;
}
