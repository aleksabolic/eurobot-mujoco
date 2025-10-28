#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/videoio.hpp>
#include <torch/serialize/archive.h>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "alphazero/action_helper.hpp"
#include "alphazero/eurobot_render_cv2.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/policy_network.hpp"

namespace {

struct EvalOptions {
  std::string model_path;
  std::string video_path;
  int episodes = 1;
  double fps = 1.0;
  bool show_window = false;
};

void print_usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --model <path> [--episodes N] [--fps FPS] [--video PATH] [--show]\n";
}

EvalOptions parse_args(int argc, char** argv) {
  EvalOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      opts.model_path = argv[++i];
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
    ext = ".avi";
  }

  stem.replace_extension();  // drop extension
  if (total_episodes > 1) {
    stem += "_ep" + std::to_string(episode_idx + 1);
  }
  stem += ext;
  return stem;
}

eurobot::RobotProfile load_profile(const YAML::Node& node) {
  if (!node) {
    throw std::runtime_error("Missing profile configuration");
  }

  eurobot::RobotProfile p;
  p.capacity = node["capacity"].as<int>();
  p.max_action_qty = node["max_action_qty"].as<int>();
  p.can_flip = node["can_flip"].as<bool>();

  const float v_max = node["v_max"] ? node["v_max"].as<float>() : 0.2;
  const float a_max = node["a_max"] ? node["a_max"].as<float>() : 1.0;

  p.travel_time = [v_max, a_max](double d) { 
    float d_min = v_max * v_max / a_max; 

    float t_motion = 0.0;
    if(d <= d_min + 1e-12) {
      t_motion = 2.0 * std::sqrt(d / a_max);
    }
    else {
      t_motion = 2.0 * (v_max / a_max) + (d - d_min) / v_max;
    }

    // start-stop time + t_motion
    return 0.4 + t_motion;
  };

  const auto handle = node["handle_time"];
  const auto read_coeff = [&](const char* key, double fallback) {
    return handle && handle[key] ? handle[key].as<double>() : fallback;
  };
  const double pick = read_coeff("PICK", 0.0);
  const double place = read_coeff("PLACE", 0.0);
  const double flip = read_coeff("FLIP", 0.0);
  const double steal = read_coeff("STEAL", 0.0);
  p.handle_time = [pick, place, flip, steal](eurobot::Verb v, int q) {
    const double coeff = [&]() {
      switch (v) {
        case eurobot::Verb::PICK: return pick;
        case eurobot::Verb::PLACE: return place;
        case eurobot::Verb::FLIP: return flip;
        case eurobot::Verb::STEAL: return steal;
      }
      return 0.0;
    }();
    return coeff * q;
  };

  return p;
}

struct WorldBundle {
  eurobot::EurobotWorld world;
  alphazero::ActionHelper action_helper;
};

WorldBundle make_world(const YAML::Node& config) {
  const auto profiles = config["profiles"];
  if (!profiles || !profiles["blue"] || !profiles["yellow"]) {
    throw std::runtime_error("profiles.blue/yellow missing in configs/world.yaml");
  }

  eurobot::RobotProfile blue_prof = load_profile(profiles["blue"]);
  eurobot::RobotProfile yellow_prof = load_profile(profiles["yellow"]);

  eurobot::RewardConfig rewards{};
  if (const auto reward = config["reward"]) {
    const auto read_reward = [&](const char* key, float& target) {
      if (reward[key]) {
        target = reward[key].as<float>();
      }
    };
    read_reward("pantry_bonus", rewards.pantry_bonus);
    read_reward("nest_bonus", rewards.nest_bonus);
    read_reward("interest_bonus", rewards.interest_bonus);
    read_reward("finish_in_nest_bonus", rewards.finish_in_nest_bonus);
  }

  bool allow_steal = true;
  uint64_t seed = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  bool record_history = false;
  if (const auto world_node = config["world"]) {
    if (world_node["allow_steal"]) allow_steal = world_node["allow_steal"].as<bool>();
    if (world_node["seed"]) seed = world_node["seed"].as<uint64_t>();
    if (world_node["record_history"]) record_history = world_node["record_history"].as<bool>();
  }

  eurobot::EurobotWorld world(blue_prof, yellow_prof, rewards, allow_steal, seed, record_history);
  alphazero::ActionHelper helper(world);
  return {std::move(world), std::move(helper)};
}

int select_action(const alphazero::ActionHelper& helper,
                  const eurobot::EurobotWorld& world,
                  const torch::Tensor& logits) {
  auto mask = helper.logit_mask(world).to(logits.device());
  auto masked_logits = logits + mask;
  auto legal_mask = helper.legal_mask(world).to(torch::kCPU);

  const auto has_legal = legal_mask.any().item<bool>();
  if (!has_legal) {
    throw std::runtime_error("ActionHelper reported no legal actions for the current state.");
  }

  auto action_idx = masked_logits.argmax().item<int64_t>();
  if (!legal_mask[action_idx].item<bool>()) {
    // fallback: pick first legal entry
    auto legal_indices = legal_mask.nonzero();
    action_idx = legal_indices[0].item<int64_t>();
  }
  return static_cast<int>(action_idx);
}

}  // namespace

int main(int argc, char** argv) {
  const EvalOptions opts = parse_args(argc, argv);

  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

  const YAML::Node config = YAML::LoadFile(resolve_config_path().string());
  auto [world, action_helper] = make_world(config);

  const int64_t obs_dim = world.obs().size(0);
  const int64_t action_dim = static_cast<int64_t>(action_helper.actions().size());

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

    bool done = false;
    int step = 0;

    while (!done) {
      torch::NoGradGuard guard;
      auto obs = world.obs().to(device);
      auto output = policy->forward(obs.unsqueeze(0));
      auto logits = output.policy_logits.squeeze(0);

      const int action_idx = select_action(action_helper, world, logits);
      const auto& action = action_helper.action_by_index(action_idx);

      done = world.step_blue(action);
      ++step;

      auto frame = renderer.draw_snapshot(world, std::nullopt, {}, opts.show_window);

      if (!video_path.empty()) {
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

    std::cout<< "Made: "<< step << " number of steps."<<std::endl;

    if (writer.isOpened()) {
      writer.release();
      std::cout << "Saved video: " << video_path << std::endl;
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
