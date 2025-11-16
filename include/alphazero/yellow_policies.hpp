#pragma once

#include <functional>
#include <optional>
#include <random>
#include <vector>

#include "alphazero/eurobot_world.hpp"

namespace eurobot {

using YellowPolicyFn = std::function<std::optional<Action>(const std::string&,
                                                           const EurobotWorld&,
                                                           const RobotState&,
                                                           std::mt19937_64&)>;

struct StaticYellowPolicyOptions {
  std::vector<Action> script;
  bool loop = true;
};

struct HeuristicYellowPolicyOptions {
  bool enable_steal = true;
  bool enable_flip = true;
  int min_pickup_stock = 1;
  int min_steal_stock = 1;
  int flip_threshold = 1;
};

YellowPolicyFn make_static_yellow_policy(StaticYellowPolicyOptions opts);
YellowPolicyFn make_heuristic_yellow_policy(HeuristicYellowPolicyOptions opts = {});

}  // namespace eurobot
