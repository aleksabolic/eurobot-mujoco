#include "alphazero/yellow_policies.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace eurobot {
namespace {

float l2_distance(const EurobotWorld& world, int a, int b) {
  const auto& nodes = world.nodes();
  const auto& pa = nodes[a].xy;
  const auto& pb = nodes[b].xy;
  const float dx = pa[0] - pb[0];
  const float dy = pa[1] - pb[1];
  return std::sqrt(dx * dx + dy * dy);
}

struct NodeCandidate {
  int node = -1;
  int available = 0;
  float distance = std::numeric_limits<float>::infinity();
  bool valid() const { return node >= 0 && available > 0; }
};

NodeCandidate find_richest_node(const EurobotWorld& world,
                                int from_node,
                                const std::vector<int>& node_ids,
                                const std::vector<std::array<int16_t, NUM_COLORS>>& counts,
                                int color,
                                int min_stock) {
  NodeCandidate best;
  const size_t n = node_ids.size();
  for (size_t i = 0; i < n; ++i) {
    const int node_id = node_ids[i];
    const int amount = static_cast<int>(counts[i][color]);
    if (amount < min_stock) continue;
    const float d = l2_distance(world, from_node, node_id);
    const bool better_stock = amount > best.available;
    const bool same_stock = amount == best.available;
    if (!best.valid() || better_stock || (same_stock && d < best.distance)) {
      best.node = node_id;
      best.available = amount;
      best.distance = d;
    }
  }
  return best;
}

struct StaticSequencePolicy {
  StaticYellowPolicyOptions opts;
  mutable size_t cursor = 0;
  mutable double last_time_left = std::numeric_limits<double>::infinity();

  std::optional<Action> operator()(const std::string&,
                                   const EurobotWorld& world,
                                   const RobotState&,
                                   std::mt19937_64&) const {
    if (opts.script.empty()) return std::nullopt;

    const double now = world.time_left();
    if (now > last_time_left + 1e-6) cursor = 0;  // reset detected
    last_time_left = now;

    if (cursor >= opts.script.size()) {
      if (!opts.loop) return std::nullopt;
      cursor = 0;
    }
    return opts.script[cursor++];
  }
};

struct HeuristicPolicy {
  explicit HeuristicPolicy(HeuristicYellowPolicyOptions options) : opts(std::move(options)) {}

  std::optional<Action> operator()(const std::string&,
                                   const EurobotWorld& world,
                                   const RobotState& robot,
                                   std::mt19937_64&) const {
    const int yellow_color = static_cast<int>(Col::YELLOW);
    const int blue_color = static_cast<int>(Col::BLUE);
    const int max_qty = robot.profile.max_action_qty;
    const int total_inv = robot.inv[0] + robot.inv[1];
    const int free_capacity = std::max(0, robot.profile.capacity - total_inv);
    const int yellow_inv = robot.inv[yellow_color];
    const int blue_inv = robot.inv[blue_color];

    if (yellow_inv > 0 && world.yellow_nest_node_idx >= 0) {
      Action a;
      a.verb = static_cast<int>(Verb::PLACE);
      a.node = world.yellow_nest_node_idx;
      a.color = yellow_color;
      a.qty = std::min(max_qty, yellow_inv);
      return a;
    }

    if (opts.enable_flip && robot.profile.can_flip && blue_inv >= opts.flip_threshold) {
      const int qty = std::min(robot.profile.max_flip_qty, blue_inv);
      if (qty > 0) {
        Action a;
        a.verb = static_cast<int>(Verb::FLIP);
        a.node = robot.curr_node;
        a.color = blue_color;
        a.qty = qty;
        return a;
      }
    }

    if (free_capacity > 0) {
      const auto pick_yellow = find_richest_node(
          world, robot.curr_node, world.pickup_node_ids, world.pickup_crate_counts,
          yellow_color, opts.min_pickup_stock);
      if (pick_yellow.valid()) {
        const int qty = std::max(1, std::min({max_qty, free_capacity, pick_yellow.available}));
        Action a;
        a.verb = static_cast<int>(Verb::PICK);
        a.node = pick_yellow.node;
        a.color = yellow_color;
        a.qty = qty;
        return a;
      }

      if (opts.enable_flip && robot.profile.can_flip) {
        const auto pick_blue = find_richest_node(
            world, robot.curr_node, world.pickup_node_ids, world.pickup_crate_counts,
            blue_color, opts.min_pickup_stock);
        if (pick_blue.valid()) {
          const int qty = std::max(1, std::min({max_qty, free_capacity, pick_blue.available}));
          Action a;
          a.verb = static_cast<int>(Verb::PICK);
          a.node = pick_blue.node;
          a.color = blue_color;
          a.qty = qty;
          return a;
        }
      }
    }

    if (opts.enable_steal && world.allow_steal() && free_capacity > 0) {
      const auto steal_yellow = find_richest_node(
          world, robot.curr_node, world.pantry_node_ids, world.pantry_crate_counts,
          yellow_color, opts.min_steal_stock);
      if (steal_yellow.valid()) {
        const int qty = std::max(1, std::min({max_qty, free_capacity, steal_yellow.available}));
        Action a;
        a.verb = static_cast<int>(Verb::STEAL);
        a.node = steal_yellow.node;
        a.color = yellow_color;
        a.qty = qty;
        return a;
      }

      if (opts.enable_flip && robot.profile.can_flip) {
        const auto steal_blue = find_richest_node(
            world, robot.curr_node, world.pantry_node_ids, world.pantry_crate_counts,
            blue_color, opts.min_steal_stock);
        if (steal_blue.valid()) {
          const int qty = std::max(1, std::min({max_qty, free_capacity, steal_blue.available}));
          Action a;
          a.verb = static_cast<int>(Verb::STEAL);
          a.node = steal_blue.node;
          a.color = blue_color;
          a.qty = qty;
          return a;
        }
      }
    }

    if (opts.enable_flip && robot.profile.can_flip && blue_inv > 0) {
      const int qty = std::min(robot.profile.max_flip_qty, blue_inv);
      if (qty > 0) {
        Action a;
        a.verb = static_cast<int>(Verb::FLIP);
        a.node = robot.curr_node;
        a.color = blue_color;
        a.qty = qty;
        return a;
      }
    }

    return std::nullopt;
  }

  HeuristicYellowPolicyOptions opts;
};

}  // namespace

YellowPolicyFn make_static_yellow_policy(StaticYellowPolicyOptions opts) {
  return StaticSequencePolicy{std::move(opts)};
}

YellowPolicyFn make_heuristic_yellow_policy(HeuristicYellowPolicyOptions opts) {
  if (opts.flip_threshold < 1) opts.flip_threshold = 1;
  if (opts.min_pickup_stock < 1) opts.min_pickup_stock = 1;
  if (opts.min_steal_stock < 1) opts.min_steal_stock = 1;
  return HeuristicPolicy{std::move(opts)};
}

}  // namespace eurobot
