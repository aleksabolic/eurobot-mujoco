#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include <alphazero/eurobot_world.hpp>

namespace alphazero {

class ActionHelper {
public:

  explicit ActionHelper(const eurobot::EurobotWorld& world)
  : n_colors_(eurobot::NUM_COLORS),
    allow_steal_(world.allow_steal()),
    pantries_(world.PANTRIES),
    pickups_(world.PICKUPS),
    nest_blue_(world.NEST_BLUE),
    nest_yell_(world.NEST_YELL),
    max_qty_blue_(world.blue.profile.max_action_qty),
    max_qty_yell_(world.yellow.profile.max_action_qty),
    pantry_cap_(world.pantry_cap())
  {
    // Node -> local indices (O(1))
    node_to_pantry_idx_.assign(world.N(), -1);
    node_to_pickup_idx_.assign(world.N(), -1);
    for (int j = 0; j < (int)pantries_.size(); ++j) node_to_pantry_idx_[pantries_[j]] = j;
    for (int j = 0; j < (int)pickups_.size();  ++j) node_to_pickup_idx_[pickups_[j]]  = j;

    // Flat action list (from world)
    actions_ = world.action_space();

    // Per-verb buckets + index map
    per_verb_indices_.assign(4, {}); // PICK, PLACE, FLIP, STEAL
    per_verb_indices_.reserve(4);
    action_to_index_.reserve(actions_.size());
    for (int i = 0; i < (int)actions_.size(); ++i) {
      per_verb_indices_[actions_[i].verb].push_back(i);
      action_to_index_.emplace(pack(actions_[i]), i);
    }
  }

  const std::vector<eurobot::Action>& actions() const { return actions_; }
  const std::vector<std::vector<int>>& per_verb_indices() const { return per_verb_indices_; }
  inline const eurobot::Action& action_by_index(int idx) const { return actions_[idx]; }

  // Bool mask (shape: [num_actions])
  torch::Tensor legal_mask(const eurobot::EurobotWorld& world, bool blue_turn = true) const;

  // Add to logits: 0 for legal, -inf (or neg_large) for illegal
  torch::Tensor logit_mask(const eurobot::EurobotWorld& world, bool blue_turn = true, float neg_large = -1e9f) const;

  int index_of(const eurobot::Action& a) const {
    auto it = action_to_index_.find(pack(a));
    return (it == action_to_index_.end()) ? -1 : it->second;
  }

private:
  static inline uint64_t pack(const eurobot::Action& a) {
    return (uint64_t(a.verb)  << 48) |
           (uint64_t(a.node)  << 32) |
           (uint64_t(a.color) << 16) |
            uint64_t(a.qty);
  }

  // Cached topology & params
  int n_colors_;
  bool allow_steal_;
  std::vector<int> pantries_, pickups_;
  int nest_blue_, nest_yell_;
  int max_qty_blue_, max_qty_yell_;
  int pantry_cap_;

  std::vector<int> node_to_pantry_idx_, node_to_pickup_idx_;

  // Actions and indices
  std::vector<eurobot::Action> actions_;
  std::vector<std::vector<int>> per_verb_indices_;
  std::unordered_map<uint64_t, int> action_to_index_;
};

} // namespace alphazero
