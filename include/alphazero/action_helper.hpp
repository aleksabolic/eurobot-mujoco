#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <torch/torch.h>
#include <alphazero/eurobot_world.hpp>

namespace alphazero {

class ActionHelper {
public:

  explicit ActionHelper(const eurobot::EurobotWorld& world) {
    // Flat action list (from world)
    actions_ = world.action_space();
  }

  const std::vector<eurobot::Action>& actions() const { return actions_; }
  inline const eurobot::Action& action_by_index(int idx) const { return actions_[idx]; }

  // Bool mask (shape: [num_actions]) over all possible actions
  torch::Tensor legal_mask(const eurobot::EurobotWorld& world, bool blue_turn = true) const;

  // Add to logits: 0 for legal, -inf (or neg_large) for illegal
  torch::Tensor logit_mask(const eurobot::EurobotWorld& world, bool blue_turn = true, float neg_large = -1e9f) const;

private:
  std::vector<eurobot::Action> actions_;
};

} // namespace alphazero
