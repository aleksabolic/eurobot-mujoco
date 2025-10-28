#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <random>

#include <torch/torch.h>

#include "alphazero/policy_network.hpp"
#include "alphazero/eurobot_world.hpp"

namespace alphazero {

struct MCTSConfig {
  int num_simulations = 800;
  double cpuct = 1.25;
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;
  double gamma = 0.995;
};

struct TreeNode {
  float prior = 1.0f;
  int   vis_count = 0;
  float value_sum = 0.0f;
  bool  is_expanded = false;
  // (action_idx, child_ptr)
  std::vector<std::pair<int, std::shared_ptr<TreeNode>>> children;

  explicit TreeNode(float p=1.0f) : prior(p) {}
  inline float value() const { return vis_count ? (value_sum / vis_count) : 0.0f; }
};


class MCTS {
 public:
  explicit MCTS(MCTSConfig config, PolicyNetwork &network);

  TreeNode search(eurobot::EurobotWorld &world);

 private:
  MCTSConfig config_;
  PolicyNetwork &policy_network_;

  void simulate(eurobot::EurobotWorld &world, const std::shared_ptr<TreeNode>& root);
  double expand(eurobot::EurobotWorld &world, std::shared_ptr<TreeNode>& node);
  std::pair<int, std::shared_ptr<TreeNode>> select_child(const std::shared_ptr<TreeNode>& node);
  void add_dirichlet_noise(std::shared_ptr<TreeNode>& node);
};

}  // namespace alphazero
