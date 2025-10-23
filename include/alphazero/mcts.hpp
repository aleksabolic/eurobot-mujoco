#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "alphazero/policy_network.hpp"

namespace alphazero {

struct MCTSConfig {
  int num_simulations = 800;
  double cpuct = 1.25;
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;
};

struct TreeNode {
  TreeNode(double prior): prior(prior) {}
  TreeNode() {}
  int vis_count = 0;
  double value_sum = 0.0;
  double prior = 0.0;
  std::vector<std::pair<int,TreeNode>> children = {}; // pair (action idx, node)
  bool is_expanded = false;

  double value(){
      if(vis_count == 0.0) return 0.0;
      return value_sum / vis_count;
  }
};

class MCTS {
 public:
  explicit MCTS(MCTSConfig config, PolicyNetwork &network);

  TreeNode search(const torch::Tensor& root_state);

 private:
  MCTSConfig config_;
  PolicyNetwork &policy_network;

  void simulate(const TreeNode &root);
  double expand(const torch::Tensor &obs, TreeNode &node);
  std::pair<int, TreeNode> select_child(const TreeNode &node);
  void backprop(std::vector<TreeNode> &path, double value);
  void add_dirichlet_noise(TreeNode &root);
};

}  // namespace alphazero
