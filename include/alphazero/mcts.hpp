#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "alphazero/policy_network.hpp"

namespace alphazero {

struct MCTSConfig {
  int simulations = 800;
  double cpuct = 1.25;
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;
};

struct TreeNode {
  int vis_count = 0;
  double prior = 0.0;

};

struct SearchResult {
  int action = -1;
  double q_value = 0.0;
  double prior = 0.0;
  int visit_count = 0;
};

class MCTS {
 public:
  explicit MCTS(MCTSConfig config, PolicyNetwork network);

  std::vector<SearchResult> search(const torch::Tensor& root_state);

  void reset();

 private:
  MCTSConfig config_;
  PolicyNetwork &policy_network;

  void expand(const alphazero::TreeNode &root);
};

}  // namespace alphazero
