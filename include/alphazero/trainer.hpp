#pragma once

#include <utility>

#include <torch/torch.h>

#include "alphazero/mcts.hpp"
#include "alphazero/policy_network.hpp"

namespace alphazero {

struct TrainingConfig {
  double learning_rate = 1e-3;
  double policy_loss_weight = 1.0;
  double value_loss_weight = 1.0;
  double entropy_weight = 1e-2;
  double weight_decay = 1e-4;
};

struct TrainingStepMetrics {
  double total_loss = 0.0;
  double policy_loss = 0.0;
  double value_loss = 0.0;
  double entropy = 0.0;
};

class AlphaZeroTrainer {
 public:
  AlphaZeroTrainer(PolicyNetwork policy, MCTSConfig mcts_config, TrainingConfig training_config);

  void set_device(torch::Device device);
  [[nodiscard]] const torch::Device& device() const noexcept { return device_; }

  [[nodiscard]] PolicyNetwork& policy() noexcept { return policy_; }
  [[nodiscard]] MCTS& mcts() noexcept { return mcts_; }

  TrainingStepMetrics train_step(const torch::Tensor& states,
                                 const torch::Tensor& target_policies,
                                 const torch::Tensor& target_values);

 private:
  PolicyNetwork policy_;
  MCTS mcts_;
  TrainingConfig config_;
  torch::optim::Adam optimizer_;
  torch::Device device_ = torch::kCPU;
};

}  // namespace alphazero
