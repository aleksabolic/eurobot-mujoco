#pragma once

#include <torch/torch.h>
#include "alphazero/policy_network.hpp"
#include "alphazero/action_helper.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/mcts.hpp"
#include "alphazero/replay_buffer.hpp"

namespace alphazero {

struct TrainingConfig {
  int    num_iterations      = 10;
  int    games_per_iter      = 4;
  int    training_steps      = 200;
  int    batch_size          = 128;
  int    replay_capacity     = 50'000;

  int    num_simulations     = 128;
  double cpuct               = 1.5;
  double dirichlet_alpha     = 0.3;
  double dirichlet_epsilon   = 0.25;

  double learning_rate       = 1e-3;
  double weight_decay        = 1e-4;
  double max_grad_norm       = 5.0;

  double policy_loss_weight  = 1.0;
  double value_loss_weight   = 1.0;
  double entropy_weight      = 0.0;

  double temperature         = 1.0;
  int    temperature_decay_steps = 30;
};

class AlphaZeroTrainer {
public:
  AlphaZeroTrainer(PolicyNetwork network,
                   eurobot::EurobotWorld world,
                   const TrainingConfig& cfg);

  void set_device(const torch::Device& device);
  void train();

private:
  void play_episode();
  void optimize_step();

  // helpers
  int sample_action_from_visits(const std::vector<float>& visits, double temperature);
  double select_temperature(int step_idx) const;

private:
  TrainingConfig cfg_;
  torch::Device device_ = torch::kCPU;

  // ownership
  PolicyNetwork policy_;
  eurobot::EurobotWorld world_;
  ActionHelper action_helper_;
  ReplayBuffer replay_;

  MCTS mcts_;
  torch::optim::Adam optimizer_;

  // cached dims
  int64_t action_dim_ = 0;
};

} // namespace alphazero
