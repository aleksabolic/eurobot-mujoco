#pragma once

#include <memory>
#include <string>
#include <utility>
#include <torch/torch.h>

#include "alphazero/policy_network.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/mcts.hpp"
#include "alphazero/replay_buffer.hpp"
#include "alphazero/tensorboard_logger.hpp"

namespace alphazero {

struct TrainingConfig {
  int    num_iterations      = 10;
  int    games_per_iter      = 4;
  int    training_steps      = 200;
  int    batch_size          = 128;
  int    replay_capacity     = 50'000;
  int    num_simulations     = 128;
  int    max_env_steps       = 50;

  double cpuct               = 1.5;
  double dirichlet_alpha     = 0.3;
  double dirichlet_epsilon   = 0.25;
  double gamma               = 0.995;

  double learning_rate       = 1e-3;
  double weight_decay        = 1e-4;
  double max_grad_norm       = 5.0;

  double policy_loss_weight  = 1.0;
  double value_loss_weight   = 1.0;
  double entropy_weight      = 0.0;

  double temperature         = 1.0;
  int    temperature_decay_steps = 30;

  bool enable_tensorboard    = true;
  std::string log_dir        = "runs";
  std::string run_name       = "";
  std::string checkpoint_path = "";
  int checkpoint_interval     = 1;
  bool resume_from_checkpoint = false;
};

struct Transition { 
  torch::Tensor obs; 
  torch::Tensor pi; 
  float value; 
};

struct WorkerBatch {
  std::vector<Transition> traj;
  int episodes = 0;
  double blue_sum = 0.0;
};

struct SelfPlayWorker {
  eurobot::EurobotWorld world;
  MCTS mcts;

  SelfPlayWorker(const eurobot::EurobotWorld& base_world,
                 PolicyNetwork& net,
                 const MCTSConfig& mcfg)
  : world(base_world), mcts(mcfg, net) {}
};

class AlphaZeroTrainer {
public:
  AlphaZeroTrainer(PolicyNetwork network,
                   eurobot::EurobotWorld world,
                   const TrainingConfig& cfg);

  void set_device(const torch::Device& device);
  void train();

private:
  std::pair<float, float> play_episode();
  std::vector<Transition> play_episode_collect(eurobot::EurobotWorld& world, MCTS& mcts);
  void optimize_step();
  bool load_checkpoint(const std::string& path);
  bool save_checkpoint(const std::string& path) const;
  void maybe_save_checkpoint(bool force = false);
  void try_resume();

  // helpers
  int sample_action_from_visits(const std::vector<float>& visits, double temperature);
  double select_temperature(int step_idx) const;

private:
  TrainingConfig cfg_;
  MCTSConfig mcts_cfg_;
  torch::Device device_ = torch::kCPU;

  // ownership
  PolicyNetwork policy_;
  eurobot::EurobotWorld world_;
  ReplayBuffer replay_;

  MCTS mcts_;
  torch::optim::Adam optimizer_;

  std::vector<SelfPlayWorker> workers_;

  // cached dims
  int64_t action_dim_ = 0;
  int64_t train_step_counter_ = 0;
  int64_t iteration_counter_ = 0;
  int64_t last_checkpoint_iteration_ = -1;
  bool has_attempted_resume_ = false;
  bool resume_successful_ = false;
  std::unique_ptr<TensorboardLogger> tb_logger_;
};

} // namespace alphazero
