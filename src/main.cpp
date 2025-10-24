// src/main.cpp
#include <iostream>
#include <torch/torch.h>

#include "alphazero/trainer.hpp"
#include "alphazero/eurobot_world.hpp"
#include "alphazero/policy_network.hpp"

using namespace alphazero;
using namespace eurobot;

int main() {
  // device
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  torch::manual_seed(42);

  // TODO: move these to .yaml
  // ---- Eurobot world setup ----
  Profile blue_prof{
    /*capacity=*/6,
    /*max_action_qty=*/6,
    /*can_flip=*/true,
    /*travel_time=*/[](double d){ return 0.5 * d; },
    /*handle_time=*/[](Verb v, int q){
      switch (v) {
        case Verb::PICK:  return 0.20 * q;
        case Verb::PLACE: return 0.20 * q;
        case Verb::FLIP:  return 0.10 * q;
        case Verb::STEAL: return 0.30 * q;
      }
      return 0.0;
    }
  };
  Profile yellow_prof = blue_prof;

  RewardConfig R{
    /*pantry_bonus=*/1.f,
    /*nest_bonus=*/5.f,
    /*interest_bonus=*/1.f,
    /*finish_in_nest_bonus=*/2.f
  };

  EurobotWorld world(blue_prof, yellow_prof, R, /*allow_steal=*/true, /*seed=*/42, /*record_history=*/false);

  // sizes
  const int64_t obs_dim = world.obs().size(0);
  const int64_t action_dim = static_cast<int64_t>(world.action_space().size());

  // ---- Network ----
  PolicyNetworkOptions network_options {
      /*input_size=*/ obs_dim,
      /*hidden_sizes=*/ {256, 256},
      /*action_size=*/ action_dim
  };

  auto net = PolicyNetwork(std::move(network_options));
  net->to(device);

  // ---- Trainer config ----
  TrainingConfig cfg;
  cfg.num_iterations        = 10;
  cfg.games_per_iter        = 4;
  cfg.training_steps        = 200;
  cfg.batch_size            = 128;
  cfg.replay_capacity       = 50'000;

  cfg.num_simulations       = 128;
  cfg.cpuct                 = 1.5;
  cfg.dirichlet_alpha       = 0.3;
  cfg.dirichlet_epsilon     = 0.25;

  cfg.learning_rate         = 1e-3;
  cfg.weight_decay          = 1e-4;
  cfg.max_grad_norm         = 5.0;

  cfg.policy_loss_weight    = 1.0;
  cfg.value_loss_weight     = 1.0;
  cfg.entropy_weight        = 0.0;

  cfg.temperature           = 1.0;
  cfg.temperature_decay_steps = 30;

  // ---- Trainer ----
  AlphaZeroTrainer trainer(net, world, cfg);
  trainer.set_device(device);

  std::cout << "device: " << (device.is_cuda() ? "CUDA" : "CPU")
            << ", obs_dim=" << obs_dim
            << ", action_dim=" << action_dim << std::endl;

  trainer.train();

  // save checkpoint
  torch::save(net, "runs/alphazero_policy.pt");

  return 0;
}
