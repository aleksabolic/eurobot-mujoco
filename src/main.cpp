#include <algorithm>
#include <iostream>
#include <utility>

#include <torch/torch.h>

#include "alphazero/mcts.hpp"
#include "alphazero/policy_network.hpp"
#include "alphazero/trainer.hpp"

int main() {
  torch::manual_seed(0);

  alphazero::PolicyNetworkOptions network_options;
  network_options.input_size = 64;
  network_options.hidden_size = 128;
  network_options.action_size = 16;
  network_options.residual_blocks = 1;

  auto policy = alphazero::PolicyNetwork(network_options);

  alphazero::MCTSConfig mcts_config;
  mcts_config.simulations = 64;

  alphazero::TrainingConfig training_config;
  training_config.learning_rate = 1e-3;

  alphazero::AlphaZeroTrainer trainer(std::move(policy), mcts_config, training_config);

  const auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  trainer.set_device(device);

  auto states =
      torch::randn({8, network_options.input_size}, torch::TensorOptions().dtype(torch::kFloat32));
  auto target_policies = torch::softmax(
      torch::randn({8, network_options.action_size}, torch::TensorOptions().dtype(torch::kFloat32)),
      -1);
  auto target_values =
      torch::randn({8}, torch::TensorOptions().dtype(torch::kFloat32)).clamp(-1.0, 1.0);

  auto metrics = trainer.train_step(states, target_policies, target_values);

  std::cout << "Training step metrics\n"
            << "  total loss : " << metrics.total_loss << "\n"
            << "  policy loss: " << metrics.policy_loss << "\n"
            << "  value loss : " << metrics.value_loss << "\n"
            << "  entropy    : " << metrics.entropy << "\n";

  auto search_results = trainer.mcts().run_search(states[0], trainer.policy());

  std::cout << "\nSample search results (" << search_results.size() << " actions)\n";
  for (size_t i = 0; i < std::min<size_t>(search_results.size(), 5); ++i) {
    const auto& result = search_results[i];
    std::cout << "  action " << result.action << ": prior=" << result.prior
              << ", q=" << result.q_value << ", visits=" << result.visit_count << "\n";
  }

  return 0;
}
