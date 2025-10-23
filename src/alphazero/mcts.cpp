#include "alphazero/mcts.hpp"

#include <torch/torch.h>

#include <stdexcept>
#include <utility>

namespace alphazero {

namespace {
torch::Tensor flatten_state(const torch::Tensor& state) {
  if (state.dim() == 1) {
    return state.view({1, -1});
  }
  if (state.dim() == 2) {
    return state;
  }
  return state.flatten(1);
}

torch::Device infer_device(const PolicyNetwork& network) {
  for (const auto& parameter : network->parameters()) {
    return parameter.device();
  }
  return torch::kCPU;
}
}  // namespace

MCTS::MCTS(MCTSConfig config, PolicyNetwork &network) : config_(std::move(config)) {
  if (config_.simulations <= 0) {
    throw std::invalid_argument("MCTS: simulations must be positive");
  }
}

std::vector<SearchResult> MCTS::run_search(const torch::Tensor& root_state,
                                           PolicyNetwork& policy_network) {
  auto device = infer_device(policy_network);
  auto prepared_state = flatten_state(root_state).to(device);
  auto output = policy_network->forward(prepared_state);

  auto policy_logits = output.policy_logits;
  auto visit_distribution = torch::softmax(policy_logits, -1);

  auto flattened = visit_distribution.squeeze(0).to(torch::kCPU);
  auto value = output.value.to(torch::kCPU).squeeze().item<double>();

  std::vector<SearchResult> results;
  results.reserve(flattened.size(0));

  for (int64_t idx = 0; idx < flattened.size(0); ++idx) {
    results.push_back(SearchResult{
        .action = static_cast<int>(idx),
        .q_value = value,
        .prior = flattened[idx].item<double>(),
        .visit_count = config_.simulations / flattened.size(0),
    });
  }

  return results;
}

void MCTS::reset() {
  // Stub for future state cleanup.
}

}  // namespace alphazero
