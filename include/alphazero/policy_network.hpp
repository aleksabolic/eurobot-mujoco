#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace alphazero {

struct PolicyNetworkOptions {
  int64_t input_size = 1;
  std::vector<int> hidden_sizes = {256, 256};
  int64_t action_size = 1;
};

struct PolicyOutput {
  torch::Tensor policy_logits;
  torch::Tensor value;
};

class PolicyNetworkImpl : public torch::nn::Module {
 public:
  explicit PolicyNetworkImpl(PolicyNetworkOptions options);

  [[nodiscard]] PolicyOutput forward(const torch::Tensor& state);
  [[nodiscard]] const PolicyNetworkOptions& options() const noexcept { return options_; }

 private:
  std::vector<torch::nn::Linear> layers_;
  PolicyNetworkOptions options_;
  torch::nn::Linear policy_head_;
  torch::nn::Linear value_head_;
};

TORCH_MODULE(PolicyNetwork);

}  // namespace alphazero
