#pragma once

#include <cstdint>
#include <utility>

#include <torch/torch.h>

namespace alphazero {

struct PolicyNetworkOptions {
  int64_t input_size = 1;
  int64_t hidden_size = 128;
  int64_t action_size = 1;
  int residual_blocks = 0;
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
  PolicyNetworkOptions options_;
  torch::nn::Sequential trunk_;
  torch::nn::Linear policy_head_;
  torch::nn::Linear value_head_;
};

TORCH_MODULE(PolicyNetwork);

}  // namespace alphazero
