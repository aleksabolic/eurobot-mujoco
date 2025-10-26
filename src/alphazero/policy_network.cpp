#include "alphazero/policy_network.hpp"

#include <torch/nn/options/linear.h>

namespace alphazero {

namespace {
torch::Tensor flatten_state(const torch::Tensor& state) {
  if (state.dim() == 1) {
    return state.view({1, -1});
  }
  if (state.dim() == 2 && state.size(0) > 1) {
    return state;
  }
  return state.flatten(1);
}
}  // namespace

PolicyNetworkImpl::PolicyNetworkImpl(PolicyNetworkOptions options)
    : options_(std::move(options)),
      policy_head_(register_module(
          "policy_head",
          torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_sizes.back(), options_.action_size)))),
      value_head_(register_module(
          "value_head",
          torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_sizes.back(), 1)))) {

  // build the trunk of the model
  int prev_dim = options_.input_size;

  for (int size : options_.hidden_sizes) {
    layers_.emplace_back(torch::nn::Linear(prev_dim, size));
    register_module("lin" + std::to_string(layers_.size()), layers_.back());
    prev_dim = size;
  }
}

PolicyOutput PolicyNetworkImpl::forward(const torch::Tensor& state) {
  auto target = policy_head_->weight.device();
  auto x = flatten_state(state).to(torch::TensorOptions().dtype(torch::kFloat32).device(target));
  for (auto& lin : layers_) x = torch::relu(lin->forward(x));
  auto logits = policy_head_->forward(x);
  auto value = value_head_->forward(x);
  return {logits, value};
}

}  // namespace alphazero
