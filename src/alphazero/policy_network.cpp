#include "alphazero/policy_network.hpp"

#include <stdexcept>

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
      trunk_(register_module("trunk", torch::nn::Sequential())),
      policy_head_(register_module(
          "policy_head",
          torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_size, options_.action_size)))),
      value_head_(register_module(
          "value_head", torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_size, 1)))) {

            
  if (options_.input_size <= 0) {
    throw std::invalid_argument("PolicyNetwork: input_size must be positive");
  }
  if (options_.hidden_size <= 0) {
    throw std::invalid_argument("PolicyNetwork: hidden_size must be positive");
  }
  if (options_.action_size <= 0) {
    throw std::invalid_argument("PolicyNetwork: action_size must be positive");
  }

  trunk_->push_back(torch::nn::Linear(
      torch::nn::LinearOptions(options_.input_size, options_.hidden_size)));
  trunk_->push_back(torch::nn::ReLU());
  trunk_->push_back(torch::nn::Linear(
      torch::nn::LinearOptions(options_.hidden_size, options_.hidden_size)));
  trunk_->push_back(torch::nn::ReLU());

  for (int i = 0; i < options_.residual_blocks; ++i) {
    trunk_->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(options_.hidden_size, options_.hidden_size)));
    trunk_->push_back(torch::nn::ReLU());
  }
}

PolicyOutput PolicyNetworkImpl::forward(const torch::Tensor& state) {
  auto x = flatten_state(state);
  x = trunk_->forward(x);
  auto logits = policy_head_->forward(x);
  auto value = torch::tanh(value_head_->forward(x));
  return {logits, value};
}

}  // namespace alphazero
