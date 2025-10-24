#include "alphazero/policy_network.hpp"

#include <stdexcept>
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
      trunk_(register_module("trunk", torch::nn::Sequential())),
      policy_head_(register_module(
          "policy_head",
          torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_sizes.back(), options_.action_size)))),
      value_head_(register_module(
          "value_head",
          torch::nn::Linear(torch::nn::LinearOptions(options_.hidden_sizes.back(), 1)))) {

  // build the trunk of the model
  int prev_dim = options_.input_size;
  for(int size : options_.hidden_sizes){
      trunk_->push_back(torch::nn::Linear(
          torch::nn::LinearOptions(prev_dim, size)));
      trunk_->push_back(torch::nn::ReLU());
  }
}

PolicyOutput PolicyNetworkImpl::forward(const torch::Tensor& state) {
  auto x = flatten_state(state);
  x = trunk_->forward(x);
  auto logits = policy_head_->forward(x);
  auto value = value_head_->forward(x);
  return {logits, value};
}

}  // namespace alphazero
