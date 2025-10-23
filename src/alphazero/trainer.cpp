#include "alphazero/trainer.hpp"

#include <stdexcept>

namespace alphazero {

namespace {
torch::optim::AdamOptions make_adam_options(const TrainingConfig& config) {
  torch::optim::AdamOptions options(config.learning_rate);
  options.weight_decay(config.weight_decay);
  return options;
}
}  // namespace

AlphaZeroTrainer::AlphaZeroTrainer(PolicyNetwork policy,
                                   MCTSConfig mcts_config,
                                   TrainingConfig training_config)
    : policy_(std::move(policy)),
      mcts_(std::move(mcts_config)),
      config_(std::move(training_config)),
      optimizer_(policy_->parameters(), make_adam_options(config_)) {
  if (!policy_) {
    throw std::invalid_argument("AlphaZeroTrainer: policy network must be initialized");
  }
}

void AlphaZeroTrainer::set_device(torch::Device device) {
  device_ = std::move(device);
  policy_->to(device_);
}

TrainingStepMetrics AlphaZeroTrainer::train_step(const torch::Tensor& states,
                                                 const torch::Tensor& target_policies,
                                                 const torch::Tensor& target_values) {
  if (states.dim() != 2) {
    throw std::invalid_argument("AlphaZeroTrainer: states must be a 2D tensor [batch, features]");
  }

  policy_->train();

  auto states_on_device = states.to(device_);
  auto policy_targets = target_policies.to(device_);
  auto value_targets = target_values.to(device_).view({-1});

  optimizer_.zero_grad();

  auto output = policy_->forward(states_on_device);
  auto logits = output.policy_logits;
  auto predicted_values = output.value.view({-1});

  auto log_probs = torch::log_softmax(logits, -1);
  auto policy_loss_tensor = -(policy_targets * log_probs).sum(-1).mean();

  auto value_loss_tensor = torch::mse_loss(predicted_values, value_targets);

  auto probabilities = torch::softmax(logits, -1);
  auto entropy_tensor = -(probabilities * log_probs).sum(-1).mean();

  auto total_loss_tensor = config_.policy_loss_weight * policy_loss_tensor +
                           config_.value_loss_weight * value_loss_tensor -
                           config_.entropy_weight * entropy_tensor;

  total_loss_tensor.backward();
  optimizer_.step();

  TrainingStepMetrics metrics{
      .total_loss = total_loss_tensor.item<double>(),
      .policy_loss = policy_loss_tensor.item<double>(),
      .value_loss = value_loss_tensor.item<double>(),
      .entropy = entropy_tensor.item<double>(),
  };

  return metrics;
}

}  // namespace alphazero
