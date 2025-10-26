#include "alphazero/mcts.hpp"

#include <ATen/core/Reduction.h>
#include <cmath>
#include <torch/torch.h>

#include <stdexcept>
#include <utility>

namespace alphazero {

MCTS::MCTS(MCTSConfig config, PolicyNetwork &network, ActionHelper &action_helper)
    : config_(std::move(config)), policy_network_(network), action_helper_(action_helper) {
  if (config_.num_simulations <= 0) {
    throw std::invalid_argument("MCTS: simulations must be positive");
  }
}

/*
 * Searches with mcts to find the best possible action for current root_state.
 * This can be seen as main MCTS function.
 */
TreeNode MCTS::search(eurobot::EurobotWorld &world) {

    const eurobot::EurobotState root_state = world.get_state();
    auto root = std::make_shared<TreeNode>(1.0f);

    expand(world, root);
    add_dirichlet_noise(root);

    for(int i = 0; i < config_.num_simulations; i++){
        world.set_state(root_state);
        simulate(world, root);
    }
    world.set_state(root_state);

    return *root;
}

/*
 * Expandes the node with all possible legal actions in current state of the world
 * and assigns prior to each child using policy network
 */
double MCTS::expand(eurobot::EurobotWorld &world, std::shared_ptr<TreeNode>& node){
    // run obs through policy network
    c10::InferenceMode guard;
    auto [logits, value] = policy_network_->forward(world.obs());
    logits = logits.squeeze(0);
    TORCH_CHECK(logits.dim() == 1, "policy logits must be 1D [A]");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "policy logits must be float32");
    TORCH_CHECK(logits.size(0) == (long)action_helper_.actions().size(),
                "policy logits size != action space size");

    // TODO: collect multi-head logits into one [#action_size] tensor

    // mask it with legal_action_mask
    auto logit_mask = action_helper_.logit_mask(world);
    auto masked_logits = logits + logit_mask;
    masked_logits = torch::where(torch::isfinite(masked_logits),
                                 masked_logits,
                                 torch::full_like(masked_logits, -1e9f));
    auto priors = torch::softmax(masked_logits, -1).contiguous();

    const int64_t A = priors.size(0);
    auto acc = priors.data_ptr<float>(); ;

    node->children.clear();
    node->children.reserve(A);
    for (int64_t a = 0; a < A; ++a) {
    float p = acc[a];
        if (p > 0.0f && std::isfinite(p)) {
            node->children.emplace_back((int)a, std::make_shared<TreeNode>(p));
        }
    }
    node->is_expanded = true;
    return value.item<double>();
}

/*
 * Selects the action of the expanded node using (P)UCT.
 */
std::pair<int, std::shared_ptr<TreeNode>> MCTS::select_child(const std::shared_ptr<TreeNode>& node){
    double best_score = -std::numeric_limits<double>::infinity();
    int best_action = -1;
    std::shared_ptr<TreeNode> best_child;
    const double sqrt_total = std::sqrt(node->vis_count + 1.0);

    for (const auto& kv : node->children) {
        const int a = kv.first;
        const auto& child = kv.second;

        double q = child->value();
        double u = config_.cpuct * child->prior * sqrt_total / (1.0 + child->vis_count);
        double score = q + u;
        if(score > best_score){
            best_score = score;
            best_action = a;
            best_child = child;
        }
    }
    return {best_action, best_child};
}

/*
 * Traverse the tree from the root until first unexpanded node, or
 * until end of the episode. At the end, backup the traversed path up the tree.
 */
void MCTS::simulate(eurobot::EurobotWorld &world, const std::shared_ptr<TreeNode>& root){
    std::vector<std::shared_ptr<TreeNode>> path;
    auto node = root;
    path.push_back(node);

    double value = 0.0;

    while(true){
        if(!node->is_expanded){
            value = expand(world, node);
            break;
        }
        auto [action_idx, child] = select_child(node);
        const auto &action = action_helper_.action_by_index(action_idx);
        const bool done = world.step_blue(action);

        path.push_back(child);
        node = child;

        if(done){
            value = world.final_scores_norm().first;
            break;
        }
    }

    // backprop
    for (int i = (int)path.size() - 1; i >= 0; --i) {
        path[i]->vis_count += 1;
        path[i]->value_sum += value;
    }
}

/*
 * Adds dirichlet noise to the root children priors
 */
void MCTS::add_dirichlet_noise(std::shared_ptr<TreeNode>& node) {
  const float eps   = config_.dirichlet_epsilon;
  const float alpha = config_.dirichlet_alpha;
  if (node->children.empty() || eps <= 0.f || alpha <= 0.f) return;

  // sample Dirichlet(alpha) via Gamma(alpha,1)
  std::vector<float> noise(node->children.size());
  std::gamma_distribution<float> gamma(alpha, 1.0f);
  float sum = 0.f;
  static thread_local std::mt19937 rng{std::random_device{}()};

  for (auto &z : noise) { z = std::max(1e-12f, gamma(rng)); sum += z; }
  const float inv_sum = 1.0f / sum;
  for (auto &z : noise) z *= inv_sum;

  // mix into child priors
  for (size_t i = 0; i < node->children.size(); ++i) {
    auto& child = node->children[i].second;
    child->prior = (1.0f - eps) * child->prior + eps * noise[i];
  }
}


} // namespace alphazero
