#include "alphazero/mcts.hpp"

#include <ATen/core/Reduction.h>
#include <cmath>
#include <torch/torch.h>

#include <stdexcept>
#include <utility>

namespace alphazero {

MCTS::MCTS(MCTSConfig config, PolicyNetwork &network)
    : config_(std::move(config)), policy_network(network) {
  if (config_.num_simulations <= 0) {
    throw std::invalid_argument("MCTS: simulations must be positive");
  }
}

/*
 * Searches with mcts to find the best possible action for current root_state.
 * This can be seen as main MCTS function.
 */
TreeNode MCTS::search(const torch::Tensor &obs) {
    TreeNode root(1.0);

    expand(obs, root);

    add_dirichlet_noise(root);

    for(int i = 0; i < config_.num_simulations; i++){
        simulate(root);
    }

    return root;
}

// TODO: check if TreeNodes are passed as ref inside vector? (yes??)
void MCTS::backprop(std::vector<TreeNode> &path, double value){
    for(int i = (int)path.size()-1; i>=0; i--){
        path[i].vis_count++;
        path[i].value_sum += value;
    }
}

double MCTS::expand(const torch::Tensor &obs, TreeNode &node){
    // auto legal_mask = legal_action_mask();

    // run obs through policy network
    auto [logits, value] = policy_network->forward(obs);

    // collect logits into one [#action_size] tensor
    // mask it with legal_action_mask
    // softmax over masked logits

    for (int i = 0; i<action_helper.size(); i++){
        if(mask[i]){
            node.children.push_back({i, TreeNode(logits[i])})
        }
    }
    node.is_expanded = true;
    return value.item();
}

/*
 * Selects the action of the expanded node using (P)UCT.
 */
std::pair<int, TreeNode> MCTS::select_child(const TreeNode &node){
    double best_score = -MAXFLOAT;
    int best_action = -1;
    TreeNode best_child;
    double sqrt_total = sqrt(node.vis_count + 1.0);

    for(auto child : node.children){
        double q = child.second.value();
        double u = config_.cpuct * child.second.prior * sqrt_total / (1.0 + child.second.vis_count);
        double score = q + u;
        if(score > best_score){
            best_score = score;
            best_action = child.first;
            best_child = child.second;
        }
    }
    return {best_action, best_child};
}

/*
 * Traverse the tree from the root until first unexpanded node, or
 * until end of the episode. At the end, backup the traversed path up the tree.
 */
void MCTS::simulate(const TreeNode &root){

    // *&
    TreeNode curr_node = root;
    std::vector<TreeNode> path = {root};

    double value = 0;
    while(true){
        if(!curr_node.is_expanded){
            value = expand(curr_node);
            break;
        }
        auto [action_idx, child] = select_child(curr_node);
        curr_node = child;
        path.push_back(child);

        bool done = ...
        if(done){
            value = env.
        }
    }

    backprop(path, value);
}

void add_dirichlet_noise(TreeNode &root){
    //TODO
}

} // namespace alphazero
