#include <algorithm>
#include <limits>
#include <numeric>
#include <torch/torch.h>
#include <alphazero/action_helper.hpp>

namespace alphazero {

torch::Tensor ActionHelper::legal_mask(const eurobot::EurobotWorld& world, bool blue_turn) const {
  const auto& rob = blue_turn ? world.blue   : world.yellow;
  const int   max_qty = blue_turn ? world.blue.profile.max_action_qty : world.yellow.profile.max_action_qty;
  const int   nest_id = blue_turn ? world.NEST_BLUE  : world.NEST_YELL;

  auto mask = torch::empty({static_cast<long>(actions_.size())},
                           torch::TensorOptions().dtype(torch::kBool));
  auto* mask_ptr = mask.data_ptr<bool>();

  const int inv_sum = int(rob.inv[0]) + int(rob.inv[1]);
  const int room_total = std::max(0, rob.profile.capacity - inv_sum);

  auto sum2 = [](const std::array<int16_t,2>& a){ return int(a[0]) + int(a[1]); };

  for (size_t i = 0; i < actions_.size(); ++i) {
    const auto& a = actions_[i];
    const auto verb  = static_cast<eurobot::Verb>(a.verb);
    const int  node  = a.node;
    const int  color = a.color;
    const int  qty   = a.qty;

    bool legal = false;
    if (qty < 1 || qty > max_qty) { mask_ptr[i] = false; continue; }

    switch (verb) {
      case eurobot::Verb::PICK: {
        // check if the node is a pickup node
        const int idx = world.pickup_idx(node);
        if (idx < 0) break;
        const int available = world.pickups[idx][color];
        legal = (available >= qty && room_total >= qty);
      } break;

      case eurobot::Verb::PLACE: {
        const int have = rob.inv[color];
        if (node == nest_id) {
          const int nest_rem = std::max(0, world.nest_cap() - (blue_turn ? world.nest_blue : world.nest_yellow));
          legal = (qty <= nest_rem && qty <= have);
          break;
        }
        const int pidx = world.pantry_idx(node);
        if (pidx < 0) break;
        const int room = std::max(0, world.pantry_cap() - sum2(world.pantries[pidx]));
        legal = (qty <= room && qty <= have);
      } break;

      case eurobot::Verb::FLIP: {
        // Flip ignores node.
        legal = (rob.profile.can_flip && rob.inv[color] >= qty);
      } break;

      case eurobot::Verb::STEAL: {
        if (!world.allow_steal()) break;
        const int pidx = world.pantry_idx(node);
        if (pidx < 0) break;
        const int have = world.pantries[pidx][color];
        legal = (qty <= room_total && qty <= have);
      } break;
    }

    mask_ptr[i] = legal;
  }

  return mask;
}

torch::Tensor ActionHelper::logit_mask(const eurobot::EurobotWorld& world, bool blue_turn, float neg_large) const {
  auto legal = legal_mask(world, blue_turn).to(torch::kFloat32);
  return (1.0f - legal) * neg_large;  // 0 for legal, -inf-ish for illegal
}


} // namespace alphazero
