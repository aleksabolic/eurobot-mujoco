#include <algorithm>
#include <limits>
#include <numeric>
#include <torch/torch.h>
#include <alphazero/action_helper.hpp>

namespace alphazero {

torch::Tensor ActionHelper::legal_mask(const eurobot::EurobotWorld& world, bool blue_turn) const {
  const auto S = world.get_state();
  const auto& rob = blue_turn ? S.blue   : S.yellow;
  const int   max_qty = blue_turn ? max_qty_blue_ : max_qty_yell_;
  const int   nest_id = blue_turn ? nest_blue_     : nest_yell_;

  std::vector<uint8_t> mask(actions_.size(), 0);

  const int inv_sum = int(rob.inv[0]) + int(rob.inv[1]);
  const int room_total = std::max(0, rob.profile.capacity - inv_sum);

  auto sum2 = [](const std::array<int16_t,2>& a){ return int(a[0]) + int(a[1]); };

  for (size_t i = 0; i < actions_.size(); ++i) {
    const auto& a = actions_[i];
    const auto verb  = static_cast<eurobot::Verb>(a.verb);
    const int  node  = a.node;
    const int  color = a.color;
    const int  qty   = a.qty;

    if (qty < 1 || qty > max_qty) { mask[i] = 0; continue; }

    switch (verb) {
      case eurobot::Verb::PICK: {
        const int idx = (node >= 0 && node < (int)node_to_pickup_idx_.size()) ? node_to_pickup_idx_[node] : -1;
        if (idx < 0) break;
        const int available = S.pickups[idx][color];
        const int k = std::min({qty, available, room_total});
        mask[i] = (k > 0);
      } break;

      case eurobot::Verb::PLACE: {
        const int have = rob.inv[color];
        if (node == nest_id) {
          const int nest_rem = std::max(0, eurobot::NEST_CAP - (blue_turn ? S.nest_blue : S.nest_yellow));
          const int k = std::min({qty, have, nest_rem});
          mask[i] = (k > 0);
          break;
        }
        const int pidx = (node >= 0 && node < (int)node_to_pantry_idx_.size()) ? node_to_pantry_idx_[node] : -1;
        if (pidx < 0) break;
        const int room = std::max(0, pantry_cap_ - sum2(S.pantries[pidx]));
        const int k = std::min({qty, have, room});
        mask[i] = (k > 0);
      } break;

      case eurobot::Verb::FLIP: {
        // Flip ignores node.
        mask[i] = (rob.profile.can_flip && qty > 0 && rob.inv[color] >= qty);
      } break;

      case eurobot::Verb::STEAL: {
        if (!allow_steal_) { mask[i] = 0; break; }
        const int pidx = (node >= 0 && node < (int)node_to_pantry_idx_.size()) ? node_to_pantry_idx_[node] : -1;
        if (pidx < 0) break;
        const int have = S.pantries[pidx][color];
        const int k = std::min({qty, have, room_total});
        mask[i] = (k > 0);
      } break;
    }
  }

  auto t = torch::from_blob(mask.data(), {(long)mask.size()}, torch::TensorOptions().dtype(torch::kBool)).clone();
  return t;
}

torch::Tensor ActionHelper::logit_mask(const eurobot::EurobotWorld& world, bool blue_turn, float neg_large) const {
  auto legal = legal_mask(world, blue_turn).to(torch::kFloat32);
  return (1.0f - legal) * neg_large;  // 0 for legal, -inf-ish for illegal
}


} // namespace alphazero
