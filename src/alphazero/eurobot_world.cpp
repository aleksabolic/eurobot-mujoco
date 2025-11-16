#include "alphazero/eurobot_world.hpp"
#include <algorithm>
#include <cassert>

namespace eurobot {

static inline float L2(const std::array<float,2>& a, const std::array<float,2>& b) {
  float dx = a[0]-b[0], dy = a[1]-b[1];
  return std::sqrt(dx*dx + dy*dy);
}

std::vector<Node> EurobotWorld::build_nodes(int& blue_nest_node_idx, int& yellow_nest_node_idx,
                                            std::vector<int>& pantry_node_ids,
                                            std::vector<int>& pickup_node_ids) {
  std::vector<Node> v;
  // Nests
  v.push_back(Node{"NestBlue",   { 1.20f,  0.775f}, NodeType::NEST});
  v.push_back(Node{"NestYellow", {-1.20f,  0.775f}, NodeType::NEST});
  blue_nest_node_idx  = 0;
  yellow_nest_node_idx= 1;

  // Pantries A..J
  const std::pair<const char*, std::array<float,2>> PANTRIES_POS[] = {
    {"PantryA", {-0.25f,  0.45f}}, {"PantryB", { 0.25f,  0.45f}},
    {"PantryC", {-1.40f, -0.20f}}, {"PantryD", {-0.70f, -0.20f}},
    {"PantryE", { 0.00f, -0.20f}}, {"PantryF", { 0.70f, -0.20f}},
    {"PantryG", { 1.40f, -0.20f}}, {"PantryH", {-0.80f, -0.90f}},
    {"PantryI", { 0.00f, -0.90f}}, {"PantryJ", { 0.80f, -0.90f}},
  };
  for (auto&& p : PANTRIES_POS) {
    v.push_back(Node{p.first, p.second, NodeType::PANTRY});
    pantry_node_ids.push_back(static_cast<int>(v.size()-1));
  }

  // Pickups P1..P8
  const std::pair<const char*, std::array<float,2>> PICKUPS_POS[] = {
    {"P1", {-1.325f,  0.20f}}, {"P2", { 1.325f,  0.20f}},
    {"P3", {-0.35f,  -0.20f}}, {"P4", { 0.35f,  -0.20f}},
    {"P5", {-1.325f, -0.60f}}, {"P6", { 1.325f, -0.60f}},
    {"P7", {-0.40f,  -0.825f}},{"P8", { 0.40f,  -0.825f}},
  };
  for (auto&& p : PICKUPS_POS) {
    v.push_back(Node{p.first, p.second, NodeType::PICKUP});
    pickup_node_ids.push_back(static_cast<int>(v.size()-1));
  }

  return v;
}

std::vector<float> EurobotWorld::build_pairwise_D(const std::vector<Node>& nodes) {
  const int N = static_cast<int>(nodes.size());
  std::vector<float> D(N*N, 0.f);
  for (int i=0;i<N;i++) for (int j=0;j<N;j++) D[i*N + j] = L2(nodes[i].xy, nodes[j].xy);
  return D;
}

void EurobotWorld::build_action_space() {
  const int n_verbs  = 4;
  const int max_qty  = blue_robot.profile.max_action_qty;

  for (int v = 0; v < n_verbs; ++v) {
    const Verb verb = static_cast<Verb>(v);

    std::vector<int> nodes;
    switch (verb) {
      case Verb::PICK:
        nodes.assign(pickup_node_ids.begin(), pickup_node_ids.end());
        break;
      case Verb::PLACE:
        nodes.assign(pantry_node_ids.begin(), pantry_node_ids.end());
        nodes.push_back(blue_nest_node_idx);  // blue’s own nest
        break;
      case Verb::STEAL:
        if (allow_steal_) nodes.assign(pantry_node_ids.begin(), pantry_node_ids.end());
        break;
      case Verb::FLIP:
        nodes = { blue_nest_node_idx };       // single canonical node to shrink space
        break;
    }

    for (int n : nodes) {
      for (int c = 0; c < NUM_COLORS; ++c) {
        for (int q = 1; q <= max_qty; ++q) {
          action_space.push_back(Action{ v, n, c, q });
        }
      }
    }
  }
}

EurobotWorld::EurobotWorld(RobotProfile blue_prof,
                           RobotProfile yellow_prof,
                           RewardConfig rewards,
                           bool allow_steal,
                           uint64_t seed,
                           bool record_history)
: r_(rewards), record_history_(record_history), allow_steal_(allow_steal), rng_(seed) {
  nodes_ = build_nodes(blue_nest_node_idx, yellow_nest_node_idx, pantry_node_ids, pickup_node_ids);
  D_     = build_pairwise_D(nodes_);

  node_to_pantry_index_.assign(N(), -1);
  node_to_pickup_index_.assign(N(), -1);
  for (int j=0;j<(int)pantry_node_ids.size();++j) node_to_pantry_index_[pantry_node_ids[j]] = j;
  for (int j=0;j<(int)pickup_node_ids.size();++j)  node_to_pickup_index_[pickup_node_ids[j]]  = j;

  // TODO: compute this the right way (this is not correct)
  max_score_ = (r_.finish_in_nest_bonus + 
                r_.interest_bonus * 10 +
                r_.pantry_bonus * (8 * 4 - NEST_CAP) + 
                std::max(r_.pantry_bonus, r_.nest_bonus) * NEST_CAP); 

  blue_robot.tag    = "blue";   blue_robot.profile   = std::move(blue_prof);   blue_robot.curr_node   = blue_nest_node_idx;
  yellow_robot.tag  = "yellow"; yellow_robot.profile = std::move(yellow_prof); yellow_robot.curr_node = yellow_nest_node_idx;

  build_action_space();
  reset(seed);
}

void EurobotWorld::reset(std::optional<uint64_t> seed) {
  if (seed) rng_.seed(*seed);
  t_left_ = TIME_LIMIT_S;

  pantry_crate_counts.assign(pantry_node_ids.size(), std::array<int16_t,NUM_COLORS>{0,0});
  pickup_crate_counts.assign(pickup_node_ids.size(),  std::array<int16_t,NUM_COLORS>{0,0});
  blue_nest_count = 0; yellow_nest_count = 0;

  for (auto& p : pickup_crate_counts) { p[(int)Col::BLUE]   = 2; p[(int)Col::YELLOW] = 2; }

  blue_robot.inv = {0,0}; yellow_robot.inv = {0,0};
  blue_robot.event.reset(); yellow_robot.event.reset();
  blue_robot.curr_node = blue_nest_node_idx; yellow_robot.curr_node = yellow_nest_node_idx;
}

size_t EurobotWorld::obs_size() const {
  return size_t(2 * N())                                    // blue/yellow one-hots
       + size_t(NUM_COLORS)                                 // blue_inv
       + size_t(pantry_node_ids.size() * NUM_COLORS)        // pantries
       + size_t(pickup_node_ids.size()  * NUM_COLORS)       // pickups
       + size_t(2)                                          // nests
       + size_t(1);                                         // time remaining
}

torch::Tensor EurobotWorld::obs() const {
  // TODO: feature engineer more obs values
  const int Nn = N();
  const size_t K = obs_size();
  auto t = torch::zeros({(long)K}, torch::kFloat32);
  auto* buf = t.data_ptr<float>();

  size_t off = 0;

  // blue node one-hot [N]
  buf[off + blue_robot.curr_node] = 1.f;
  off += Nn;

  // yellow node one-hot [N]
  buf[off + yellow_robot.curr_node] = 1.f;
  off += Nn;

  // blue_inv [NUM_COLORS]
  for (int c = 0; c < NUM_COLORS; ++c) 
    buf[off++] = static_cast<float>(blue_robot.inv[c]) / blue_robot.profile.capacity;

  // pantries [len(pantry_node_ids)*NUM_COLORS]
  for (const auto& row : pantry_crate_counts) {
    for (int c = 0; c < NUM_COLORS; ++c) 
      buf[off++] = static_cast<float>(row[c]) / PANTRY_CAP;
  }

  // pickups [len(pickup_node_ids)*NUM_COLORS]
  for (const auto& row : pickup_crate_counts) {
    for (int c = 0; c < NUM_COLORS; ++c) 
      buf[off++] = static_cast<float>(row[c]) / 2.0; // 2 crates per pickup per color
  }
  
  // nests [2]
  buf[off++] = static_cast<float>(blue_nest_count) / NEST_CAP;
  buf[off++] = static_cast<float>(yellow_nest_count) / NEST_CAP;

  // time remaining
  buf[off++] = t_left_ / TIME_LIMIT_S;

  // make an owning tensor (clone)
  return t;
}

bool EurobotWorld::check_valid_action(Verb verb, int node, int color, int qty, const RobotState& robot) const {
  if (node < 0 || node >= N()) return false;
  if (color < 0 || color >= NUM_COLORS) return false;
  if (qty < 0 || qty > robot.profile.max_action_qty) return false;
  if (verb == Verb::FLIP && qty > robot.profile.max_flip_qty) return false;
  return true;
}

void EurobotWorld::schedule(RobotState& rob, const Action& a) {
  Verb verb = static_cast<Verb>(a.verb);
  int node = a.node, color = a.color, qty = a.qty;
  if (!check_valid_action(verb, node, color, qty, rob)) {
    throw std::runtime_error("Invalid action");
  }

  float d = (node != rob.curr_node) ? dist(rob.curr_node, node) : 0.f;
  double t_move = (verb == Verb::PICK ||
                   verb == Verb::PLACE||
                   verb == Verb::STEAL) && d>0.f
                  ? rob.profile.travel_time(d) : 0.0;
  double t_hand = rob.profile.handle_time(verb, qty);
  double t_total = t_move + t_hand;

  bool did_move = (t_move > 0.0);
  int actor_id  = (rob.tag=="blue") ? 0 : 1;

  if (t_total <= 0.0) {
    finish_event(rob.tag, static_cast<int>(verb), node, color, qty, did_move);
    return;
  }
  rob.event = Event{t_total, actor_id, static_cast<int>(verb), node, color, qty, did_move};
}

void EurobotWorld::schedule_yellow_scripted() {
  if (!yellow_policy) return;
  auto a = yellow_policy("yellow", *this, yellow_robot, rng_);
  if (!a) return;
  schedule(yellow_robot, *a);
}

void EurobotWorld::advance_until_next() {
  double tb = blue_robot.event  ? blue_robot.event->t_remaining  : std::numeric_limits<double>::infinity();
  double ty = yellow_robot.event? yellow_robot.event->t_remaining: std::numeric_limits<double>::infinity();
  double dt = std::min(tb, ty);
  if (!std::isfinite(dt) || dt <= 0.0) return;

  t_left_ = std::max(0.0, t_left_ - dt);
  if (blue_robot.event)   blue_robot.event->t_remaining   -= dt;
  if (yellow_robot.event) yellow_robot.event->t_remaining -= dt;

  const double eps = 1e-9;
  if (blue_robot.event && blue_robot.event->t_remaining <= eps) {
    auto e = *blue_robot.event; blue_robot.event.reset();
    finish_event("blue", e.verb, e.node, e.color, e.qty, e.did_move);
  }
  if (yellow_robot.event && yellow_robot.event->t_remaining <= eps) {
    auto e = *yellow_robot.event; yellow_robot.event.reset();
    finish_event("yellow", e.verb, e.node, e.color, e.qty, e.did_move);
  }
}

void EurobotWorld::finish_event(const std::string& actor_tag, int verb_i, int node, int color, int qty, bool did_move) {
  RobotState& rob = (actor_tag=="blue") ? blue_robot : yellow_robot;
  const RobotProfile& prof = rob.profile;
  if (did_move) rob.curr_node = node;

  Verb verb = static_cast<Verb>(verb_i);
  if (verb == Verb::PICK) {
    int idx = node_to_pickup_index_[node];
    if (idx != -1) {
      int can_take = pickup_crate_counts[idx][color];
      int inv_sum = rob.inv[0] + rob.inv[1];
      int room = std::max(0, prof.capacity - inv_sum);
      int take = std::max(0, std::min({qty, can_take, room}));
      if (take > 0) {
        pickup_crate_counts[idx][color] -= static_cast<int16_t>(take);
        rob.inv[color]       += static_cast<int16_t>(take);
      }
    }
  }

  if (verb == Verb::PLACE) {
    int have = rob.inv[color];
    int idx = node_to_pantry_index_[node];
    if (idx != -1) {
      int total_here = pantry_crate_counts[idx][0] + pantry_crate_counts[idx][1];
      int room = std::max(0, PANTRY_CAP - total_here);
      int put = std::max(0, std::min({qty, have, room}));
      if (put > 0) {
        pantry_crate_counts[idx][color] += static_cast<int16_t>(put);
        rob.inv[color]        -= static_cast<int16_t>(put);
      }
    }
    if (actor_tag=="blue" && node==blue_nest_node_idx) {
      int put = std::max(0, std::min(qty, have));
      int delta = std::min(put, std::max(0, NEST_CAP - blue_nest_count));
      if (delta > 0) { rob.inv[color] -= static_cast<int16_t>(delta); blue_nest_count += delta; }
    }
    if (actor_tag=="yellow" && node==yellow_nest_node_idx) {
      int put = std::max(0, std::min(qty, have));
      int delta = std::min(put, std::max(0, NEST_CAP - yellow_nest_count));
      if (delta > 0) { rob.inv[color] -= static_cast<int16_t>(delta); yellow_nest_count += delta; }
    }
  }

  if (verb == Verb::FLIP) {
    /*
    Flip action is defined as flip #qty color crates to opposing color.
    So if the action is (flip YELLOW 2), and current inv={2,2}, future inv={4,0}
    */
    if (!prof.can_flip) { /* no-op */ }
    else {
      int flip_cnt = std::min({qty, prof.max_flip_qty, static_cast<int>(rob.inv[color])});
      rob.inv[color] -= static_cast<int16_t>(flip_cnt);
      rob.inv[1-color] += static_cast<int16_t>(flip_cnt);
    }
  }

  if (verb == Verb::STEAL) {
    if (!allow_steal_) { /* no-op */ return; }
    int idx = node_to_pantry_index_[node];
    if (idx != -1) {
      int have = pantry_crate_counts[idx][color];
      int inv_sum = rob.inv[0] + rob.inv[1];
      int room = std::max(0, prof.capacity - inv_sum);
      int take = std::max(0, std::min({qty, have, room}));
      if (take > 0) {
        pantry_crate_counts[idx][color] -= static_cast<int16_t>(take);
        rob.inv[color]        += static_cast<int16_t>(take);
      }
    }
  }
}

bool EurobotWorld::step_blue(const Action& a) {
  if (!blue_robot.event)   schedule(blue_robot, a);
  if (!yellow_robot.event) schedule_yellow_scripted();

  while (t_left_ > 1e-9 && blue_robot.event) {
    advance_until_next();
    if (!yellow_robot.event && blue_robot.event) schedule_yellow_scripted();
  }
  return (t_left_ <= 1e-9);
}

/*
 * Public step function (simmilar to gym)
 */
std::pair<torch::Tensor, bool> EurobotWorld::step(const Action& a){
    bool done = step_blue(a);
    return {obs(), done};
}

std::pair<float,float> EurobotWorld::final_scores() const {
  auto sum_color = [](const std::vector<std::array<int16_t,NUM_COLORS>>& M, int c){
    int s=0; for (auto& r : M) s += r[c]; return s;
  };
  float blue_score   = sum_color(pantry_crate_counts, (int)Col::BLUE)   * r_.pantry_bonus + blue_nest_count   * r_.nest_bonus;
  float yellow_score = sum_color(pantry_crate_counts, (int)Col::YELLOW) * r_.pantry_bonus + yellow_nest_count * r_.nest_bonus;

  int blue_interest=0, yellow_interest=0;
  for (auto& p : pantry_crate_counts) {
    if (p[(int)Col::BLUE]   > p[(int)Col::YELLOW]) ++blue_interest;
    if (p[(int)Col::YELLOW] > p[(int)Col::BLUE])   ++yellow_interest;
  }
  blue_score   += r_.interest_bonus * blue_interest;
  yellow_score += r_.interest_bonus * yellow_interest;

  if (blue_robot.curr_node   == blue_nest_node_idx)  blue_score   += r_.finish_in_nest_bonus;
  if (yellow_robot.curr_node == yellow_nest_node_idx)  yellow_score += r_.finish_in_nest_bonus;

  return {blue_score, yellow_score};
}

std::pair<float,float> EurobotWorld::final_scores_norm() const {
  auto [b,y] = final_scores();
  return { b/max_score_, y/max_score_ };
}

EurobotState EurobotWorld::get_state() const {
  EurobotState s;
  s.t_left = t_left_;
  s.pantry_crate_counts = pantry_crate_counts;
  s.pickup_crate_counts  = pickup_crate_counts;
  s.blue_nest_count = blue_nest_count;
  s.yellow_nest_count = yellow_nest_count;
  s.blue_robot = blue_robot; s.yellow_robot = yellow_robot;
  return s;
}

void EurobotWorld::set_state(const EurobotState& s) {
  t_left_ = s.t_left;
  pantry_crate_counts = s.pantry_crate_counts;
  pickup_crate_counts  = s.pickup_crate_counts;
  blue_nest_count = s.blue_nest_count;
  yellow_nest_count = s.yellow_nest_count;
  blue_robot = s.blue_robot; yellow_robot = s.yellow_robot;
}


// -------- Action masks ---------
torch::Tensor EurobotWorld::legal_mask(bool blue_turn) const {
  const auto& rob = blue_turn ? blue_robot : yellow_robot;
  const int max_qty = blue_turn ? blue_robot.profile.max_action_qty : yellow_robot.profile.max_action_qty;
  const int nest_id = blue_turn ? blue_nest_node_idx : yellow_nest_node_idx;

  auto mask = torch::empty({static_cast<long>(action_space.size())},
                           torch::TensorOptions().dtype(torch::kBool));
  auto* mask_ptr = mask.data_ptr<bool>();

  const int inv_sum = int(rob.inv[0]) + int(rob.inv[1]);
  const int room_total = std::max(0, rob.profile.capacity - inv_sum);

  auto sum2 = [](const std::array<int16_t,2>& a){ return int(a[0]) + int(a[1]); };

  for (size_t i = 0; i < action_space.size(); ++i) {
    const auto& a = action_space[i];
    const auto verb  = static_cast<Verb>(a.verb);
    const int  node  = a.node;
    const int  color = a.color;
    const int  qty   = a.qty;

    bool legal = false;
    if (qty < 1 || qty > max_qty) { mask_ptr[i] = false; continue; }

    switch (verb) {
      case Verb::PICK: {
        // check if the node is a pickup node
        const int idx = node_to_pickup_index_[node];
        if (idx < 0) break;
        const int available = pickup_crate_counts[idx][color];
        legal = (available >= qty && room_total >= qty);
      } break;

      case Verb::PLACE: {
        const int have = rob.inv[color];
        if (node == nest_id) {
          const int nest_rem = std::max(0, NEST_CAP - (blue_turn ? blue_nest_count : yellow_nest_count));
          legal = (qty <= nest_rem && qty <= have);
          break;
        }
        const int pidx = node_to_pantry_index_[node];
        if (pidx < 0) break;
        const int room = std::max(0, PANTRY_CAP - sum2(pantry_crate_counts[pidx]));
        legal = (qty <= room && qty <= have);
      } break;

      case Verb::FLIP: {
        // Flip ignores node.
        const int max_flip = blue_turn ? blue_robot.profile.max_flip_qty : yellow_robot.profile.max_flip_qty;
        legal = (rob.profile.can_flip && qty <= max_flip && rob.inv[color] >= qty);
      } break;

      case Verb::STEAL: {
        if (!allow_steal_) break;
        const int pidx = node_to_pantry_index_[node];
        if (pidx < 0) break;
        const int have = pantry_crate_counts[pidx][color];
        legal = (qty <= room_total && qty <= have);
      } break;
    }

    mask_ptr[i] = legal;
  }

  return mask;
}

// Add to logits: 0 for legal, -inf (or neg_large) for illegal
torch::Tensor EurobotWorld::logit_mask(bool blue_turn, float neg_large) const {
    auto legal = legal_mask(blue_turn).to(torch::kFloat32);
    return (1.0f - legal) * neg_large;  // 0 for legal, -inf-ish for illegal
}

} // namespace euro
