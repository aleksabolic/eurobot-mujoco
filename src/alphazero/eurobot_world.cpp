#include "alphazero/eurobot_world.hpp"
#include <algorithm>
#include <cassert>

namespace eurobot {

static inline float L2(const std::array<float,2>& a, const std::array<float,2>& b) {
  float dx = a[0]-b[0], dy = a[1]-b[1];
  return std::sqrt(dx*dx + dy*dy);
}

std::vector<Node> EurobotWorld::build_nodes(int& nest_blue, int& nest_yellow,
                                            std::vector<int>& pantries,
                                            std::vector<int>& pickups) {
  std::vector<Node> v;
  // Nests
  v.push_back(Node{"NestBlue",   { 1.20f,  0.775f}, NodeType::NEST});
  v.push_back(Node{"NestYellow", {-1.20f,  0.775f}, NodeType::NEST});
  nest_blue  = 0;
  nest_yellow= 1;

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
    pantries.push_back(static_cast<int>(v.size()-1));
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
    pickups.push_back(static_cast<int>(v.size()-1));
  }

  return v;
}

std::vector<float> EurobotWorld::build_pairwise_D(const std::vector<Node>& nodes) {
  const int N = static_cast<int>(nodes.size());
  std::vector<float> D(N*N, 0.f);
  for (int i=0;i<N;i++) for (int j=0;j<N;j++) D[i*N + j] = L2(nodes[i].xy, nodes[j].xy);
  return D;
}

std::vector<Action> EurobotWorld::action_space() const {
  const int n_verbs  = 4;
  const int max_qty  = blue.profile.max_action_qty;

  std::vector<Action> actions;

  for (int v = 0; v < n_verbs; ++v) {
    const Verb verb = static_cast<Verb>(v);

    std::vector<int> nodes;
    switch (verb) {
      case Verb::PICK:
        nodes.assign(PICKUPS.begin(), PICKUPS.end());
        break;
      case Verb::PLACE:
        nodes.assign(PANTRIES.begin(), PANTRIES.end());
        nodes.push_back(NEST_BLUE);  // blue’s own nest
        break;
      case Verb::STEAL:
        if (allow_steal_) nodes.assign(PANTRIES.begin(), PANTRIES.end());
        break;
      case Verb::FLIP:
        nodes = { NEST_BLUE };       // single canonical node to shrink space
        break;
    }

    for (int n : nodes) {
      for (int c = 0; c < NUM_COLORS; ++c) {
        for (int q = 1; q <= max_qty; ++q) {
          actions.push_back(Action{ v, n, c, q });
        }
      }
    }
  }
  return actions;
}

EurobotWorld::EurobotWorld(RobotProfile blue_prof,
                           RobotProfile yellow_prof,
                           RewardConfig rewards,
                           bool allow_steal,
                           uint64_t seed,
                           bool record_history)
: r_(rewards), record_history_(record_history), allow_steal_(allow_steal), rng_(seed) {
  nodes_ = build_nodes(NEST_BLUE, NEST_YELL, PANTRIES, PICKUPS);
  D_     = build_pairwise_D(nodes_);

  pantry_idx_.assign(N(), -1);
  pickup_idx_.assign(N(), -1);
  for (int j=0;j<(int)PANTRIES.size();++j) pantry_idx_[PANTRIES[j]] = j;
  for (int j=0;j<(int)PICKUPS.size();++j)  pickup_idx_[PICKUPS[j]]  = j;

  blue.tag    = "blue";   blue.profile   = std::move(blue_prof);   blue.node   = NEST_BLUE;
  yellow.tag  = "yellow"; yellow.profile = std::move(yellow_prof); yellow.node = NEST_YELL;
  reset(seed);
}

void EurobotWorld::reset(std::optional<uint64_t> seed) {
  if (seed) rng_.seed(*seed);
  t_left_ = TIME_LIMIT_S;

  pantries.assign(PANTRIES.size(), std::array<int16_t,NUM_COLORS>{0,0});
  pickups.assign(PICKUPS.size(),  std::array<int16_t,NUM_COLORS>{0,0});
  nest_blue = 0; nest_yellow = 0;

  for (auto& p : pickups) { p[(int)Col::BLUE]   = 2; p[(int)Col::YELLOW] = 2; }

  blue.inv = {0,0}; yellow.inv = {0,0};
  blue.event.reset(); yellow.event.reset();
  blue.node = NEST_BLUE; yellow.node = NEST_YELL;
}

size_t EurobotWorld::obs_size() const {
  return size_t(2 * N())                                    // blue/yellow one-hots
       + size_t(NUM_COLORS)                                 // blue_inv
       + size_t(PANTRIES.size() * NUM_COLORS)               // pantries
       + size_t(PICKUPS.size()  * NUM_COLORS)               // pickups
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
  if (blue.node >= 0 && blue.node < Nn) buf[off + blue.node] = 1.f;
  off += Nn;

  // yellow node one-hot [N]
  if (yellow.node >= 0 && yellow.node < Nn) buf[off + yellow.node] = 1.f;
  off += Nn;

  // blue_inv [NUM_COLORS]
  for (int c = 0; c < NUM_COLORS; ++c) buf[off++] = static_cast<float>(blue.inv[c]);

  // pantries [len(PANTRIES)*NUM_COLORS]
  for (const auto& row : pantries) {
    for (int c = 0; c < NUM_COLORS; ++c) buf[off++] = static_cast<float>(row[c]);
  }

  // pickups [len(PICKUPS)*NUM_COLORS]
  for (const auto& row : pickups) {
    for (int c = 0; c < NUM_COLORS; ++c) buf[off++] = static_cast<float>(row[c]);
  }

  // time remaining
  buf[off++] = t_left_;

  // make an owning tensor (clone)
  return t;
}

bool EurobotWorld::check_valid_action(int node, int color, int qty, const RobotState& robot) const {
  if (node < 0 || node >= N()) return false;
  if (color < 0 || color >= NUM_COLORS) return false;
  if (qty < 0 || qty > robot.profile.max_action_qty) return false;
  return true;
}

void EurobotWorld::schedule(RobotState& rob, const Action& a) {
  int verb  = a.verb, node = a.node, color = a.color, qty = a.qty;
  // TODO: this sometimes happens, check why and fix it
  if (!check_valid_action(node, color, qty, rob)) {
    throw std::runtime_error("Invalid action");
  }

  float d = (node != rob.node) ? dist(rob.node, node) : 0.f;
  double t_move = (static_cast<Verb>(verb)==Verb::PICK ||
                   static_cast<Verb>(verb)==Verb::PLACE||
                   static_cast<Verb>(verb)==Verb::STEAL) && d>0.f
                  ? rob.profile.travel_time(d) : 0.0;
  double t_hand = rob.profile.handle_time(static_cast<Verb>(verb), qty);
  double t_total = t_move + t_hand;

  bool did_move = (t_move > 0.0);
  int actor_id  = (rob.tag=="blue") ? 0 : 1;

  // TODO: redundant
  if (t_total <= 0.0) {
    finish_event(rob.tag, verb, node, color, qty, did_move);
    return;
  }
  rob.event = Event{t_total, actor_id, verb, node, color, qty, did_move};
}

void EurobotWorld::schedule_yellow_scripted() {
  if (!yellow_policy) return;
  auto a = yellow_policy("yellow", *this, yellow, rng_);
  if (!a) return;
  schedule(yellow, *a);
}

void EurobotWorld::advance_until_next() {
  double tb = blue.event  ? blue.event->t_remaining  : std::numeric_limits<double>::infinity();
  double ty = yellow.event? yellow.event->t_remaining: std::numeric_limits<double>::infinity();
  double dt = std::min(tb, ty);
  if (!std::isfinite(dt) || dt <= 0.0) return;

  t_left_ = std::max(0.0, t_left_ - dt);
  if (blue.event)   blue.event->t_remaining   -= dt;
  if (yellow.event) yellow.event->t_remaining -= dt;

  const double eps = 1e-9;
  if (blue.event && blue.event->t_remaining <= eps) {
    auto e = *blue.event; blue.event.reset();
    finish_event("blue", e.verb, e.node, e.color, e.qty, e.did_move);
  }
  if (yellow.event && yellow.event->t_remaining <= eps) {
    auto e = *yellow.event; yellow.event.reset();
    finish_event("yellow", e.verb, e.node, e.color, e.qty, e.did_move);
  }
}

void EurobotWorld::finish_event(const std::string& actor_tag, int verb_i, int node, int color, int qty, bool did_move) {
  RobotState& rob = (actor_tag=="blue") ? blue : yellow;
  const RobotProfile& prof = rob.profile;
  if (did_move) rob.node = node;

  Verb verb = static_cast<Verb>(verb_i);
  if (verb == Verb::PICK) {
    int idx = pickup_idx_[node];
    if (idx != -1) {
      int can_take = pickups[idx][color];
      int inv_sum = rob.inv[0] + rob.inv[1];
      int room = std::max(0, prof.capacity - inv_sum);
      int take = std::max(0, std::min({qty, can_take, room}));
      if (take > 0) {
        pickups[idx][color] -= static_cast<int16_t>(take);
        rob.inv[color]       += static_cast<int16_t>(take);
      }
    }
  }

  if (verb == Verb::PLACE) {
    int have = rob.inv[color];
    int idx = pantry_idx_[node];
    if (idx != -1) {
      int total_here = pantries[idx][0] + pantries[idx][1];
      int room = std::max(0, PANTRY_CAP - total_here);
      int put = std::max(0, std::min({qty, have, room}));
      if (put > 0) {
        pantries[idx][color] += static_cast<int16_t>(put);
        rob.inv[color]        -= static_cast<int16_t>(put);
      }
    }
    if (actor_tag=="blue" && node==NEST_BLUE) {
      int put = std::max(0, std::min(qty, have));
      int delta = std::min(put, std::max(0, NEST_CAP - nest_blue));
      if (delta > 0) { rob.inv[color] -= static_cast<int16_t>(delta); nest_blue += delta; }
    }
    if (actor_tag=="yellow" && node==NEST_YELL) {
      int put = std::max(0, std::min(qty, have));
      int delta = std::min(put, std::max(0, NEST_CAP - nest_yellow));
      if (delta > 0) { rob.inv[color] -= static_cast<int16_t>(delta); nest_yellow += delta; }
    }
  }

  if (verb == Verb::FLIP) {
    /*
    Flip action is defined as flip #qty color crates to opposing color.
    So if the action is (flip YELLOW 2), and current inv={2,2}, future inv={4,0}
    */
    if (!prof.can_flip) { /* no-op */ }
    else {
      int flip_cnt = std::min(qty, (int)rob.inv[color]);
      rob.inv[color] -= static_cast<int16_t>(flip_cnt);
      rob.inv[1-color] += static_cast<int16_t>(flip_cnt);
    }
  }

  if (verb == Verb::STEAL) {
    if (!allow_steal_) { /* no-op */ return; }
    int idx = pantry_idx_[node];
    if (idx != -1) {
      int have = pantries[idx][color];
      int inv_sum = rob.inv[0] + rob.inv[1];
      int room = std::max(0, prof.capacity - inv_sum);
      int take = std::max(0, std::min({qty, have, room}));
      if (take > 0) {
        pantries[idx][color] -= static_cast<int16_t>(take);
        rob.inv[color]        += static_cast<int16_t>(take);
      }
    }
  }
}

bool EurobotWorld::step_blue(const Action& a) {
  if (!blue.event)   schedule(blue, a);
  if (!yellow.event) schedule_yellow_scripted();

  while (t_left_ > 1e-9 && blue.event) {
    advance_until_next();
    if (!yellow.event && blue.event) schedule_yellow_scripted();
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
  float blue_score   = sum_color(pantries, (int)Col::BLUE)   * r_.pantry_bonus + nest_blue   * r_.nest_bonus;
  float yellow_score = sum_color(pantries, (int)Col::YELLOW) * r_.pantry_bonus + nest_yellow * r_.nest_bonus;

  int blue_interest=0, yellow_interest=0;
  for (auto& p : pantries) {
    if (p[(int)Col::BLUE]   > p[(int)Col::YELLOW]) ++blue_interest;
    if (p[(int)Col::YELLOW] > p[(int)Col::BLUE])   ++yellow_interest;
  }
  blue_score   += r_.interest_bonus * blue_interest;
  yellow_score += r_.interest_bonus * yellow_interest;

  if (blue.node   == NEST_BLUE)  blue_score   += r_.finish_in_nest_bonus;
  if (yellow.node == NEST_YELL)  yellow_score += r_.finish_in_nest_bonus;

  return {blue_score, yellow_score};
}

std::pair<float,float> EurobotWorld::final_scores_norm() const {
  auto [b,y] = final_scores();
  return { b/160.f, y/160.f };
}

EurobotState EurobotWorld::get_state() const {
  EurobotState s;
  s.t_left = t_left_;
  s.pantries = pantries;
  s.pickups  = pickups;
  s.nest_blue = nest_blue;
  s.nest_yellow = nest_yellow;
  s.blue = blue; s.yellow = yellow;
  return s;
}

void EurobotWorld::set_state(const EurobotState& s) {
  t_left_ = s.t_left;
  pantries = s.pantries;
  pickups  = s.pickups;
  nest_blue = s.nest_blue;
  nest_yellow = s.nest_yellow;
  blue = s.blue; yellow = s.yellow;
}

} // namespace euro
