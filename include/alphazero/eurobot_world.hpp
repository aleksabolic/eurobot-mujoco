#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <stdexcept>
#include <torch/torch.h>

namespace eurobot {

// constants
inline constexpr int NEST_CAP = 6;
inline constexpr int PANTRY_CAP = 8;
inline constexpr double TIME_LIMIT_S = 100.0;
inline constexpr int NUM_COLORS = 2;

enum class NodeType : int { NEST=0, PANTRY=1, PICKUP=2 };
enum class Col      : int { BLUE=0, YELLOW=1 };

enum class Verb : int { PICK=0, PLACE=1, FLIP=2, STEAL=3 };

struct Node {
  std::string name;
  std::array<float,2> xy{};
  NodeType kind{NodeType::PANTRY};
};

struct Action {
  int verb;   // cast to Verb
  int node;
  int color;  // cast to Col
  int qty;
};

struct RewardConfig {
  float nest_bonus = 2.0;        // + per counted crate in nest (cap)
  float pantry_bonus = 3.0;      // + per valid pantry crate (BLUE placed by BLUE)
  float interest_bonus = 5.0;    // + per pantry where BLUE has strict BLUE majority at end
  float finish_in_nest_bonus = 10.0;
};

struct RobotProfile {
  int  capacity = 0;
  int  max_action_qty = 0;
  bool can_flip = false;
  std::function<double(double /*distance*/)> travel_time;
  std::function<double(Verb /*verb*/, int /*qty*/)> handle_time;
};

struct Event {
  double t_remaining = 0.0;
  int actor_id = -1; // 0=blue, 1=yellow
  int verb = 0, node = 0, color = 0, qty = 0;
  bool did_move = false;
};

struct RobotState {
  std::string tag;
  RobotProfile profile;
  int node = 0; // TODO: rename to curr_node
  std::array<int16_t,NUM_COLORS> inv{0,0};
  std::optional<Event> event;
};

struct EurobotState {
  double t_left = 0.0;
  std::vector<std::array<int16_t,NUM_COLORS>> pantries;
  std::vector<std::array<int16_t,NUM_COLORS>> pickups;
  int nest_blue = 0, nest_yellow = 0;
  RobotState blue, yellow;
};

class EurobotWorld {
  public:
    EurobotWorld(RobotProfile blue_prof,
                 RobotProfile yellow_prof,
                 RewardConfig rewards,
                 bool allow_steal = true,
                 uint64_t seed = 0,
                 bool record_history = false);
    void reset(std::optional<uint64_t> seed = std::nullopt);
    torch::Tensor obs() const;
    size_t obs_size() const;
    bool step_blue(const Action& a);
    std::pair<torch::Tensor, bool> step(const Action& a);

    EurobotState get_state() const;
    void set_state(const EurobotState& s);

    std::pair<float,float> final_scores() const;
    std::pair<float,float> final_scores_norm() const;

    // TODO: rename these badly named props
    int N() const { return static_cast<int>(nodes_.size()); }
    bool allow_steal() const {return allow_steal_;}
    int pantry_cap() const {return PANTRY_CAP;}
    int nest_cap() const {return NEST_CAP;}
    int num_colors() const {return NUM_COLORS;}
    int max_score() const {return max_score_;}

    // returns -1 if node at idx is not pantry and otherwise index of same pantry relative to world.pantries
    int pantry_idx(int idx) const {return pantry_idx_[idx];} 
    // returns -1 if node at idx is not pickup and otherwise index of same pickup relative to world.pickups
    int pickup_idx(int idx) const {return pickup_idx_[idx];}

    const std::vector<Node>& nodes() const { return nodes_; }
    const RewardConfig& reward_config() const { return r_; }

    // TODO: rename this 
    int NEST_BLUE = -1, NEST_YELL = -1; 
    std::vector<int> PANTRIES, PICKUPS;
    std::vector<Action> action_space;

    // Scripted yellow hook: (actor_tag, world, robot, rng) -> optional<Action>
    std::function<std::optional<Action>(const std::string&,
                                        const EurobotWorld&,
                                        const RobotState&,
                                        std::mt19937_64&)> yellow_policy;

    // TODO: rename                                   
    RobotState blue, yellow;
    std::vector<std::array<int16_t,NUM_COLORS>> pantries, pickups;
    int nest_blue = 0, nest_yellow = 0;

    // Bool mask (shape: [num_actions]) over all possible actions
    torch::Tensor legal_mask(bool blue_turn = true) const;

    // Add to logits: 0 for legal, -inf (or neg_large) for illegal
    torch::Tensor logit_mask(bool blue_turn = true, float neg_large = -1e9f) const;

  private:
    void schedule(RobotState& rob, const Action& a);
    void schedule_yellow_scripted();
    void advance_until_next();
    void finish_event(const std::string& actor_tag, int verb, int node, int color, int qty, bool did_move);

    //utils
    bool check_valid_action(int node, int color, int qty, const RobotState& robot) const;
    inline float dist(int i, int j) const { return D_[i*N() + j]; }
    static std::vector<Node> build_nodes(int& nest_blue, int& nest_yellow,
                                           std::vector<int>& pantries,
                                           std::vector<int>& pickups);
    static std::vector<float> build_pairwise_D(const std::vector<Node>& nodes);
    void build_action_space();

  private:
    RewardConfig r_;
    bool record_history_ = false;
    bool allow_steal_ = true;

    double t_left_ = 0.0;
    std::mt19937_64 rng_{12345};

    std::vector<Node>  nodes_;
    std::vector<float> D_; // row-major NxN

    std::vector<int> pantry_idx_, pickup_idx_; // size N()

    float max_score_ = 156.0; // theoretical maximum final score for single agent for this env
};
}
