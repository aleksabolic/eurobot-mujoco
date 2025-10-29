#include "alphazero/eurobot_render.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace eurobot {

namespace {

constexpr double TABLE_X_MIN = -1.5;
constexpr double TABLE_X_MAX = 1.5;
constexpr double TABLE_Y_MIN = -1.0;
constexpr double TABLE_Y_MAX = 1.0;
constexpr double PANTRY_HALF = 0.10;
constexpr double PICKUP_R = 0.075;
constexpr double NEST_HALF_X = 0.30;
constexpr double NEST_HALF_Y = 0.225;

const cv::Scalar BLUE(255, 102, 51);
const cv::Scalar YELL(51, 230, 255);
const cv::Scalar BLK(0, 0, 0);
const cv::Scalar WHT(255, 255, 255);
const cv::Scalar LBL(96, 96, 96);
const cv::Scalar FACE_BLUE(255, 210, 191);
const cv::Scalar FACE_YELLOW(179, 250, 255);
const cv::Scalar FACE_TIED(242, 242, 217);
const cv::Scalar FACE_AVAIL(242, 191, 242);
const cv::Scalar FACE_EMPTY(217, 217, 217);

template <typename T>
T clamp(T v, T lo, T hi) {
  return std::max(lo, std::min(v, hi));
}

}  // namespace

EurobotCV2Renderer::EurobotCV2Renderer(
    cv::Size canvas,
    std::optional<std::filesystem::path> background_path,
    double background_alpha)
    : canvas_(canvas),
      width_(canvas.width),
      height_(canvas.height),
      background_alpha_(background_alpha) {
  const double avail_w = static_cast<double>(width_ - pad_left_ - pad_right_);
  const double avail_h = static_cast<double>(height_ - pad_top_ - pad_bottom_);
  const double span_x = TABLE_X_MAX - TABLE_X_MIN;
  const double span_y = TABLE_Y_MAX - TABLE_Y_MIN;

  scale_ = avail_w > 0 && avail_h > 0 ? std::min(avail_w / span_x, avail_h / span_y) : 1.0;
  board_width_ = static_cast<int>(std::round(span_x * scale_));
  board_height_ = static_cast<int>(std::round(span_y * scale_));

  const int dx = std::max(0, static_cast<int>(std::floor((avail_w - board_width_) / 2.0)));
  const int dy = std::max(0, static_cast<int>(std::floor((avail_h - board_height_) / 2.0)));
  board_left_ = pad_left_ + dx;
  board_top_ = pad_top_ + dy;

  origin_x_ = board_left_ - TABLE_X_MIN * scale_;
  origin_y_ = (height_ - (board_top_ + board_height_)) - TABLE_Y_MIN * scale_;

  namespace fs = std::filesystem;

  fs::path bg_path;
  if (background_path) {
    bg_path = *background_path;
  } else {
    try {
      const fs::path here = fs::weakly_canonical(fs::path(__FILE__));
      bg_path = here.parent_path().parent_path().parent_path() / "assets" / "table.png";
    } catch (const fs::filesystem_error&) {
      bg_path.clear();
    }
  }

  if (!bg_path.empty() && fs::exists(bg_path) && board_width_ > 0 && board_height_ > 0) {
    cv::Mat img = cv::imread(bg_path.string(), cv::IMREAD_COLOR);
    if (!img.empty()) {
      cv::resize(img, table_image_, cv::Size(board_width_, board_height_), 0, 0, cv::INTER_AREA);
    }
  }
}

void EurobotCV2Renderer::set_window_title(std::string title) {
  window_title_ = std::move(title);
  window_created_ = false;
  window_centered_ = false;
}

EurobotState EurobotCV2Renderer::resolve_state(
    const EurobotWorld& world,
    const std::optional<EurobotState>& override) const {
  if (override) {
    return *override;
  }
  return world.get_state();
}

std::pair<double, double> EurobotCV2Renderer::compute_scores(
    const EurobotWorld& world,
    const EurobotState& state) const {
  const RewardConfig& rewards = world.reward_config();

  auto sum_color = [](const std::vector<std::array<int16_t, NUM_COLORS>>& matrix, int color) {
    int total = 0;
    for (const auto& row : matrix) {
      total += static_cast<int>(row[color]);
    }
    return total;
  };

  const int blue_pantry = sum_color(state.pantry_crate_counts, static_cast<int>(Col::BLUE));
  const int yellow_pantry = sum_color(state.pantry_crate_counts, static_cast<int>(Col::YELLOW));

  double blue_score = rewards.pantry_bonus * static_cast<double>(blue_pantry) +
                      rewards.nest_bonus * static_cast<double>(state.blue_nest_count);
  double yellow_score = rewards.pantry_bonus * static_cast<double>(yellow_pantry) +
                        rewards.nest_bonus * static_cast<double>(state.yellow_nest_count);

  for (const auto& row : state.pantry_crate_counts) {
    const int b = static_cast<int>(row[static_cast<int>(Col::BLUE)]);
    const int y = static_cast<int>(row[static_cast<int>(Col::YELLOW)]);
    if (b > y) {
      blue_score += rewards.interest_bonus;
    } else if (y > b) {
      yellow_score += rewards.interest_bonus;
    }
  }

  if (state.blue_robot.curr_node == world.blue_nest_node_idx) {
    blue_score += rewards.finish_in_nest_bonus;
  }
  if (state.yellow_robot.curr_node == world.yellow_nest_node_idx) {
    yellow_score += rewards.finish_in_nest_bonus;
  }

  return {blue_score, yellow_score};
}

cv::Point EurobotCV2Renderer::world_to_px(double x, double y) const {
  const int px = static_cast<int>(std::round(origin_x_ + x * scale_));
  const int py = static_cast<int>(std::round(height_ - (origin_y_ + y * scale_)));
  return cv::Point(px, py);
}

int EurobotCV2Renderer::length_to_px(double length) const {
  return static_cast<int>(std::round(length * scale_));
}

void EurobotCV2Renderer::ensure_window() {
  if (window_created_) {
    return;
  }
  cv::namedWindow(window_title_, cv::WINDOW_AUTOSIZE);
  window_created_ = true;
}

void EurobotCV2Renderer::try_center_window(const cv::Mat& frame) {
  if (window_centered_ || !window_created_) {
    return;
  }

  int window_w = frame.cols;
  int window_h = frame.rows;

#if CV_VERSION_MAJOR >= 4
  try {
    cv::Rect rect = cv::getWindowImageRect(window_title_);
    if (rect.width > 0 && rect.height > 0) {
      window_w = rect.width;
      window_h = rect.height;
    }
  } catch (...) {
    // ignored; fall back to frame size
  }
#endif

  int screen_w = 1920;
  int screen_h = 1080;

  if (const char* w_env = std::getenv("EUROBOT_SCREEN_WIDTH")) {
    screen_w = std::max(screen_w, std::atoi(w_env));
  }
  if (const char* h_env = std::getenv("EUROBOT_SCREEN_HEIGHT")) {
    screen_h = std::max(screen_h, std::atoi(h_env));
  }

  const int nx = std::max(0, (screen_w - window_w) / 2);
  const int ny = std::max(0, (screen_h - window_h) / 2);
  cv::moveWindow(window_title_, nx, ny);
  window_centered_ = true;
}

cv::Mat EurobotCV2Renderer::draw_snapshot(
    const EurobotWorld& world,
    std::optional<EurobotState> state_override,
    const std::vector<std::string>& recent_events,
    bool show,
    std::optional<double> blue_return) {
  EurobotState state = resolve_state(world, state_override);
  cv::Mat frame(height_, width_, CV_8UC3, WHT);

  // header background
  cv::rectangle(frame, cv::Point(0, 0), cv::Point(width_, pad_top_ - 10),
                cv::Scalar(240, 240, 240), cv::FILLED);

  if (!table_image_.empty()) {
    const int h = std::min(table_image_.rows, height_ - board_top_);
    const int w = std::min(table_image_.cols, width_ - board_left_);
    if (h > 0 && w > 0) {
      cv::Mat roi = frame(cv::Rect(board_left_, board_top_, w, h));
      const cv::Mat overlay = table_image_(cv::Rect(0, 0, w, h));
      cv::addWeighted(overlay, background_alpha_, roi, 1.0 - background_alpha_, 0.0, roi);
    }
  }

  // border
  const cv::Point border_min = world_to_px(TABLE_X_MIN, TABLE_Y_MIN);
  const cv::Point border_max = world_to_px(TABLE_X_MAX, TABLE_Y_MAX);
  cv::rectangle(frame, border_min, border_max, BLK, 1);

  const auto& nodes = world.nodes();

  // nests
  const int label_pad = 4;
  for (size_t idx = 0; idx < nodes.size(); ++idx) {
    const Node& node = nodes[idx];
    if (node.kind != NodeType::NEST) {
      continue;
    }

    const cv::Point p0 = world_to_px(node.xy[0] - NEST_HALF_X, node.xy[1] - NEST_HALF_Y);
    const cv::Point p1 = world_to_px(node.xy[0] + NEST_HALF_X, node.xy[1] + NEST_HALF_Y);
    const cv::Point tl(std::min(p0.x, p1.x), std::min(p0.y, p1.y));
    const cv::Point br(std::max(p0.x, p1.x), std::max(p0.y, p1.y));
    cv::rectangle(frame, tl, br, BLK, 1);

    cv::Scalar fill = FACE_BLUE;
    std::string label = "B" + std::to_string(state.blue_nest_count);
    if (static_cast<int>(idx) == world.yellow_nest_node_idx) {
      fill = FACE_YELLOW;
      label = "Y" + std::to_string(state.yellow_nest_count);
    }

    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.5;
    const int thickness = 1;
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

    const cv::Point box_tl(tl.x + label_pad, tl.y + label_pad);
    const cv::Point box_br(box_tl.x + text_size.width + label_pad * 2,
                           box_tl.y + text_size.height + baseline + label_pad);
    cv::rectangle(frame, box_tl, box_br, fill, cv::FILLED);

    const int text_x = box_tl.x + label_pad;
    const int text_y = box_br.y - label_pad - baseline;
    cv::putText(frame, label, cv::Point(text_x, text_y), font_face, font_scale, BLK, thickness,
                cv::LINE_AA);
  }

  // pantries
  for (size_t k = 0; k < world.pantry_node_ids.size(); ++k) {
    const int node_idx = world.pantry_node_ids[k];
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
      continue;
    }
    const Node& node = nodes[static_cast<size_t>(node_idx)];
    const cv::Point p0 = world_to_px(node.xy[0] - PANTRY_HALF, node.xy[1] - PANTRY_HALF);
    const cv::Point p1 = world_to_px(node.xy[0] + PANTRY_HALF, node.xy[1] + PANTRY_HALF);
    const int tl_x = std::min(p0.x, p1.x);
    const int tl_y = std::min(p0.y, p1.y);
    const int br_x = std::max(p0.x, p1.x);
    const int br_y = std::max(p0.y, p1.y);

    const auto& pantry_row = state.pantry_crate_counts.size() > k ? state.pantry_crate_counts[k]
                                                               : std::array<int16_t, NUM_COLORS>{0, 0};
    const int b = static_cast<int>(pantry_row[static_cast<int>(Col::BLUE)]);
    const int y = static_cast<int>(pantry_row[static_cast<int>(Col::YELLOW)]);

    const cv::Scalar fill = b > y ? FACE_BLUE : (y > b ? FACE_YELLOW : FACE_TIED);
    cv::rectangle(frame, cv::Point(tl_x, tl_y), cv::Point(br_x, br_y), fill, cv::FILLED);
    cv::rectangle(frame, cv::Point(tl_x, tl_y), cv::Point(br_x, br_y), BLK, 1);

    std::string pantry_letter = node.name;
    const std::string prefix = "Pantry";
    if (pantry_letter.rfind(prefix, 0) == 0) {
      pantry_letter = pantry_letter.substr(prefix.size());
    }

    int letter_baseline = 0;
    const cv::Size letter_size =
        cv::getTextSize(pantry_letter, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &letter_baseline);
    int letter_x =
        clamp(tl_x + label_pad, label_pad, width_ - letter_size.width - label_pad);
    int letter_y =
        clamp(tl_y + letter_size.height + label_pad, letter_size.height + label_pad,
              height_ - label_pad);

    cv::putText(frame, pantry_letter, cv::Point(letter_x, letter_y), cv::FONT_HERSHEY_SIMPLEX,
                0.4, LBL, 1, cv::LINE_AA);

    const std::string stock = "B" + std::to_string(b) + "/Y" + std::to_string(y);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(stock, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
    const int text_x = tl_x + (br_x - tl_x - text_size.width) / 2;
    const int text_y = tl_y + (br_y - tl_y + text_size.height) / 2;
    cv::putText(frame, stock, cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, BLK, 1,
                cv::LINE_AA);
  }

  // pickups
  for (size_t k = 0; k < world.pickup_node_ids.size(); ++k) {
    const int node_idx = world.pickup_node_ids[k];
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
      continue;
    }
    const Node& node = nodes[static_cast<size_t>(node_idx)];
    const cv::Point center = world_to_px(node.xy[0], node.xy[1]);
    const int radius = std::max(2, length_to_px(PICKUP_R));

    const auto& pickup_row = state.pickup_crate_counts.size() > k ? state.pickup_crate_counts[k]
                                                      : std::array<int16_t, NUM_COLORS>{0, 0};
    const int b = static_cast<int>(pickup_row[static_cast<int>(Col::BLUE)]);
    const int y = static_cast<int>(pickup_row[static_cast<int>(Col::YELLOW)]);
    const cv::Scalar fill = (b + y) > 0 ? FACE_AVAIL : FACE_EMPTY;

    cv::circle(frame, center, radius, fill, cv::FILLED);
    cv::circle(frame, center, radius, BLK, 1);

    const std::string name = node.name;
    int baseline = 0;
    const cv::Size name_size =
        cv::getTextSize(name, cv::FONT_HERSHEY_SIMPLEX, 0.35, 1, &baseline);
    int name_x = center.x - name_size.width / 2;
    int name_y = center.y - radius - label_pad;
    if (name_y - name_size.height - baseline < 0) {
      name_y = center.y + radius + name_size.height + label_pad;
      if (name_y > height_ - label_pad) {
        name_y = center.y + name_size.height / 2;
      }
    }
    name_x = clamp(name_x, label_pad, width_ - name_size.width - label_pad);
    cv::putText(frame, name, cv::Point(name_x, name_y), cv::FONT_HERSHEY_SIMPLEX, 0.35, LBL, 1,
                cv::LINE_AA);

    const std::string stock = "B" + std::to_string(b) + " Y" + std::to_string(y);
    const cv::Size stock_size =
        cv::getTextSize(stock, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
    const int stock_x = center.x - stock_size.width / 2;
    const int stock_y = center.y + stock_size.height / 2;
    cv::putText(frame, stock, cv::Point(stock_x, stock_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, BLK, 1,
                cv::LINE_AA);
  }

  // robots
  const auto draw_robot = [&](const Node& node, const cv::Scalar& color) {
    const cv::Point center = world_to_px(node.xy[0], node.xy[1]);
    const int radius = std::max(3, length_to_px(0.12));
    cv::Mat overlay = frame.clone();
    cv::circle(overlay, center, radius, color, cv::FILLED);
    cv::addWeighted(overlay, 0.55, frame, 0.45, 0.0, frame);
    cv::circle(frame, center, radius, color, 2);
    cv::circle(frame, center, radius, BLK, 2);
  };

  if (state.blue_robot.curr_node >= 0 && state.blue_robot.curr_node < static_cast<int>(nodes.size())) {
    draw_robot(nodes[static_cast<size_t>(state.blue_robot.curr_node)], BLUE);
  }
  if (state.yellow_robot.curr_node >= 0 && state.yellow_robot.curr_node < static_cast<int>(nodes.size())) {
    draw_robot(nodes[static_cast<size_t>(state.yellow_robot.curr_node)], YELL);
  }

  // header text
  const int header_x = pad_left_;
  const int header_y = 24;
  const auto inv_blue = state.blue_robot.inv;
  const auto inv_yell = state.yellow_robot.inv;

  cv::putText(frame,
              cv::format("t_left = %.1fs", state.t_left),
              cv::Point(header_x, header_y),
              cv::FONT_HERSHEY_SIMPLEX,
              0.6,
              BLK,
              2,
              cv::LINE_AA);
  cv::putText(frame,
              cv::format("BLUE inv: B%d Y%d",
                         static_cast<int>(inv_blue[static_cast<int>(Col::BLUE)]),
                         static_cast<int>(inv_blue[static_cast<int>(Col::YELLOW)])),
              cv::Point(header_x, header_y + 18),
              cv::FONT_HERSHEY_SIMPLEX,
              0.5,
              BLK,
              1,
              cv::LINE_AA);
  cv::putText(frame,
              cv::format("YELL inv: B%d Y%d",
                         static_cast<int>(inv_yell[static_cast<int>(Col::BLUE)]),
                         static_cast<int>(inv_yell[static_cast<int>(Col::YELLOW)])),
              cv::Point(header_x, header_y + 36),
              cv::FONT_HERSHEY_SIMPLEX,
              0.5,
              BLK,
              1,
              cv::LINE_AA);

  const auto [blue_score, yellow_score] = compute_scores(world, state);
  const int score_x = pad_left_ + 220;
  std::vector<std::string> score_texts = {
      cv::format("BLUE score: %.1f", blue_score),
      cv::format("YELL score: %.1f", yellow_score),
  };
  if (blue_return) {
    score_texts.push_back(cv::format("BLUE RL return: %.3f", *blue_return));
  }

  int score_max_width = 0;
  for (size_t i = 0; i < score_texts.size(); ++i) {
    cv::putText(frame, score_texts[i], cv::Point(score_x, header_y + 18 * static_cast<int>(i)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, BLK, 1, cv::LINE_AA);
    int baseline = 0;
    const cv::Size text_size =
        cv::getTextSize(score_texts[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    score_max_width = std::max(score_max_width, text_size.width);
  }

  const int event_x = score_x + score_max_width + 100;
  const int max_events = 3;
  const int events_to_draw = std::min<int>(static_cast<int>(recent_events.size()), max_events);
  for (int i = 0; i < events_to_draw; ++i) {
    const std::string& label = recent_events[static_cast<size_t>(recent_events.size() - 1 - i)];
    cv::putText(frame, label, cv::Point(event_x, header_y + 16 * i),
                cv::FONT_HERSHEY_SIMPLEX, 0.43, BLK, 1, cv::LINE_AA);
  }

  if (show) {
    ensure_window();
    cv::imshow(window_title_, frame);
    cv::waitKey(1);
    try_center_window(frame);
  }

  return frame;
}

}  // namespace eurobot
