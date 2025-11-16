#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "alphazero/eurobot_world.hpp"

namespace eurobot {

#if defined(EUROBOT_HAVE_X11)
class X11Window;
#endif

/**
 * Lightweight OpenCV renderer that mirrors the Python EurobotCV2Renderer.
 * It renders the current world snapshot (or a provided EurobotState) to a cv::Mat
 * and optionally shows the frame in a named window.
 */
 class EurobotCV2Renderer {
 public:
  EurobotCV2Renderer(
      cv::Size canvas = cv::Size(900, 600),
      std::optional<std::filesystem::path> background_path = std::nullopt,
      double background_alpha = 0.35);
  ~EurobotCV2Renderer();

  cv::Mat draw_snapshot(
      const EurobotWorld& world,
      std::optional<EurobotState> state_override = std::nullopt,
      const std::vector<std::string>& recent_events = {},
      bool show = false,
      std::optional<double> blue_return = std::nullopt);

  void set_window_title(std::string title);

 private:
  cv::Point world_to_px(double x, double y) const;
  int length_to_px(double length) const;
  std::pair<double, double> compute_scores(const EurobotWorld& world,
                                           const EurobotState& state) const;
  EurobotState resolve_state(const EurobotWorld& world,
                             const std::optional<EurobotState>& override) const;
  void ensure_window();
  void try_center_window(const cv::Mat& frame);
  void show_frame(const cv::Mat& frame, bool show);
  void destroy_window();

 private:
  cv::Size canvas_;
  int width_ = 0;
  int height_ = 0;

  int pad_left_ = 20;
  int pad_right_ = 20;
  int pad_bottom_ = 20;
  int pad_top_ = 80;  // reserve space for info banner

  double scale_ = 1.0;
  int board_width_ = 0;
  int board_height_ = 0;
  int board_left_ = 0;
  int board_top_ = 0;
  double origin_x_ = 0.0;
  double origin_y_ = 0.0;

  double background_alpha_ = 0.35;
  cv::Mat table_image_;

  std::string window_title_ = "Eurobot (cv2)";
  bool window_created_ = false;
  bool window_centered_ = false;
#if defined(EUROBOT_HAVE_X11)
  std::unique_ptr<class X11Window> x11_window_;
#endif
};

}  // namespace eurobot
