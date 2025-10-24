#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>

namespace alphazero {

class TensorboardLogger {
public:
  TensorboardLogger(const std::string& log_dir, const std::string& run_name = "");
  ~TensorboardLogger();

  TensorboardLogger(const TensorboardLogger&) = delete;
  TensorboardLogger& operator=(const TensorboardLogger&) = delete;

  void add_scalar(const std::string& tag, int64_t step, double value);

  const std::string& file_path() const { return file_path_; }

private:
  void write_event(const std::string& event_bytes);
  static std::string make_file_version_event(double wall_time);
  static std::string make_scalar_event(const std::string& tag,
                                       int64_t step,
                                       double wall_time,
                                       double value);
  static std::string make_summary_scalar(const std::string& tag, float value);

  static void append_varint(uint64_t value, std::string& out);
  static void append_fixed32(uint32_t value, std::string& out);
  static void append_fixed64(uint64_t value, std::string& out);

  static uint32_t crc32c(const void* data, size_t length);
  static uint32_t masked_crc32c(const void* data, size_t length);

  std::ofstream stream_;
  std::string file_path_;
};

} // namespace alphazero
