#include "alphazero/tensorboard_logger.hpp"

#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace alphazero {

namespace {

uint64_t double_to_bits(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(double));
  return bits;
}

uint32_t float_to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(float));
  return bits;
}

int current_process_id() {
#ifdef _WIN32
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

std::string default_run_subdir(const std::string& run_name) {
  if (run_name.empty()) return {};

  std::string sanitized;
  sanitized.reserve(run_name.size());
  for (char c : run_name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '/') {
      sanitized.push_back(c);
    } else {
      sanitized.push_back('_');
    }
  }
  return sanitized;
}

const std::array<uint32_t, 256>& crc32c_table() {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> tbl{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t crc = i;
      for (int k = 0; k < 8; ++k) {
        if (crc & 1U) {
          crc = (crc >> 1U) ^ 0x82F63B78U;
        } else {
          crc >>= 1U;
        }
      }
      tbl[i] = crc;
    }
    return tbl;
  }();
  return table;
}

} // namespace

TensorboardLogger::TensorboardLogger(const std::string& log_dir, const std::string& run_name) {
  namespace fs = std::filesystem;

  fs::path base = log_dir.empty() ? fs::path("runs") : fs::path(log_dir);

  const auto run_subdir = default_run_subdir(run_name);
  if (!run_subdir.empty()) {
    base /= run_subdir;
  }
  fs::create_directories(base);

  const auto now = std::chrono::system_clock::now();
  const auto ts_seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  const auto ts_nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

  std::string file_name = "events.out.tfevents." + std::to_string(ts_seconds) + "." + std::to_string(current_process_id()) + "." + std::to_string(ts_nanos);
  file_path_ = (base / file_name).string();

  stream_.open(file_path_, std::ios::binary | std::ios::out);
  if (!stream_) {
    throw std::runtime_error("TensorboardLogger: failed to open event file at " + file_path_);
  }

  const double wall_time = std::chrono::duration<double>(now.time_since_epoch()).count();
  write_event(make_file_version_event(wall_time));
}

TensorboardLogger::~TensorboardLogger() {
  if (stream_.is_open()) {
    stream_.flush();
    stream_.close();
  }
}

void TensorboardLogger::add_scalar(const std::string& tag, int64_t step, double value) {
  if (!stream_) return;
  const auto now = std::chrono::system_clock::now();
  const double wall_time = std::chrono::duration<double>(now.time_since_epoch()).count();
  auto event_bytes = make_scalar_event(tag, step, wall_time, value);
  write_event(event_bytes);
}

void TensorboardLogger::write_event(const std::string& event_bytes) {
  if (!stream_) return;

  const uint64_t length = static_cast<uint64_t>(event_bytes.size());
  char length_bytes[sizeof(uint64_t)];
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    length_bytes[i] = static_cast<char>((length >> (8 * i)) & 0xFFU);
  }

  const uint32_t length_crc = masked_crc32c(length_bytes, sizeof(uint64_t));
  const uint32_t data_crc = masked_crc32c(event_bytes.data(), event_bytes.size());

  char length_crc_bytes[sizeof(uint32_t)];
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    length_crc_bytes[i] = static_cast<char>((length_crc >> (8 * i)) & 0xFFU);
  }

  char data_crc_bytes[sizeof(uint32_t)];
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    data_crc_bytes[i] = static_cast<char>((data_crc >> (8 * i)) & 0xFFU);
  }

  stream_.write(length_bytes, sizeof(uint64_t));
  stream_.write(length_crc_bytes, sizeof(uint32_t));
  stream_.write(event_bytes.data(), static_cast<std::streamsize>(event_bytes.size()));
  stream_.write(data_crc_bytes, sizeof(uint32_t));
  stream_.flush();
}

std::string TensorboardLogger::make_file_version_event(double wall_time) {
  std::string event;
  event.push_back(static_cast<char>((1 << 3) | 1));
  append_fixed64(double_to_bits(wall_time), event);

  event.push_back(static_cast<char>((2 << 3) | 0));
  append_varint(0, event);

  const std::string version = "brain.Event:2";
  event.push_back(static_cast<char>((3 << 3) | 2));
  append_varint(static_cast<uint64_t>(version.size()), event);
  event.append(version);

  return event;
}

std::string TensorboardLogger::make_scalar_event(const std::string& tag,
                                                 int64_t step,
                                                 double wall_time,
                                                 double value) {
  std::string event;
  event.push_back(static_cast<char>((1 << 3) | 1));
  append_fixed64(double_to_bits(wall_time), event);

  event.push_back(static_cast<char>((2 << 3) | 0));
  append_varint(static_cast<uint64_t>(step), event);

  auto summary = make_summary_scalar(tag, static_cast<float>(value));

  event.push_back(static_cast<char>((5 << 3) | 2));
  append_varint(static_cast<uint64_t>(summary.size()), event);
  event.append(summary);

  return event;
}

std::string TensorboardLogger::make_summary_scalar(const std::string& tag, float value) {
  std::string summary_value;
  summary_value.push_back(static_cast<char>((1 << 3) | 2));
  append_varint(static_cast<uint64_t>(tag.size()), summary_value);
  summary_value.append(tag);

  summary_value.push_back(static_cast<char>((2 << 3) | 5));
  append_fixed32(float_to_bits(value), summary_value);

  std::string summary;
  summary.push_back(static_cast<char>((1 << 3) | 2));
  append_varint(static_cast<uint64_t>(summary_value.size()), summary);
  summary.append(summary_value);

  return summary;
}

void TensorboardLogger::append_varint(uint64_t value, std::string& out) {
  while (value >= 0x80U) {
    out.push_back(static_cast<char>((value & 0x7FU) | 0x80U));
    value >>= 7U;
  }
  out.push_back(static_cast<char>(value & 0x7FU));
}

void TensorboardLogger::append_fixed32(uint32_t value, std::string& out) {
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFU));
  }
}

void TensorboardLogger::append_fixed64(uint64_t value, std::string& out) {
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xFFU));
  }
}

uint32_t TensorboardLogger::crc32c(const void* data, size_t length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  uint32_t crc = 0xFFFFFFFFU;
  const auto& table = crc32c_table();
  for (size_t i = 0; i < length; ++i) {
    const uint8_t idx = static_cast<uint8_t>((crc ^ bytes[i]) & 0xFFU);
    crc = (crc >> 8U) ^ table[idx];
  }
  return crc ^ 0xFFFFFFFFU;
}

uint32_t TensorboardLogger::masked_crc32c(const void* data, size_t length) {
  const uint32_t crc = crc32c(data, length);
  return ((crc >> 15U) | (crc << 17U)) + 0xA282EAD8U;
}

} // namespace alphazero
