#pragma once
#include <deque>
#include <random>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <torch/torch.h>
#include <torch/serialize/archive.h>

namespace alphazero {

struct ReplaySample {
  torch::Tensor observations;
  torch::Tensor policies;
  torch::Tensor values;
};

class ReplayBuffer {
public:
  explicit ReplayBuffer(size_t capacity)
  : capacity_(capacity ? capacity : 1),
    rng_(std::random_device{}()) {}

  size_t size() const noexcept { return obs_.size(); }
  size_t capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return obs_.empty(); }

  // obs: 1D/ND float tensor; pi: 1D/ND float tensor; value: scalar float
  void add(const torch::Tensor& obs, const torch::Tensor& pi, float value) {
    TORCH_CHECK(obs.defined() && pi.defined(), "ReplayBuffer.add: undefined tensor");
    auto o = obs.detach().contiguous();
    auto p = pi.detach().contiguous();

    // enforce consistent shapes across entries
    if (obs_shape_.empty()) {
      obs_shape_ = o.sizes().vec();
      pi_shape_  = p.sizes().vec();
      TORCH_CHECK(p.scalar_type() == torch::kFloat32 || p.scalar_type() == torch::kFloat64,
                  "policy must be float");
      TORCH_CHECK(o.scalar_type() == torch::kFloat32 || o.scalar_type() == torch::kFloat64,
                  "observation must be float");
    } else {
      TORCH_CHECK(o.sizes().vec() == obs_shape_, "obs shape mismatch");
      TORCH_CHECK(p.sizes().vec() == pi_shape_,  "pi shape mismatch");
    }

    if (obs_.size() == capacity_) {
      obs_.pop_front(); pi_.pop_front(); values_.pop_front();
    }
    obs_.push_back(o);
    pi_.push_back(p);
    values_.push_back(value);
  }

  ReplaySample sample(size_t batch_size, c10::optional<torch::Device> device = c10::nullopt) {
    if (empty()) throw std::runtime_error("ReplayBuffer.sample: buffer is empty");
    const size_t N = size();
    batch_size = std::min(batch_size, N);
    if (batch_size == 0) throw std::runtime_error("ReplayBuffer.sample: batch_size==0 after clamp");

    // indices 0..N-1, shuffle, take first B
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng_);
    idx.resize(batch_size);

    std::vector<torch::Tensor> obs_batch;   obs_batch.reserve(batch_size);
    std::vector<torch::Tensor> pi_batch;    pi_batch.reserve(batch_size);
    std::vector<float>         v_batch;     v_batch.reserve(batch_size);

    for (size_t k : idx) {
      obs_batch.push_back(obs_[k]);
      pi_batch.push_back(pi_[k]);
      v_batch.push_back(values_[k]);
    }

    auto obsT = torch::stack(obs_batch, 0);
    auto piT  = torch::stack(pi_batch, 0);
    auto valT = torch::from_blob(v_batch.data(), {(long)batch_size}, torch::kFloat32).clone();

    if (device.has_value()) {
      obsT = obsT.to(*device);
      piT  = piT.to(*device);
      valT = valT.to(*device);
    }

    return ReplaySample{obsT, piT, valT};
  }

  void save(torch::serialize::OutputArchive& archive) const {
    archive.write("capacity", torch::tensor(static_cast<int64_t>(capacity_), torch::kInt64));
    archive.write("size", torch::tensor(static_cast<int64_t>(obs_.size()), torch::kInt64));
    if (obs_.empty()) return;

    auto stack_tensors = [](const std::deque<torch::Tensor>& source) {
      std::vector<torch::Tensor> tmp;
      tmp.reserve(source.size());
      for (const auto& t : source) {
        tmp.push_back(t.cpu());
      }
      return torch::stack(tmp, 0);
    };

    auto obs_stack = stack_tensors(obs_);
    auto pi_stack  = stack_tensors(pi_);
    archive.write("obs", obs_stack);
    archive.write("pi",  pi_stack);

    std::vector<float> v(values_.begin(), values_.end());
    auto val_tensor = torch::from_blob(v.data(), {(long)v.size()}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    archive.write("values", val_tensor);
  }

  void load(torch::serialize::InputArchive& archive) {
    obs_.clear();
    pi_.clear();
    values_.clear();
    obs_shape_.clear();
    pi_shape_.clear();

    torch::Tensor cap_tensor;
    archive.read("capacity", cap_tensor);
    const auto stored_capacity = std::max<int64_t>(1, cap_tensor.item<int64_t>());
    capacity_ = static_cast<size_t>(stored_capacity);

    torch::Tensor size_tensor;
    archive.read("size", size_tensor);
    const int64_t size = size_tensor.item<int64_t>();
    if (size <= 0) return;

    torch::Tensor obs_tensor;
    torch::Tensor pi_tensor;
    torch::Tensor val_tensor;
    archive.read("obs", obs_tensor);
    archive.read("pi",  pi_tensor);
    archive.read("values", val_tensor);

    const auto obs_sizes = obs_tensor.sizes();
    obs_shape_.assign(obs_sizes.begin() + 1, obs_sizes.end());
    const auto pi_sizes = pi_tensor.sizes();
    pi_shape_.assign(pi_sizes.begin() + 1, pi_sizes.end());

    const int64_t entries = std::min<int64_t>(size, obs_tensor.size(0));
    for (int64_t i = 0; i < entries; ++i) {
      obs_.push_back(obs_tensor[i].clone());
      pi_.push_back(pi_tensor[i].clone());
      values_.push_back(val_tensor[i].item<float>());
    }
  }

private:
  size_t capacity_;
  std::deque<torch::Tensor> obs_;
  std::deque<torch::Tensor> pi_;
  std::deque<float>         values_;

  std::vector<int64_t> obs_shape_;
  std::vector<int64_t> pi_shape_;

  std::mt19937 rng_;
};

} // namespace alphazero
