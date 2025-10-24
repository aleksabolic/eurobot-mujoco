#pragma once
#include <deque>
#include <random>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <torch/torch.h>

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

    // TODO: remove these checks
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
