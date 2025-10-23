#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

namespace alphazero {

class ActionHelper{
 public:
  explicit ActionHelper(){

      // compute all feasable actions

  }

 private:
  std::vector<std::vector<int>> actions;

};

}  // namespace alphazero
