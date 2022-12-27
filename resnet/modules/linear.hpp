#pragma once

#include "common.h"
#include "module.hpp"
#include "functional/linear.hpp"

namespace Impl {

class Linear : public Module {
private:
  Tensor weight, bias;
  float_16 *weight_f16;
  int in_features, out_features;

public:
  Linear(int in_features, int out_features);
  ~Linear();
  Tensor forward(const Tensor &x) override;
  Tensor forward(Tensor &&x) override;
  void printModule(const std::string &prefix) override;

  void setWeight(const Tensor &new_weight);
  void setBias(const Tensor &new_bias);

};
}

namespace functional {
using namespace Impl;

Tensor linear(const Tensor &input, const Tensor &weight, const Tensor &bias);
Tensor linear(const Tensor &input, const float_16 *weight, const Tensor &bias);
}
