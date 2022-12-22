#pragma once

#include "common.h"
#include "module.hpp"
#include "functional/conv2d.hpp"

namespace Impl {

class Conv2d : public Module {
  // TODO
private:
  Tensor weight, bias;
  float_16 *weight_f16;
  int in_channels, out_channels, kernel_size, stride, padding, dilation, groups;

public:
  Conv2d(int in_channels,
         int out_channels,
         int kernel_size,
         int stride = 1,
         int padding = 0,
         int dilation = 1,
         int groups = 1,
         bool bias = true);

  void setWeight(const Tensor &new_weight);
  void setBias(const Tensor &new_bias);

  Tensor forward(Tensor x) override;

  void printModule(const std::string &prefix) override;
};

}

namespace functional {
using namespace Impl;

Tensor conv2d(const Tensor &input,
              const Tensor &weight,
              const Tensor &bias = Tensor(),
              int stride = 1,
              int padding = 0,
              int dilation = 1,
              int groups = 1);

Tensor conv2d(const Tensor &input,
              const float_16 *weight,
              const Tensor &bias,
              int out_channels,
              int kernel_size,
              int stride,
              int padding,
              int dilation,
              int groups);
}
