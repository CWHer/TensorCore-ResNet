#pragma once

#include "common.h"
#include "tensor.hpp"
#include "functional/add.hpp"

namespace Impl {
class TensorOps {
public:
  static void relu_(Tensor &x);

  static void add_(Tensor &x, const Tensor &y);

  static Tensor argmax(const Tensor &x, int dim);

  static float sum_equal(const Tensor &lhs, const Tensor &rhs);
};
}
