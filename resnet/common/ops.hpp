#pragma once

#include "common.h"
#include "tensor.hpp"
namespace Impl {
class TensorOps
{
public:
    static void relu_(Tensor &x)
    {
        // TODO
    }

    static void add_(Tensor &x, const Tensor &y)
    {
        // TODO
    }

    static Tensor argmax(const Tensor &x, int dim);

    static float sum_equal(const Tensor &lhs, const Tensor &rhs);
};
}
