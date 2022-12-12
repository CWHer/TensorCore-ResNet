#pragma once

#include "common.h"
#include "tensor.hpp"

struct TensorOps
{
    static void relu_(Tensor &x)
    {
        // TODO
    }

    static void add_(Tensor &x, const Tensor &y)
    {
        // TODO
    }

    static void add_(Tensor &x, float y)
    {
        checkCppErrorsMsg(x.device != DeviceType::CPU,
                          "add_() is not implemented for CUDA");
        for (int i = 0; i < x.total_size; i++)
            *(x.data + i) += y;
    }

    static void mul_(Tensor &x, float y)
    {
        checkCppErrorsMsg(x.device != DeviceType::CPU,
                          "mul_() is not implemented for CUDA");
        for (int i = 0; i < x.total_size; i++)
            *(x.data + i) *= y;
    }
};