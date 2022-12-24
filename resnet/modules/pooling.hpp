#pragma once

#include "common.h"
#include "module.hpp"

namespace Impl {
class MaxPool2d : public Module
{
    // TODO
private:
    // FIXME: dummy variables
    int kernel_size, stride, padding, dilation, ceil_mode;

public:
    MaxPool2d(int kernel_size, int stride = 1, int padding = 0,
              int dilation = 1, bool ceil_mode = false)
        : kernel_size(kernel_size), stride(stride), padding(padding),
          dilation(dilation), ceil_mode(ceil_mode)
    {
        // TODO
    }

    Tensor forward(const Tensor &x) override
    {
        // TODO
        int out_height = (x.sizes()[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int out_width = (x.sizes()[3] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        return Tensor({x.sizes()[0], x.sizes()[1], out_height, out_width});
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":MaxPool2d" << std::endl;
    }
};

class AdaptiveAvgPool2d : public Module
{
    // TODO
private:
    // FIXME: dummy variables
    int output_size;

public:
    AdaptiveAvgPool2d(int output_size) : output_size(output_size)
    {
        // TODO
    }

    Tensor forward(const Tensor &x) override
    {
        // TODO
        return Tensor({x.sizes()[0], x.sizes()[1], output_size, output_size});
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":AdaptiveAvgPool2d" << std::endl;
    }
};
}
