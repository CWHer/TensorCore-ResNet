#pragma once

#include "common.h"
#include "module.hpp"

class Conv2d : public Module
{
    // TODO
private:
    // FIXME: dummy variables
    int in_channels, out_channels, kernel_size, stride, padding, dilation, groups;

public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
           int padding = 0, int dilation = 1, int groups = 1, bool bias = true)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
          stride(stride), padding(padding), dilation(dilation), groups(groups)
    {
        // TODO
    }

    Tensor forward(Tensor x)
    {
        // TODO
        int out_height = (x.sizes()[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int out_width = (x.sizes()[3] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        return Tensor({x.sizes()[0], out_channels, out_height, out_width});
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << "Conv2d" << std::endl;
    }
};
