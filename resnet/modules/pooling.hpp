#pragma once

#include "common.h"
#include "module.hpp"

namespace Impl {
class MaxPool2d : public Module
{
private:
    int kernel_size, stride, padding;

public:
    MaxPool2d(int kernel_size, int stride = 1, int padding = 0);

    Tensor forward(const Tensor &x) override;
    Tensor forward(Tensor &&x) override;

    void printModule(const std::string &prefix) override;
};

class AvgPool2d : public Module
{
private:
    int kernel_size, stride, padding;

public:
    AvgPool2d(int kernel_size, int stride = 1, int padding = 0);

    Tensor forward(const Tensor &x) override;
    Tensor forward(Tensor &&x) override;

    void printModule(const std::string &prefix) override;
};

}
