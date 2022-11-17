#pragma once

#include "common.h"
#include "tensor.hpp"

class Module
{
    // TODO

public:
    virtual void loadWeights(const std::string &path) = 0;
    virtual Tensor forward(Tensor x) = 0;
    virtual void to(DeviceType device) = 0;
};