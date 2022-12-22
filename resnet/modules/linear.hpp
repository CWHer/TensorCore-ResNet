#pragma once

#include "common.h"
#include "module.hpp"

namespace Impl {

class Linear : public Module
{
    // TODO
private:
    Tensor weight, bias;

public:
    Linear(int in_features, int out_features)
        : weight({out_features, in_features}), bias({out_features})
    {
        // TODO
        addTensor("weight", weight);
        addTensor("bias", bias);
    }

    Tensor forward(Tensor x) override
    {
        // TODO
        return Tensor({x.sizes()[0], weight.sizes()[0]});
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":Linear" << std::endl;
    }
};
}
