#pragma once

#include "common.h"
#include "module.hpp"

class BatchNorm2dImpl : public Module
{
    // TODO
private:
public:
    BatchNorm2dImpl(int num_features, double eps = 1e-5, double momentum = 0.1,
                    bool affine = true, bool track_running_stats = true)
    {
        // TODO
    }

    Tensor forward(Tensor x) override
    {
        // TODO
        return Tensor(x.sizes());
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":BatchNorm2d" << std::endl;
    }
};

NETWORK_MODULE(BatchNorm2d);