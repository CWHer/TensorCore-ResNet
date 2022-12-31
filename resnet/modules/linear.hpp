#pragma once

#include "common.h"
#include "module.hpp"

#include "functional/gemm.h"
#include "functional/linear.h"

class Linear : public Module
{
private:
    Tensor weight, bias;
    f16 *weight_f16;

public:
    Linear(int in_features, int out_features) : weight_f16(nullptr)
    {
        // @param weight Weight tensor of shape (input_channel, output_channel)
        // @param bias Bias tensor of shape (output_channel)
        addTensor("weight", weight);
        addTensor("bias", bias);
    }

    ~Linear()
    {
        if (weight_f16 != nullptr)
            checkCudaErrors(cudaCacheFree(weight_f16));
    }

    Tensor forward(Tensor x) override
    {
        timer.start("forward");
        checkCppErrorsMsg(x.sizes().size() != 2, "Linear: input must be 2D");
        checkCppErrorsMsg(x.sizes()[1] != weight.sizes()[1], "Linear: input size must match weight size");

        if (weight_f16 == nullptr)
        {
            checkCudaErrors(cudaCacheMalloc((void **)&weight_f16, weight.totalSize() * sizeof(f16)));
            makeLinearWeight(weight.data_ptr(), weight_f16, weight.sizes()[0], weight.sizes()[1]);
        }

        int batch_size = x.sizes()[0];
        int input_channel = x.sizes()[1];
        int output_channel = bias.sizes()[0];

        f16 *input_f16 = float2half(x.data_ptr(), x.totalSize());
        auto output = Tensor({x.sizes()[0], weight.sizes()[0]}, DeviceType::CUDA);

        linear(input_f16, output.data_ptr(),
               weight_f16, bias.data_ptr(),
               batch_size, input_channel, output_channel);

        checkCudaErrors(cudaCacheFree(input_f16));

        timer.end("forward");
        return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":Linear" << std::endl;
    }

    void printStat(const std::string &prefix) override
    {
        printModule(prefix);
        timer.printStat("forward");
    }
};