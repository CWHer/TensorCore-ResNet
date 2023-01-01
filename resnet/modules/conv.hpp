#pragma once

#include "common.h"
#include "module.hpp"

#include "functional/gemm.h"
#include "functional/conv.h"

class Conv2d : public Module
{
private:
    int in_channels, out_channels, kernel_size;
    int stride, padding, dilation, groups;
    Tensor weight;
    f16 *weight_f16;

public:
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1,
           int padding = 0, int dilation = 1, int groups = 1, bool bias = false)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size),
          stride(stride), padding(padding), dilation(dilation), groups(groups), weight_f16(nullptr)
    {
        checkCppErrorsMsg(dilation != 1 || groups != 1,
                          "Conv2d: dilation and groups not supported yet");
        checkCppErrorsMsg(bias, "Conv2d: bias not supported yet");

        addTensor("weight", weight);
    }

    ~Conv2d()
    {
        if (weight_f16 != nullptr)
            checkCudaErrors(cudaCacheFree(weight_f16));
    }

    Tensor forward(Tensor x) override
    {
        timer.start("forward");

        checkCppErrorsMsg(x.sizes().size() != 4, "Conv2d: input must be 4D");

        if (weight_f16 == nullptr)
            weight_f16 = float2half(weight.data_ptr(), weight.totalSize());

        int batch_size = x.sizes()[0];
        int input_channel = x.sizes()[1];
        int input_height = x.sizes()[2];
        int input_width = x.sizes()[3];

        int out_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int out_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        auto output = Tensor({batch_size, out_channels, out_height, out_width}, DeviceType::CUDA);

        conv2d(x.data_ptr(), output.data_ptr(), weight_f16,
               batch_size, input_channel, input_height, input_width,
               out_channels, out_height, out_width, kernel_size, stride, padding);

        timer.end("forward");
        return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":Conv2d" << std::endl;
    }

    void printStat(const std::string &prefix) override
    {
        printModule(prefix);
        timer.printStat("forward");
    }
};
