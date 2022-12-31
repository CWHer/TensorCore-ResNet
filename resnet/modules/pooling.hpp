#pragma once

#include "common.h"
#include "module.hpp"

#include "functional/pooling.h"

class MaxPool2d : public Module
{
private:
    int kernel_size, stride, padding;

public:
    MaxPool2d(int kernel_size, int stride = 1, int padding = 0)
        : kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    Tensor forward(Tensor x) override
    {
        timer.start("forward");

        checkCppErrorsMsg(x.sizes().size() != 4, "MaxPool2d only support 4D tensor");
        checkCppErrorsMsg(x.getDevice() != DeviceType::CUDA, "MaxPool2d only support CUDA");

        // NOTE: x is in NCHW format
        int batch_size = x.sizes()[0];
        int num_channels = x.sizes()[1];
        int height = x.sizes()[2];
        int width = x.sizes()[3];

        int output_height = (height + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
        int output_width = (width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;

        float *input_data = x.data_ptr();

        Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);
        float *output_data = output.data_ptr();

        hostMaxPool2d(batch_size, num_channels,
                      input_data, height, width,
                      output_data, output_height, output_width,
                      kernel_size, padding, stride);

        timer.end("forward");
        return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":MaxPool2d" << std::endl;
    }

    void printStat(const std::string &prefix) override
    {
        printModule(prefix);
        timer.printStat("forward");
    }
};

class AvgPool2d : public Module
{
private:
    int kernel_size, stride, padding;

public:
    AvgPool2d(int kernel_size, int stride = 1, int padding = 0)
        : kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    Tensor forward(Tensor x) override
    {
        timer.start("forward");

        checkCppErrorsMsg(x.sizes().size() != 4, "AvgPool2d only support 4D tensor");
        checkCppErrorsMsg(x.getDevice() != DeviceType::CUDA, "AvgPool2d only support CUDA");
        // FIXME: (or not) this is a shape (N, 128K, H, W) oriented implementation,
        //  we DO NOT guarantee the correctness of inputs with other shapes
        checkCppErrorsMsg(x.sizes()[1] % 128 != 0,
                          "AvgPool2d is shape oriented, only support N x 128K x H x W");

        // NOTE: x is in NCHW format
        int batch_size = x.sizes()[0];
        int num_channels = x.sizes()[1];
        int height = x.sizes()[2];
        int width = x.sizes()[3];

        int output_height = (height + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
        int output_width = (width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;

        float *input_data = x.data_ptr();

        Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);
        float *output_data = output.data_ptr();

        hostAvgPool2d(batch_size, num_channels,
                      input_data, height, width,
                      output_data, output_height, output_width,
                      kernel_size, padding, stride);

        timer.end("forward");
        return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":AvgPool2d" << std::endl;
    }

    void printStat(const std::string &prefix) override
    {
        printModule(prefix);
        timer.printStat("forward");
    }
};
