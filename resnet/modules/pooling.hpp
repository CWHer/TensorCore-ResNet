#pragma once

#include "common.h"
#include "module.hpp"

#include "pooling.cuh"

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
        checkCppErrorsMsg(x.sizes().size() != 4, "MaxPool2d only support 4D tensor");
        checkCppErrorsMsg(x.getDevice() != DeviceType::CUDA, "MaxPool2d only support CUDA");
        // FIXME: (or not) this is a shape (2N, 64, H, W) oriented implementation,
        //  we DO NOT guarantee the correctness of inputs with other shapes
        checkCppErrorsMsg(x.sizes()[0] % 2 != 0 || x.sizes()[1] != 64,
                          "MaxPool2d is shape oriented, only support 2N x 64 x H x W");

        // NOTE: x is in NCHW format
        int batch_size = x.sizes()[0];
        int num_channels = x.sizes()[1];
        int height = x.sizes()[2];
        int width = x.sizes()[3];

        int output_height = (height + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
        int output_width = (width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;

        float *input_data = x.data_ptr();

        // FIXME: can we find a way to pre-allocate and reuse the memory,
        //  instead of allocating them every time (though this should be the way)
        Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);
        float *output_data = output.data_ptr();

        static const int N_THREADS = 128;
        dim3 block_dim(N_THREADS);
        dim3 grid_dim(batch_size * num_channels / N_THREADS);
        deviceMaxPool2dKernel<32><<<grid_dim, block_dim>>>(
            input_data, height, width, output_data, output_height, output_width,
            kernel_size, padding, stride);

        return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":MaxPool2d" << std::endl;
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

        // FIXME: can we find a way to pre-allocate and reuse the memory,
        //  instead of allocating them every time (though this should be the way)
        Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);
        float *output_data = output.data_ptr();

        static const int N_THREADS = 128;
        dim3 block_dim(N_THREADS);
        dim3 grid_dim(batch_size * num_channels / N_THREADS);
        deviceAvgPool2dKernel<32><<<grid_dim, block_dim>>>(
            input_data, height, width, output_data, output_height, output_width,
            kernel_size, padding, stride);

        return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":AvgPool2d" << std::endl;
    }
};
