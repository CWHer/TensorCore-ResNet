#pragma once

#include "common.h"
#include "module.hpp"

#include "functional/pool.hpp"

namespace Impl {
class MaxPool2d : public Module
{
private:
    int kernel_size, stride, padding;

public:
    MaxPool2d(int kernel_size, int stride = 1, int padding = 0)
        : kernel_size(kernel_size), stride(stride), padding(padding)
    {
    }

    Tensor forward(const Tensor &x) override
    {
        checkCppErrorsMsg(x.sizes().size() != 4, "MaxPool2d only support 4D tensor");
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


        // FIXME: can we find a way to pre-allocate and reuse the memory,
        //  instead of allocating them every time (though this should be the way)
        Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);
        if (x.getDevice() == DeviceType::CUDA) {
          maxpool2d(x.data_ptr(),
                    batch_size,
                    num_channels,
                    height,
                    width,
                    output.data_ptr(),
                    output_height,
                    output_width,
                    kernel_size,
                    padding,
                    stride);
        } else {
          // If not cuda, move to cuda
          Tensor x_cuda = x;
          x_cuda.to(DeviceType::CUDA);
          maxpool2d(x_cuda.data_ptr(),
                    batch_size,
                    num_channels,
                    height,
                    width,
                    output.data_ptr(),
                    output_height,
                    output_width,
                    kernel_size,
                    padding,
                    stride);
          output.to(DeviceType::CPU);
        }

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

    Tensor forward(const Tensor &x) override
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


        // FIXME: can we find a way to pre-allocate and reuse the memory,
        //  instead of allocating them every time (though this should be the way)
        Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);

        avgpool2d(x.data_ptr(), batch_size, num_channels, height, width,
                  output.data_ptr(), output_height, output_width, kernel_size, padding, stride);

        return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":AvgPool2d" << std::endl;
    }
};

}
