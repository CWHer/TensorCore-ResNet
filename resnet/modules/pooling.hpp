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
        // NOTE: too large H or W will cause the kernel to fail due to shared memory limitation
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

        dim3 block_dim(128);
        dim3 grid_dim(batch_size / 2);
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

// class AvgPool2d : public Module
// {
// private:
//     int kernel_size, stride, padding, dilation;

// public:
//     AvgPool2d(int kernel_size, int stride = 1,
//               int padding = 0, int dilation = 1)
//         : kernel_size(kernel_size), stride(stride),
//           padding(padding), dilation(dilation)
//     {
//     }

//     Tensor forward(Tensor x) override
//     {
//         checkCppErrorsMsg(x.getDevice() ! = DeviceType::CUDA,
//                           "AvgPool2d only support CUDA");

//         int in_batch = x.sizes()[0];
//         int in_channel = x.sizes()[1];
//         int in_height = x.sizes()[2];
//         int in_width = x.sizes()[3];
//         int total_sizes = in_batch * in_channel * in_height * in_width;
//         // ref1:https://discuss.pytorch.org/t/what-is-adaptiveavgpool2d/26897
//         // ref2:https://cloud.tencent.com/developer/article/1451499
//         int stride_h = in_height / out_height;
//         int stride_w = in_width / out_width;
//         int kernel_size_h = in_height - (out_height - 1) * stride_h;
//         int kernel_size_w = in_width - (out_width - 1) * stride_w;
//         int padding_h = 0;
//         int padding_w = 0;
//         float *intput_data = x.data_ptr();

//         // gpu execuetion
//         float *in_data_device;
//         checkCudaErrors(cudaMalloc(&in_data_device, total_sizes * sizeof(float)));
//         checkCudaErrors(cudaMemcpy(in_data_device, intput_data, total_sizes * sizeof(float), cudaMemcpyHostToDevice));
//         int nthreads = in_batch * in_channel * out_height * out_width;
//         float *out_data_device;
//         checkCudaErrors(cudaMalloc(&out_data_device, nthreads * sizeof(float)));
//         dim3 dim_grid(int(ceil(float(nthreads) / 400.0)));
//         dim3 dim_block(20, 20);
//         // do the computation, each thread compute one figures
//         AdaptiveAvgPool2dKernel<<<dim_grid, dim_block>>>(in_batch, in_channel, nthreads,
//                                                          in_data_device, in_height, in_width,
//                                                          out_data_device, out_height, out_width,
//                                                          kernel_size_h, kernel_size_w,
//                                                          padding_h, padding_w,
//                                                          stride_h, stride_w);
//         // copy the result to output
//         float *output_data = new float[nthreads];
//         checkCudaErrors(cudaMemcpy(output_data, out_data_device, nthreads * sizeof(float), cudaMemcpyDeviceToHost));
//         checkCppErrors(cudaFree(in_data_device));
//         checkCppErrors(cudaFree(out_data_device));
//         return Tensor({in_channel, out_height, out_width}, DeviceType::CPU, output_data);
//     }

//     void printModule(const std::string &prefix) override
//     {
//         std::cout << prefix << ":AvgPool2d" << std::endl;
//     }
// };
