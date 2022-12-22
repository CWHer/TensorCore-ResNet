#pragma once

#include "common.h"
#include "module.hpp"

__global__ void MaxPool2dKernel(const int in_batch, const int in_channel, const int nthreads,
                                const float* in_data, const int in_height, const int in_width,
                                float* out_data, const int out_height, const int out_width,
                                const int kernel_size_h, const int kernel_size_w,
                                const int padding_h, const int padding_w, 
                                const int stride_h, const int stride_w)
{
    int global_tid = blockIdx.x * blockDim.x * blockDim.y
                     + threadIdx.y * blockDim.x + threadIdx.x;
    if (global_tid < nthreads)
    {
        // output data index
        const int pooling_width = global_tid % out_width;
        const int pooling_height = (global_tid / out_width) % out_height;
        const int channel = (global_tid / out_width / out_height) % in_channel;
        const int batch = global_tid / out_width / out_height / in_channel;
        
        // input data index
        const int start_h = max(pooling_height * stride_h - padding_h, 0);
        const int start_w = max(pooling_width * stride_w - padding_w, 0);
        const int end_h = min(start_h + kernel_size_h, in_height + padding_h);
        const int end_w = min(start_w + kernel_size_w, in_width + padding_w);
        //printf("gltid=%d, batch=%d, channel=%d, h=%d, w=%d\t\t\tsh=%d, sw=%d, eh=%d, ew=%d\n", global_tid, batch, channel, pooling_height, pooling_width, start_h, start_w, end_h, end_w);
        // 1d addr starta(No.figure)
        const float* in_data_start_idx = in_data + (batch * in_channel + channel) * in_height * in_width;
        // 2d computation
        float max_val = -FLT_MAX;
        for (int h = start_h; h < end_h; h++)
        {
            for (int w = start_w; w < end_w; w++)
            {
                int cur_idx = h * in_width + w;
                max_val = in_data_start_idx[cur_idx] > max_val ? in_data_start_idx[cur_idx] : max_val;
            }
        }
        out_data[global_tid] = max_val;
    }
}


__global__ void AdaptiveAvgPool2dKernel(const int in_batch, const int in_channel, const int nthreads,
                                        const float* in_data, const int in_height, const int in_width,
                                        float* out_data, const int out_height, const int out_width,
                                        const int kernel_size_h, const int kernel_size_w,
                                        const int padding_h, const int padding_w, 
                                        const int stride_h, const int stride_w)
{
    int global_tid = blockIdx.x * blockDim.x * blockDim.y
                     + threadIdx.y * blockDim.x + threadIdx.x;
    if (global_tid < nthreads)
    {
        // output data index
        const int pooling_width = global_tid % out_width;
        const int pooling_height = (global_tid / out_width) % out_height;
        const int channel = (global_tid / out_width / out_height) % in_channel;
        const int batch = global_tid / out_width / out_height / in_channel;
        
        // input data index
        const int start_h = max(pooling_height * stride_h - padding_h, 0);
        const int start_w = max(pooling_width * stride_w - padding_w, 0);
        const int end_h = min(start_h + kernel_size_h, in_height + padding_h);
        const int end_w = min(start_w + kernel_size_w, in_width + padding_w);
        // 1d addr starta(No.figure)
        const float* in_data_start_idx = in_data + (batch * in_channel + channel) * in_height * in_width;
        // 2d computation
        float sum = 0.0;
        for (int h = start_h; h < end_h; h++)
        {
            for (int w = start_w; w < end_w; w++)
            {
                sum += in_data_start_idx[h * in_width + w];
            }
        }
        out_data[global_tid] = sum / ((end_h - start_h) * (end_w - start_w));
    }
}

class MaxPool2d : public Module
{
private:
    int kernel_size, stride, padding, dilation, ceil_mode;

public:
    MaxPool2d(int kernel_size = 3, int stride = 2, int padding = 0,
              int dilation = 1, bool ceil_mode = false)
        : kernel_size(kernel_size), stride(stride), padding(padding),
          dilation(dilation), ceil_mode(ceil_mode)
    {}

    Tensor forward(Tensor x) override
    {
        int in_batch = x.sizes()[0];
        int in_channel = x.sizes()[1];
        int in_height = x.sizes()[2];
        int in_width = x.sizes()[3];
        int total_sizes = in_batch * in_channel * in_height * in_width;
        int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        float* intput_data = x.data_ptr();

        // gpu execution
        float* in_data_device;
        checkCudaErrors(cudaMalloc(&in_data_device, total_sizes * sizeof(float)));
        checkCudaErrors(cudaMemcpy(in_data_device, intput_data, total_sizes * sizeof(float), cudaMemcpyHostToDevice));
        int nthreads = in_batch * in_channel * out_height * out_width;
        float* out_data_device;
        checkCudaErrors(cudaMalloc(&out_data_device, nthreads * sizeof(float)));
        dim3 dim_grid(int(ceil(float(nthreads) / 400.0)));
        dim3 dim_block(20, 20);
        // do the computation, each thread compute one figures
        MaxPool2dKernel<<<dim_grid, dim_block>>>(in_batch, in_channel, nthreads,
                                                 in_data_device, in_height, in_width,
                                                 out_data_device, out_height, out_width,
                                                 kernel_size, kernel_size,
                                                 padding, padding,
                                                 stride, stride);
        // copy the result to output
        float* output_data = new float[nthreads];
        checkCudaErrors(cudaMemcpy(output_data, out_data_device, nthreads * sizeof(float), cudaMemcpyDeviceToHost));
        checkCppErrors(cudaFree(in_data_device));
        checkCppErrors(cudaFree(out_data_device));
        return Tensor({in_batch, in_channel, out_height, out_width}, DeviceType::CPU, output_data);
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":MaxPool2d" << std::endl;
    }
};

class AdaptiveAvgPool2d : public Module
{
private:
    int out_height;
    int out_width;
    Tensor output;

public:
    AdaptiveAvgPool2d(int output_height, int output_width) : out_height(output_height), out_width(output_width) {}

    Tensor forward(Tensor x) override
    {
        int in_batch = x.sizes()[0];
        int in_channel = x.sizes()[1];
        int in_height = x.sizes()[2];
        int in_width = x.sizes()[3];
        int total_sizes = in_batch * in_channel * in_height * in_width;
        // ref1:https://discuss.pytorch.org/t/what-is-adaptiveavgpool2d/26897
        // ref2:https://cloud.tencent.com/developer/article/1451499
        int stride_h = in_height / out_height;
        int stride_w = in_width / out_width;
        int kernel_size_h = in_height - (out_height - 1) * stride_h;
        int kernel_size_w = in_width - (out_width - 1) * stride_w;
        int padding_h = 0;
        int padding_w = 0;
        float* intput_data = x.data_ptr();
        
        // gpu execuetion
        float* in_data_device;
        checkCudaErrors(cudaMalloc(&in_data_device, total_sizes * sizeof(float)));
        checkCudaErrors(cudaMemcpy(in_data_device, intput_data, total_sizes * sizeof(float), cudaMemcpyHostToDevice));
        int nthreads = in_batch * in_channel * out_height * out_width;
        float* out_data_device;
        checkCudaErrors(cudaMalloc(&out_data_device, nthreads * sizeof(float)));
        dim3 dim_grid(int(ceil(float(nthreads) / 400.0)));
        dim3 dim_block(20, 20);
        // do the computation, each thread compute one figures
        AdaptiveAvgPool2dKernel<<<dim_grid, dim_block>>>(in_batch, in_channel, nthreads,
                                                         in_data_device, in_height, in_width,
                                                         out_data_device, out_height, out_width,
                                                         kernel_size_h, kernel_size_w,
                                                         padding_h, padding_w,
                                                         stride_h, stride_w);
        // copy the result to output
        float* output_data = new float[nthreads];
        checkCudaErrors(cudaMemcpy(output_data, out_data_device, nthreads * sizeof(float), cudaMemcpyDeviceToHost));
        checkCppErrors(cudaFree(in_data_device));
        checkCppErrors(cudaFree(out_data_device));
        return Tensor({in_channel, out_height, out_width}, DeviceType::CPU, output_data);
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":AdaptiveAvgPool2d" << std::endl;
    }
};