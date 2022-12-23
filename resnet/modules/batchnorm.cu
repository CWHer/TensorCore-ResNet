#pragma once

#include "common.h"
#include "module.hpp"

//#include "batchnorm.cu"

// update the data
__global__ void BatchNorm2dKernel_update(float *input_data, unsigned int ele_per_blocks, float mean, float var, float weight, float bias, float eps)
{
    // update input data
    unsigned int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idx = global_idx * ele_per_blocks;
    float denominator = sqrt(var + eps);
    for (size_t i = 0; i < ele_per_blocks; i++)
    {
        input_data[idx + i] = ((input_data[idx + i] - mean) / denominator) * weight + bias;
    }
}

static inline void get_single_channel_data(float *input_data, float *single_channel_device, unsigned int batch_sizes, unsigned int channel_sizes, unsigned int height, unsigned int width, unsigned int channel_idx)
{
    unsigned int figure_sizes = height * width;
    for (int batch = 0; batch < batch_sizes; batch++)
    {
        unsigned int in_idx_bias = batch * channel_sizes * figure_sizes + channel_idx * figure_sizes;
        unsigned int out_idx_bias = batch * figure_sizes;
        checkCudaErrors(cudaMemcpy(single_channel_device + out_idx_bias, input_data + in_idx_bias, figure_sizes * sizeof(float), cudaMemcpyHostToDevice));
    }
}

static inline void restore_single_channel_data(float *input_data, float *single_channel_device, unsigned int batch_sizes, unsigned int channel_sizes, unsigned int height, unsigned int width, unsigned int channel_idx)
{
    unsigned int figure_sizes = height * width;
    for (int batch = 0; batch < batch_sizes; batch++)
    {
        unsigned int in_idx_bias = batch * channel_sizes * figure_sizes + channel_idx * figure_sizes;
        unsigned int out_idx_bias = batch * figure_sizes;
        checkCudaErrors(cudaMemcpy(input_data + in_idx_bias, single_channel_device + out_idx_bias, figure_sizes * sizeof(float), cudaMemcpyDeviceToHost));
    }
}


void BatchNorm2dKernel(float *input_data, float *mean_data, float *var_data, float *weight_data, float *bias_data, const float eps,
                     const unsigned int batch_size, const unsigned int num_channels, const unsigned int height, const unsigned int width)
{
    // 1. allocate memory and copy data to device
    unsigned int channel_data_sizes = batch_size * height * width;
    float *single_channel_data;
    checkCudaErrors(cudaMalloc(&single_channel_data, channel_data_sizes * sizeof(float)));
    // 2. do the computation
    for (int i = 0; i < num_channels; i++)
    {
        // construct the array
        get_single_channel_data(input_data, single_channel_data, batch_size, num_channels, height, width, i);

        // update data
        dim3 dim_grid(batch_size);
        dim3 dim_block(height);
        BatchNorm2dKernel_update<<<dim_grid, dim_block>>>(single_channel_data, width, mean_data[i], var_data[i], weight_data[i], bias_data[i], eps);
        // copy the result to output
        restore_single_channel_data(input_data, single_channel_data, batch_size, num_channels, height, width, i);
    }
    checkCudaErrors(cudaFree(single_channel_data));
}

class BatchNorm2d : public Module
{
private:
    float eps;
    int num_features;
    float *mean_data, *var_data, *weight_data, *bias_data;

public:
    BatchNorm2d(Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, float eps = 1e-5) : eps(eps)
    {
        addTensor("weight", weight);             // [num_features]
        addTensor("bias", bias);                 // [num_features]
        addTensor("running_mean", running_mean); // [num_features]
        addTensor("running_var", running_var);   // [num_features]
        weight_data = weight.data_ptr();
        bias_data = bias.data_ptr();
        mean_data = running_mean.data_ptr();
        var_data = running_var.data_ptr();
        num_features = weight.sizes()[0];
    }

    Tensor forward(Tensor x) override
    {
        checkCppErrorsMsg(x.sizes().size() != 4, "BatchNorm2d only support 4D input");
        checkCppErrorsMsg(x.sizes()[1] != num_features, "BatchNorm2d input channel size mismatch");
        checkCppErrorsMsg(x.getDevice() != DeviceType::CPU,
                          "BatchNorm2d only support CUDA");

        // NOTE: x is in NCHW format
        // NOTE: computation is done channel-wise
        unsigned int batch_size = x.sizes()[0];
        unsigned int num_channels = x.sizes()[1];
        unsigned int height = x.sizes()[2];
        unsigned int width = x.sizes()[3];

        float *input_data = x.data_ptr();
        
        BatchNorm2dKernel(input_data, mean_data, var_data, weight_data, bias_data, eps,
                          batch_size, num_channels, height, width);
        
        return x;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":BatchNorm2d" << std::endl;
    }
};