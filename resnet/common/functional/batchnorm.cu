#pragma once

#include "common.h"

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

