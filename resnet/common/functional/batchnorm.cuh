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

void hostBatchNorm2d(float *input_data, float *mean_data, float *var_data,
                     float *weight_data, float *bias_data,
                     float eps, unsigned int batch_size, unsigned int num_channels,
                     unsigned int height, unsigned int width)
{
    // do the computation
    for (int i = 0; i < num_channels; i++)
    {
        // update data
        dim3 dim_grid(batch_size);
        dim3 dim_block(height);
        BatchNorm2dKernel_update<<<dim_grid, dim_block>>>(single_channel_data, width, mean_data[i], var_data[i], weight_data[i], bias_data[i], eps);
        // copy the result to output
        restore_single_channel_data(input_data, single_channel_data, batch_size, num_channels, height, width, i);
    }
    checkCudaErrors(cudaFree(single_channel_data));
}
