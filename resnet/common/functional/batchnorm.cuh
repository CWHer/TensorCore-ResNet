#pragma once

#include "common.h"

// update the data
__global__ void BatchNorm2dKernelUpdateByHW(float *input_data, float *mean_data, float *var_data,
                                            float *weight_data, float *bias_data,
                                            float eps, unsigned int batch_size, unsigned int num_channels,
                                            unsigned int height, unsigned int width,
                                            unsigned int total_ele, unsigned int ele_per_thread)
{
    unsigned int global_tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int start_idx = global_tid * ele_per_thread;
    unsigned int end_idx = min(start_idx + ele_per_thread, total_ele);
    for (int i = start_idx; i < end_idx; i++)
    {
        const int channel = (global_tid / width / height) % num_channels;
        const float cur_mean = mean_data[channel];
        const float cur_var = var_data[channel];
        const float cur_weight = weight_data[channel];
        const float cur_bias = bias_data[channel];
        // update input data
        float denominator = sqrt(cur_var + eps);
        input_data[i] = ((input_data[i] - cur_mean) / denominator) * cur_weight + cur_bias;
    }
}

__global__ void BatchNorm2dKernelUpdateByDim(float *input_data, float *mean_data, float *var_data,
                                             float *weight_data, float *bias_data,
                                             float eps, unsigned int batch_size, unsigned int num_channels,
                                             unsigned int height, unsigned int width,
                                             unsigned int total_ele, unsigned int ele_per_thread)
{
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int channel = threadIdx.x;
    unsigned int start_idx = global_tid * ele_per_thread;
    unsigned int end_idx = min(start_idx + ele_per_thread, total_ele);
    for (int i = start_idx; i < end_idx; i++)
    {
        const float cur_mean = mean_data[channel];
        const float cur_var = var_data[channel];
        const float cur_weight = weight_data[channel];
        const float cur_bias = bias_data[channel];
        // update input data
        float denominator = sqrt(cur_var + eps);
        input_data[i] = ((input_data[i] - cur_mean) / denominator) * cur_weight + cur_bias;
    }
}

void hostBatchNorm2d(float *input_data, float *mean_data, float *var_data,
                     float *weight_data, float *bias_data,
                     float eps, unsigned int batch_size, unsigned int num_channels,
                     unsigned int height, unsigned int width)
{
#if INDEX_BY_HW
    unsigned int total_ele = batch_size * num_channels * height * width;
    unsigned int ele_per_thread = 100;
    dim3 dim_grid((int)ceil((float)total_ele / 1024 / ele_per_thread));
    dim3 dim_block(32, 32);
    BatchNorm2dKernelUpdateByHW<<<dim_grid, dim_block>>>(input_data, mean_data, var_data, weight_data, bias_data, eps, batch_size, num_channels, height, width, total_ele, ele_per_thread);
#else
    unsigned int total_ele = batch_size * num_channels * height * width;
    unsigned int ele_per_thread = height * width;
    dim3 dim_grid((int)ceil((float)total_ele / ele_per_thread / num_channels));
    dim3 dim_block(num_channels);
    BatchNorm2dKernelUpdateByDim<<<dim_grid, dim_block>>>(input_data, mean_data, var_data, weight_data, bias_data, eps, batch_size, num_channels, height, width, total_ele, ele_per_thread);
#endif
}