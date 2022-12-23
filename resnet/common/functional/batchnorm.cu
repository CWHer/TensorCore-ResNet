#pragma once

#include "common.h"

__global__ void block_sum_reduce(float *const block_sums, const float *input, const unsigned int len)
{
    extern __shared__ float s_data[];
    int max_ele_per_block = blockDim.x * 2;
    int global_idx = max_ele_per_block * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    // printf("tid=%d, gtid=%d\n", tid, global_idx);

    s_data[tid] = s_data[tid + blockDim.x] = 0.0;
    __syncthreads();

    // copy data to share mem
    if (global_idx < len)
    {
        s_data[tid] = input[global_idx];
        if (global_idx + blockDim.x < len)
            s_data[tid + blockDim.x] = input[global_idx + blockDim.x];
    }
    __syncthreads();

    // do reduction
    for (unsigned int i = blockDim.x; i > 0; i >>= 1)
    {
        if (tid < i)
        {
            s_data[tid] += s_data[tid + i];
        }
        __syncthreads();
    }

    // write the result back
    if (tid == 0)
        block_sums[blockIdx.x] = s_data[0];
}

float array_reduction(float *input, unsigned int len)
{
    float total_sum = 0.0;
    // num of threads in one block
    int block_sizes = 1024;
    // num of elements in one block(binary algorithm)
    int max_eles_per_block = block_sizes * 2;
    // num of blocks in one grid
    int grid_sizes = (int)ceil(float(len) / float(max_eles_per_block));

    // allocate memory for total sum produced by each block
    float *block_sums;
    checkCudaErrors(cudaMalloc(&block_sums, sizeof(float) * grid_sizes));
    checkCudaErrors(cudaMemset(block_sums, 0, sizeof(float) * grid_sizes));
    dim3 grid_dim(grid_sizes);
    dim3 block_dim(block_sizes);
    block_sum_reduce<<<grid_dim, block_dim, sizeof(float) * max_eles_per_block>>>(block_sums, input, len);
    // print_d_array(block_sums, grid_sizes);
    //  sum the elements in the block_sums
    if (grid_sizes <= max_eles_per_block)
    {
        float *total_sum_device;
        checkCudaErrors(cudaMalloc(&total_sum_device, sizeof(float)));
        checkCudaErrors(cudaMemset(total_sum_device, 0, sizeof(float)));
        block_sum_reduce<<<dim3(1), block_dim, sizeof(float) * max_eles_per_block>>>(total_sum_device, block_sums, grid_sizes);
        checkCudaErrors(cudaMemcpy(&total_sum, total_sum_device, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(total_sum_device));
    }
    else
    {
        float *in_block_sums;
        checkCudaErrors(cudaMalloc(&in_block_sums, sizeof(float) * grid_sizes));
        checkCudaErrors(cudaMemcpy(in_block_sums, block_sums, sizeof(float) * grid_sizes, cudaMemcpyDeviceToDevice));
        total_sum = array_reduction(in_block_sums, grid_sizes);
        checkCudaErrors(cudaFree(in_block_sums));
    }
    checkCudaErrors(cudaFree(block_sums));
    return total_sum;
}

float cal_mean(float *input_data, unsigned int channel_data_sizes)
{
    // get sum of array
    float sum = array_reduction(input_data, channel_data_sizes);
    // std::cout << "sum = " << sum << std::endl;
    //  calculate mean
    return sum / channel_data_sizes;
}

__global__ void set_diff2(float *input, const unsigned int ele_per_blocks, const float mean)
{
    unsigned int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idx = global_idx * ele_per_blocks;
    for (size_t i = 0; i < ele_per_blocks; i++)
    {
        float diff = input[idx + i] - mean;
        input[idx + i] = diff * diff;
    }
}

float cal_var(const float *input_data, unsigned int channel_data_sizes, float mean, unsigned int batch_sizes, unsigned int height, unsigned int width)
{
    float *diff2;
    checkCudaErrors(cudaMalloc(&diff2, channel_data_sizes * sizeof(float)));
    checkCudaErrors(cudaMemcpy(diff2, input_data, channel_data_sizes * sizeof(float), cudaMemcpyDeviceToDevice));
    set_diff2<<<dim3(batch_sizes), dim3(height)>>>(diff2, width, mean);
    // get sum of array
    float sum = array_reduction(diff2, channel_data_sizes);
    checkCudaErrors(cudaFree(diff2));
    // calculate mean
    // std::cout << "sum_diff2 = " << sum << std::endl;
    return sum / channel_data_sizes;
}

// update the data
__global__ void BatchNorm2dKernel_update(float *input_data, unsigned int ele_per_blocks, float mean, float var, float eps)
{
    // update input data
    unsigned int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int idx = global_idx * ele_per_blocks;
    float denominator = sqrt(var + eps);
    for (size_t i = 0; i < ele_per_blocks; i++)
    {
        input_data[idx + i] = (input_data[idx + i] - mean) / denominator;
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


// Tensor forward(Tensor x) override
// {
//     unsigned int in_batch_sizes = x.sizes()[0];
//     unsigned int in_channel = x.sizes()[1];
//     unsigned int in_height = x.sizes()[2];
//     unsigned int in_width = x.sizes()[3];
//     float *input_data = x.data_ptr();
//     float *mean = new float[in_channel]; // mean
//     float *var = new float[in_channel];  // standard deviation
//     // initialize
//     for (int i = 0; i < in_channel; i++)
//     {
//         mean[i] = 0.0;
//         var[i] = 1.0;
//     }

//     // gpu execution
//     // 1. allocate memory and copy data to device
//     unsigned int channel_data_sizes = in_batch_sizes * in_height * in_width;
//     float *single_channel_data;
//     checkCudaErrors(cudaMalloc(&single_channel_data, channel_data_sizes * sizeof(float)));
//     // 2. do the computation
//     for (int i = 0; i < in_channel; i++)
//     {
//         // construct the array
//         get_single_channel_data(input_data, single_channel_data, in_batch_sizes, in_channel, in_height, in_width, i);
//         // 2.1. calculate mean
//         mean[i] = cal_mean(single_channel_data, channel_data_sizes);
//         // std::cout << "mean[" << i << "] = " << mean[i] << std::endl;

//         // 2.2. calculate population standard deviation and sample standard deviation
//         var[i] = cal_var(single_channel_data, channel_data_sizes, mean[i], in_batch_sizes, in_height, in_width);
//         // std::cout << "var[" << i << "] = " << var[i] << std::endl;

//         // 2.3. update data
//         dim3 dim_grid(in_batch_sizes);
//         dim3 dim_block(in_height);
//         BatchNorm2dKernel_update<<<dim_grid, dim_block>>>(single_channel_data, in_width, mean[i], var[i], eps);
//         // copy the result to output
//         restore_single_channel_data(input_data, single_channel_data, in_batch_sizes, in_channel, in_height, in_width, i);
//     }
//     checkCudaErrors(cudaFree(single_channel_data));
//     return x;
// }
