/**
 * @file test/test_cuda_reduction.cu
 * @brief Test the reduction of cuda
 */

#include <gtest/gtest.h>
#include "common.h"

float reduction_cpu(float *input, int len)
{
    float sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        sum += input[i];
    }
    return sum;
}

void print_d_array(float *d_array, int len)
{
    float *h_array = new float[len];
    checkCudaErrors(cudaMemcpy(h_array, d_array, sizeof(float) * len, cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i < len; ++i)
    {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_array;
}

__global__ void block_sum_reduce(float *const block_sums, const float *input, const int len)
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

float reduction_gpu(float *input, int len)
{
    float total_sum = 0.0;
    // num of threads in one block
    int block_sizes = 1024;
    // num of elements in one block(binary algorithm)
    int max_eles_per_block = block_sizes * 2;
    // num of blocks in one grid
    int grid_sizes = (int)std::ceil(float(len) / float(max_eles_per_block));

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
        total_sum = reduction_gpu(in_block_sums, grid_sizes);
        checkCudaErrors(cudaFree(in_block_sums));
    }
    checkCudaErrors(cudaFree(block_sums));
    return total_sum;
}

TEST(cuda_reduction, reduction)
{
    int num_elements = 4000;
    // create an array
    float *array_host = new float[num_elements];
    // init
    for (int i = 1; i < num_elements; i++)
    {
        array_host[i] = randomFloat(-1.0, 1.0);
    }

    // do reduction in cpu
    float res_cpu = reduction_cpu(array_host, num_elements);

    // do reduction in gpu
    float *array_device;
    checkCudaErrors(cudaMalloc(&array_device, sizeof(float) * num_elements));
    checkCudaErrors(cudaMemcpy(array_device, array_host, sizeof(float) * num_elements, cudaMemcpyHostToDevice));
    // print_d_array(array_device, num_elements);
    float res_gpu = reduction_gpu(array_device, num_elements);
    checkCudaErrors(cudaFree(array_device));

    // check correctness
    float eps = 1e-1;
    // std::cout << "res_cpu = " << res_cpu << "\t res_gpu = " << res_gpu << std::endl;
    EXPECT_NEAR(res_cpu, res_gpu, eps);
}