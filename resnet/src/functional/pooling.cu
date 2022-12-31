#include "pooling.h"

template <int block_size>
__global__ void deviceMaxPool2dKernel(float *input_data, int height, int width,
                                      float *output_data, int out_height, int out_width,
                                      int kernel_size, int padding, int stride)
{
    int input_grid_offset = blockIdx.x * height * width;
    int output_grid_offset = blockIdx.x * out_height * out_width;

    int row = block_size * threadIdx.y;
    int col = block_size * threadIdx.x;
    for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
            if (row + i < out_height && col + j < out_width)
            {
                float value = -1e10;
                for (int x = 0; x < kernel_size; ++x)
                    for (int y = 0; y < kernel_size; ++y)
                    {
                        int input_row = (row + i) * stride + x - padding;
                        int input_col = (col + j) * stride + y - padding;
                        if (0 <= input_row && input_row < height &&
                            0 <= input_col && input_col < width)
                        {
                            float input_value = input_data[input_grid_offset +
                                                           input_row * width + input_col];
                            value = max(value, input_value);
                        }
                    }
                output_data[output_grid_offset +
                            (row + i) * out_width + (col + j)] = value;
            }
}

void hostMaxPool2d(int batch_size, int num_channels,
                   float *input_data, int height, int width,
                   float *output_data, int out_height, int out_width,
                   int kernel_size, int padding, int stride)
{
    static const int BLOCK_SIZE = 4;
    dim3 grid_dim(batch_size * num_channels);
    dim3 block_dim((out_height - 1) / BLOCK_SIZE + 1,
                   (out_width - 1) / BLOCK_SIZE + 1);
    deviceMaxPool2dKernel<4><<<grid_dim, block_dim>>>(
        input_data, height, width, output_data, out_height, out_width,
        kernel_size, padding, stride);
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void deviceAvgPool2dKernel(float *input_data, int height, int width,
                                      float *output_data, int out_height, int out_width,
                                      int kernel_size, int padding, int stride)
{
    // NOTE: ... (block) x 128 (thread) x H x W (within)
    int input_grid_offset = blockDim.x * height * width;
    int output_grid_offset = blockDim.x * out_height * out_width;
    int input_thread_offset = height * width;
    int output_thread_offset = out_height * out_width;

    input_data += blockIdx.x * input_grid_offset +
                  threadIdx.x * input_thread_offset;
    output_data += blockIdx.x * output_grid_offset +
                   threadIdx.x * output_thread_offset;

    int output_offset = 0;
    float r = 1.0f / (kernel_size * kernel_size);
    for (int i = kernel_size - 1 - padding; i < height + padding; i += stride)
        for (int j = kernel_size - 1 - padding; j < width + padding; j += stride)
        {
            float ret = 0;
            for (int x = i - kernel_size + 1; x <= i; x++)
                for (int y = j - kernel_size + 1; y <= j; y++)
                {
                    float value = 0 <= x && x < height &&
                                          0 <= y && y < width
                                      ? input_data[x * width + y]
                                      : 0; // zero padding
                    ret += value;
                }
            output_data[output_offset++] = ret * r;
        }
}

void hostAvgPool2d(int batch_size, int num_channels,
                   float *input_data, int height, int width,
                   float *output_data, int out_height, int out_width,
                   int kernel_size, int padding, int stride)
{
    static const int N_THREADS = 128;
    dim3 block_dim(N_THREADS);
    dim3 grid_dim(batch_size * num_channels / N_THREADS);
    deviceAvgPool2dKernel<<<grid_dim, block_dim>>>(
        input_data, height, width, output_data, out_height, out_width,
        kernel_size, padding, stride);
    checkCudaErrors(cudaDeviceSynchronize());
}