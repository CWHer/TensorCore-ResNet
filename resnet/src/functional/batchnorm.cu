#include "batchnorm.h"

template <int block_size>
__global__ void dimBatchNorm2dKernel(float *input_data, float *output_data,
                                     float *mean_data, float *var_data,
                                     float *weight_data, float *bias_data,
                                     float eps, int batch_size, int num_channels,
                                     int height, int width, int total_channels)
{
    int channel_num = blockIdx.x * blockDim.z + threadIdx.z;
    // clang-format off
    if (channel_num >= total_channels) return;
    // clang-format on

    int row = block_size * threadIdx.y;
    int col = block_size * threadIdx.x;

    const int channel = channel_num % num_channels;
    const float cur_mean = mean_data[channel];
    const float cur_var = var_data[channel];
    const float cur_weight = weight_data[channel];
    const float cur_bias = bias_data[channel];
    float r = 1.0f / sqrtf(cur_var + eps);

    int offset = channel_num * height * width;
    input_data += offset, output_data += offset;

    for (int i = 0; i < block_size; i++)
        for (int j = 0; j < block_size; j++)
            if (row + i < height && col + j < width)
                output_data[(row + i) * width + col + j] =
                    ((input_data[(row + i) * width + col + j] - cur_mean) * r) * cur_weight + cur_bias;
}

void hostBatchNorm2d(float *input_data, float *output_data,
                     float *mean_data, float *var_data,
                     float *weight_data, float *bias_data,
                     float eps, int batch_size, int num_channels,
                     int height, int width)
{
    // shape of input varies from (64, 112, 112) to (512, 7, 7)
    static const int N_THREADS = 128;
    static const int BLOCK_SIZE = 7;
    int n_hblock = (width - 1) / BLOCK_SIZE + 1;
    int n_vblock = (height - 1) / BLOCK_SIZE + 1;
    int n_blocks = std::max(1, N_THREADS / (n_hblock * n_vblock));
    dim3 grid_dim((batch_size * num_channels - 1) / n_blocks + 1);
    dim3 block_dim(n_hblock, n_vblock, n_blocks);
    dimBatchNorm2dKernel<BLOCK_SIZE><<<grid_dim, block_dim>>>(
        input_data, output_data, mean_data, var_data, weight_data, bias_data,
        eps, batch_size, num_channels, height, width, batch_size * num_channels);
    checkCudaErrors(cudaDeviceSynchronize());
}