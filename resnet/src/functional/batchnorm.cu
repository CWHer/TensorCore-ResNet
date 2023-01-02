#include "common.hpp"

template<int block_size>
__global__ static void dimBatchNorm2dKernel(float *RESTRICT input_data, const float *RESTRICT mean_data,
                                            const float *RESTRICT var_data, const float *RESTRICT weight_data,
                                            const float *RESTRICT bias_data, float eps, unsigned int batch_size,
                                            unsigned int num_channels, unsigned int height, unsigned int width,
                                            unsigned int total_channels) {
  const auto channel_num = blockIdx.x * blockDim.z + threadIdx.z;
  if (channel_num >= total_channels)
    return;

  const auto row = block_size * threadIdx.y;
  auto col = block_size * threadIdx.x;

  const auto channel = channel_num % num_channels;
  const auto cur_mean = mean_data[channel];
  const auto cur_var = var_data[channel];
  const auto cur_weight = weight_data[channel];
  const auto cur_bias = bias_data[channel];
  auto r = 1.0f / sqrtf(cur_var + eps);

  const auto offset = channel_num * height * width;
  input_data += offset;

  for (int i = 0; i < block_size; i++)
    for (int j = 0; j < block_size; j++)
      if (likely(row + i < height && col + j < width))
        input_data[(row + i) * width + col + j] =
            ((input_data[(row + i) * width + col + j] - cur_mean) * r) * cur_weight + cur_bias;
}

template<int block_size>
__global__ static void dimBatchNorm2dReluKernel(float *RESTRICT to_be_modified_data, const float *RESTRICT mean_data,
                                                const float *RESTRICT var_data, const float *RESTRICT weight_data,
                                                const float *RESTRICT bias_data, float eps, unsigned int batch_size,
                                                unsigned int num_channels, unsigned int height, unsigned int width,
                                                unsigned int total_channels) {
  const auto channel_num = blockIdx.x * blockDim.z + threadIdx.z;
  if (unlikely(channel_num >= total_channels))
    return;

  const auto row = block_size * threadIdx.y;
  auto col = block_size * threadIdx.x;

  const auto channel = channel_num % num_channels;
  const auto cur_mean = mean_data[channel];
  const auto cur_var = var_data[channel];
  const auto cur_weight = weight_data[channel];
  const auto cur_bias = bias_data[channel];
  auto r = 1.0f / sqrtf(cur_var + eps);

  const auto offset = channel_num * height * width;
  to_be_modified_data += offset;

  for (int i = 0; i < block_size; i++)
    for (int j = 0; j < block_size; j++)
      if (likely(row + i < height && col + j < width)) {
        auto value = ((to_be_modified_data[(row + i) * width + col + j] - cur_mean) * r) * cur_weight + cur_bias;
        to_be_modified_data[(row + i) * width + col + j] = value > 0.0f ? value : 0.0f;
      }

}

void hostBatchNorm2d(float * RESTRICT to_be_modified_data, const float *RESTRICT mean_data,
                     const float * RESTRICT var_data, const float *RESTRICT weight_data,
                     const float * RESTRICT bias_data, float eps, int batch_size, int num_channels,
                     int height, int width) {
  // shape of input varies from (64, 112, 112) to (512, 7, 7)
  static const int N_THREADS = 128;
  static const int BLOCK_SIZE = 4;
  int n_hblock = (width - 1) / BLOCK_SIZE + 1;
  int n_vblock = (height - 1) / BLOCK_SIZE + 1;
  int n_blocks = std::max(1, N_THREADS / (n_hblock * n_vblock));
  dim3 grid_dim((batch_size * num_channels - 1) / n_blocks + 1);
  dim3 block_dim(n_hblock, n_vblock, n_blocks);
  dimBatchNorm2dKernel<BLOCK_SIZE><<<grid_dim, block_dim>>>(
      to_be_modified_data, mean_data, var_data, weight_data, bias_data,
      eps, batch_size, num_channels, height, width, batch_size * num_channels);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void hostBatchNorm2dRelu(float * RESTRICT to_be_modified_data, const float * RESTRICT mean_data,
                         const float * RESTRICT var_data, const float *RESTRICT weight_data,
                         const float *RESTRICT bias_data, float eps, int batch_size, int num_channels,
                         int height, int width) {
  // shape of input varies from (64, 112, 112) to (512, 7, 7)
  static const int N_THREADS = 128;
  static const int BLOCK_SIZE = 4;
  int n_hblock = (width - 1) / BLOCK_SIZE + 1;
  int n_vblock = (height - 1) / BLOCK_SIZE + 1;
  int n_blocks = std::max(1, N_THREADS / (n_hblock * n_vblock));
  dim3 grid_dim((batch_size * num_channels - 1) / n_blocks + 1);
  dim3 block_dim(n_hblock, n_vblock, n_blocks);
  dimBatchNorm2dReluKernel<BLOCK_SIZE><<<grid_dim, block_dim>>>(
      to_be_modified_data, mean_data, var_data, weight_data, bias_data,
      eps, batch_size, num_channels, height, width, batch_size * num_channels);
  checkCudaErrors(cudaPeekAtLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}
