/** @file linear.cu
*/

// Linear layer is performed by trivial GEMM

#include "functional/gemm.hpp"
#include "functional/add.hpp"
#include "functional/linear.hpp"
#include "functional/macros.hpp"

/**
 * @brief Linear layer implementation.
 *
 * @param input Input tensor of shape (batch_size, input_channel)
 * @param output Output tensor of shape (batch_size, output_channel)
 * @param weight Weight tensor of shape (input_channel, output_channel)
 * @param bias Bias tensor of shape (output_channel)
 * @param input_channel
 * @param output_channel
 */
void linear(const float_16 * RESTRICT input,
            float * RESTRICT output,
            const float_16 * RESTRICT weight,
            const float * RESTRICT bias,
            int batch_size,
            int input_channel,
            int output_channel,
            Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU : {
    gemm(input,
         weight,
         output,
         batch_size,
         output_channel,
         input_channel,
         GEMM::Major::row_major,
         Impl::DeviceType::CPU);
  }
    break;
  case Impl::DeviceType::CUDA : {
    gemm(input,
         weight,
         output,
         batch_size,
         output_channel,
         input_channel,
         GEMM::Major::row_major,
         Impl::DeviceType::CUDA);
  }
    break;
  }

  constexpr int stream_num = 8;
  cudaStream_t streams[stream_num];
  for (auto &stream : streams) {
    cudaStreamCreate(&stream);
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    add_(&output[batch * output_channel], bias, output_channel, device_type, streams[batch % stream_num]);
  }

  for (auto &stream : streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
}

__global__ void transpose_kernel(const float * RESTRICT input, float_16 * RESTRICT output, int m, int n) {
  CUDA_KERNEL_LOOP(index, m * n) {
    auto i = index / n;
    auto j = index % n;
    if (i < m && j < n) {
      output[j * m + i] = __float2half(input[i * n + j]);
    }
  }
}

void prepare_linear_weight(const float *RESTRICT input, float_16 *RESTRICT output, int row, int col, Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU : {
    for (int i = 0; i < row; ++i) {
      for (int j = 0; j < col; ++j) {
        output[j * row + i] = __float2half(input[i * col + j]);
      }
    }
  }
    break;
  case Impl::DeviceType::CUDA : {
    transpose_kernel<<<KERNEL_LOOP_BLOCKS(row * col), KERNEL_LOOP_THREADS>>>(input, output, row, col);
  }
    break;
  }
}
