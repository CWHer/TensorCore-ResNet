/** @file conv2d.cu
*/
#include <memory>

#include "functional/gemm.hpp"
#include "functional/conv2d.hpp"
#include "functional/macros.h"
#include "functional/add.hpp"

static void check_cuda_error() {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

__global__ static void cuda_bias_extend(float *extended,
                                        const float *bias,
                                        int N,
                                        int out_channels,
                                        int conv_result_size) {
  CUDA_KERNEL_LOOP(i, N * out_channels * conv_result_size) {
    auto curr = i;
    auto conv_result_idx = curr % conv_result_size;
    curr /= conv_result_size;
    auto out_channel_idx = curr % out_channels;
    curr /= out_channels;
    auto batch_idx = curr % N;

    extended[batch_idx * out_channels * conv_result_size + out_channel_idx * conv_result_size + conv_result_idx] =
        bias[out_channel_idx];
  }
}

void bias_extend(float *Result,
                 const float *bias,
                 int N,
                 int out_channels,
                 int conv_result_size,
                 Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU: {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < out_channels; ++j) {
        for (int k = 0; k < conv_result_size; ++k) {
          Result[i * out_channels * conv_result_size + j * conv_result_size + k] = bias[j];
        }
      }
    }

  }
    break;
  case Impl::DeviceType::CUDA:
    cuda_bias_extend<<<KERNEL_LOOP_BLOCKS(N * out_channels * conv_result_size), KERNEL_LOOP_THREADS>>>(Result,
                                                                                                       bias,
                                                                                                       N,
                                                                                                       out_channels,
                                                                                                       conv_result_size);
    break;
  }
}

/**
 * @copydoc conv2d
 * @brief Convolutional layer result sizes
 */
int conv2d_output_sizes(int N, int C, int H, int W, int out_channels, int kernel_size, int stride, int padding) {
  auto shape = conv2d_result_shape(N, C, H, W, out_channels, kernel_size, stride, padding);
  return shape[0] * shape[1] * shape[2] * shape[3];
}

/**
 * @copydoc conv2d
 * @brief Convolutional layer result shape
 */
std::vector<int> conv2d_result_shape(int N,
                                     int C,
                                     int H,
                                     int W,
                                     int out_channels,
                                     int kernel_size,
                                     int stride,
                                     int padding) {
  int output_height = (H + 2 * padding - kernel_size) / stride + 1;
  int output_width = (W + 2 * padding - kernel_size) / stride + 1;
  return {N, out_channels, output_height, output_width};
}

/** @brief Convolutional layer forward propagation.
 *
 * @param input Input float, row major (organizes at last element), of shape (N, C, H, W).
 * @param output Result float, row major (organizes at last element), of shape (N, out_channels, H_out, W_out).
 *  where H_out = floor((H + 2 * padding - kernel_size) / stride) + 1
 *  and W_out = floor((W + 2 * padding - kernel_size) / stride) + 1
 *  @param weight Weight float_16, row major (organizes at last element), of shape (out_channels, C, kernel_size, kernel_size).
 *  @param bias Bias float, row major (organizes at last element), in shape of (out_channels).
 *  @param N Batch size.
 *  @param C Number of input channels.
 *  @param H Height of input.
 *  @param W Width of input.
 *  @param kernel_size Size of kernel (kernel_size x kernel_size).
 *  @param stride Stride of convolution.
 *  @param padding Padding of convolution.
 */
void conv2d(const float *input,
            float *output,
            const float_16 *weight,
            const float *bias,
            int N,
            int C,
            int H,
            int W,
            int out_channels,
            int kernel_size,
            int stride,
            int padding,
            Impl::DeviceType device_type) {
  int output_height = (H + 2 * padding - kernel_size) / stride + 1;
  int output_width = (W + 2 * padding - kernel_size) / stride + 1;
  int conv_result_size = output_height * output_width;
  int expanded_kernel_width = C * kernel_size * kernel_size;

  float *bias_expanded;

  int batched_n = 128;

  if (device_type == Impl::DeviceType::CUDA) {
    auto total_batch_size = (N + batched_n - 1) / batched_n;
    if (total_batch_size == 1) {
      auto im2col_result = create_im2col_result_store_device(N, C, H, W, kernel_size, kernel_size, stride, padding);
      im2col(input, im2col_result.get(), N, C, H, W, kernel_size, kernel_size, stride, padding, device_type);
      gemm_batched_B(weight,
                     im2col_result.get(),
                     output,
                     out_channels,
                     conv_result_size,
                     expanded_kernel_width,
                     N,
                     GEMM::Major::row_major,
                     device_type);
    } else {
      auto im2col_result =
          create_im2col_result_store_device(batched_n, C, H, W, kernel_size, kernel_size, stride, padding);
      for (int i = 0; i < total_batch_size; ++i) {
        auto batch_size = std::min(batched_n, N - i * batched_n);
        if (batch_size == 0) {
          break;
        }
        if (batch_size != batched_n) {
          im2col_result =
              create_im2col_result_store_device(batch_size, C, H, W, kernel_size, kernel_size, stride, padding);
        }
        im2col(input + i * batched_n * C * H * W,
               im2col_result.get(),
               batch_size,
               C,
               H,
               W,
               kernel_size,
               kernel_size,
               stride,
               padding,
               device_type);
        gemm_batched_B(weight,
                       im2col_result.get(),
                       output + i * batched_n * out_channels * conv_result_size,
                       out_channels,
                       conv_result_size,
                       expanded_kernel_width,
                       batch_size,
                       GEMM::Major::row_major,
                       device_type);
      }
    }

    cudaMalloc(&bias_expanded, out_channels * conv_result_size * sizeof(float));
  } else {
    auto im2col_result = create_im2col_result_store_host(N, C, H, W, kernel_size, kernel_size, stride, padding);
    // After im2col, im2col_result is of shape (N, C * kernel_size * kernel_size, H_out * W_out)
    im2col(input, im2col_result.get(), N, C, H, W, kernel_size, kernel_size, stride, padding, device_type);

    /**
     * FIXME: Current im2col implementation, according to pytorch, moves N to the top level, which is not efficient.
     * the N should be moved to make sure the number of column be C * kernel_size * kernel_size, then we can utilize
     * a large GEMM and then swap the dimensions to the right shape.
     */
    gemm_batched_B(weight,
                   im2col_result.get(),
                   output,
                   out_channels,
                   conv_result_size,
                   expanded_kernel_width,
                   N,
                   GEMM::Major::row_major,
                   device_type);

    bias_expanded = new float[out_channels * conv_result_size];
  }

  bias_extend(bias_expanded, bias, 1, out_channels, conv_result_size, device_type);

  for (int i = 0; i < N; ++i) {
    add_(output + i * out_channels * conv_result_size, bias_expanded, out_channels * conv_result_size, device_type);
  }

  if (device_type == Impl::DeviceType::CUDA) {
    cudaFree(bias_expanded);
  } else {
    delete[] bias_expanded;
  }

  check_cuda_error();
}

void conv2d(const float *input,
            float *output,
            const float *weight,
            const float *bias,
            int N,
            int C,
            int H,
            int W,
            int out_channels,
            int kernel_size,
            int stride,
            int padding,
            Impl::DeviceType device_type) {
  int output_height = (H + 2 * padding - kernel_size) / stride + 1;
  int output_width = (W + 2 * padding - kernel_size) / stride + 1;
  int conv_result_size = output_height * output_width;
  auto weight_shape = out_channels * conv_result_size;
  // copy weight to float_16
  switch (device_type) {
  case Impl::DeviceType::CPU: {
    auto weight_16 = new float_16[weight_shape];
    for (int i = 0; i < weight_shape; i++) {
      weight_16[i] = weight[i];
    }
    conv2d(input, output, weight_16, bias, N, C, H, W, out_channels, kernel_size, stride, padding, device_type);
    delete[] weight_16;
    break;
  }
  case Impl::DeviceType::CUDA: {
    float_16 *weight_16;
    cudaMalloc(&weight_16, weight_shape * sizeof(float_16));
    float_16 *weight_16_ptr = weight_16;
    for (int i = 0; i < weight_shape; i++) {
      *weight_16_ptr = weight[i];
      weight_16_ptr++;
    }
    conv2d(input, output, weight_16, bias, N, C, H, W, out_channels, kernel_size, stride, padding, device_type);
    cudaFree(weight_16);
    break;
  }
  }
}

