/** @file conv2d.cu
*/
#include <memory>

#include "functional/gemm.hpp"
#include "functional/conv2d.hpp"
#include "functional/macros.h"

static void check_cuda_error() {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

void _add(float *Result, const float *adder, int length) {
  for (int i = 0; i < length; ++i) {
    Result[i] += adder[i];
  }
}

[[maybe_unused]] void col2im_naive(float *dst, const float *source, int C, int H, int W) {
  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        dst[c * (H * W) + h * W + w] = source[c + (h * W + w) * C];
      }
    }
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

void gemm_batched_B(const float_16 *A,
                    const float_16 *B,
                    float_32 *C,
                    size_t M,
                    size_t N,
                    size_t K,
                    size_t batch_size,
                    GEMM::Major major,
                    Impl::DeviceType device_type) {
  if (major != GEMM::Major::row_major) {
    // Batches does not support col-major
    throw std::runtime_error("Batches does not support col-major");
  }

  for (int i = 0; i < batch_size; i++) {
    auto B_ptr = B + i * K * N;
    auto C_ptr = C + i * M * N;
    gemm(A, B_ptr, C_ptr, M, N, K, major, device_type);
  }

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
            int padding) {
  int output_height = (H + 2 * padding - kernel_size) / stride + 1;
  int output_width = (W + 2 * padding - kernel_size) / stride + 1;
  int conv_result_size = output_height * output_width;
  int expanded_kernel_width = C * kernel_size * kernel_size;

  auto im2col_result = create_im2col_result_store(N, C, H, W, kernel_size, kernel_size, stride, padding);
  // After im2col, im2col_result is of shape (N, C * kernel_size * kernel_size, H_out * W_out)
  im2col(input, im2col_result.get(), N, C, H, W, kernel_size, kernel_size, stride, padding, Impl::DeviceType::CPU);

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
                 Impl::DeviceType::CPU);

  // Expanding bias from (out_channels) to (N, out_channels, output_height, output_width)
  auto bias_expanded = std::make_unique<float[]>(N * out_channels * conv_result_size);

  for (int n = 0; n < N; ++n) {
    for (int i = 0; i < out_channels; ++i) {
      for (int j = 0; j < conv_result_size; ++j) {
        bias_expanded[n * out_channels * conv_result_size + i * conv_result_size + j] = bias[i];
      }
    }
  }

  // Add bias
  _add(output, bias_expanded.get(), N * out_channels * conv_result_size);

  check_cuda_error();
}

