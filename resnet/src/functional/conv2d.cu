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

__global__ [[maybe_unused]] void col2im_loop(float *dst, const float *src, int C, int H, int W) {
  CUDA_KERNEL_LOOP(loop_idx, C * H * W) {
    int curr_index = loop_idx;
    int w = curr_index % W;
    curr_index /= W;
    int h = curr_index % H;
    curr_index /= H;
    int c = curr_index;
    if (c > C || h > H || w > W) {
      continue;
    }

    dst[c * (H * W) + h * W + w] = src[c + (h * W + w) * C];
  }
}

[[maybe_unused]] void col2im_cuda(float *dst, const float *source, int C, int H, int W) {
  float *dst_device;
  cudaMalloc(&dst_device, sizeof(float) * C * H * W);

  float *src_device;
  cudaMalloc(&src_device, sizeof(float) * C * H * W);
  cudaMemcpy(src_device, source, sizeof(float) * C * H * W, cudaMemcpyHostToDevice);

  col2im_loop<<<KERNEL_LOOP_BLOCKS(C * H * W), KERNEL_LOOP_THREADS>>>(dst_device, src_device, C, H, W);

  cudaMemcpy(dst, dst_device, sizeof(float) * C * H * W, cudaMemcpyDeviceToHost);
  cudaFree(dst_device);
  cudaFree(src_device);
}

/** @brief Convolutional layer forward propagation.
 *
 * @param input Input float, row major (organizes at last element), of shape (C, H, W).
 * @param output Result float, row major (organizes at last element), of shape (out_channels, H_out, W_out).
 *  where H_out = floor((H + 2 * padding - kernel_size) / stride) + 1
 *  and W_out = floor((W + 2 * padding - kernel_size) / stride) + 1
 *  @param weight Weight float_16, row major (organizes at last element), of shape (out_channels, C, kernel_size, kernel_size).
 *  @param bias Bias float, row major (organizes at last element), in shape of (out_channels).
 *  @param C Number of input channels.
 *  @param H Height of input.
 *  @param W Width of input.
 *  @param kernel_size Size of kernel (kernel_size x kernel_size).
 *  @param stride Stride of convolution.
 *  @param padding Padding of convolution.
 */
void conv2d(float *input,
            float *output,
            float_16 *weight,
            float *bias,
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

  auto im2col_result = create_im2col_result_store(C, H, W, kernel_size, kernel_size, stride, padding);
  // After im2col, im2col_result is of shape (C * kernel_size * kernel_size, H_out * W_out)
  im2col(input, im2col_result.get(), C, H, W, kernel_size, kernel_size, stride, padding);


  // Multiply with weight, shaped as (out_channels, C * kernel_size * kernel_size)
  auto gemm_result = std::make_unique<float[]>(conv_result_size * out_channels);

  gemm(weight,
       im2col_result.get(),
       output,
       out_channels,
       conv_result_size,
       expanded_kernel_width,
       GEMM::Major::row_major,
       Impl::DeviceType::CPU);

  // Expanding bias from (out_channels) to (out_channels, output_height, output_width)
  auto bias_expanded = std::make_unique<float[]>(out_channels * conv_result_size);

  for (int i = 0; i < out_channels; ++i) {
    for (int j = 0; j < conv_result_size; ++j) {
      bias_expanded[i * conv_result_size + j] = bias[i];
    }
  }

  // Add bias
  _add(output, bias_expanded.get(), out_channels * output_height * output_width);

  check_cuda_error();
}

/**
 * @copydoc conv2d
 * @brief Convolutional layer result shape
 */
std::vector<int> conv2d_result_shape(int C, int H, int W, int out_channels, int kernel_size, int stride, int padding) {
  int output_height = (H + 2 * padding - kernel_size) / stride + 1;
  int output_width = (W + 2 * padding - kernel_size) / stride + 1;
  return {out_channels, output_height, output_width};
}
