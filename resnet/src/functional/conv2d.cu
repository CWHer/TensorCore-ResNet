/** @file conv2d.cu
*/
#include <memory>

#include "functional/gemm.hpp"
#include "functional/conv2d.hpp"
#include "functional/macros.h"

/* @brief im2col result shape
 * Should not be used.
 */
size_t im2col_result_size(int C, int H, int W, int filter_height, int filter_width, int stride, int padding) {
  int output_height = (H + 2 * padding - filter_height) / stride + 1;
  int output_width = (W + 2 * padding - filter_width) / stride + 1;
  int output_size = output_height * output_width;
  return C * filter_height * filter_width * output_size;
}

std::unique_ptr<float_16[]> create_im2col_result_store(int C,
                                                       int H,
                                                       int W,
                                                       int filter_height,
                                                       int filter_width,
                                                       int stride,
                                                       int padding) {
  // Allocate memory for im2col result.
  int output_height = (H + 2 * padding - filter_height) / stride + 1;
  int output_width = (W + 2 * padding - filter_width) / stride + 1;
  int output_size = output_height * output_width;

  size_t im2col_size = C * filter_height * filter_width * output_size;
  return std::make_unique<float_16[]>(im2col_size);
}

/**
 * @brief im2col, performs as the matlab function.
 *
 * @param input float of shape (C, H, W)
 * @param output Output should be shaped as:
 * (output_height * output_width, C * filter_height * filter_width)
 *
 * where output_height = (H + 2 * padding - filter_height) / stride + 1
 * and output_width = (W + 2 * padding - filter_width) / stride + 1
 *
 * Data are arranged per channel of columns, this results in factor C.
 */
void im2col_naive(float *input,
                  float_16 *output,
                  int C,
                  int H,
                  int W,
                  int filter_height,
                  int filter_width,
                  int stride,
                  int padding) {
  int output_height = (H + 2 * padding - filter_height) / stride + 1;
  int output_width = (W + 2 * padding - filter_width) / stride + 1;
  int output_size = output_height * output_width;

  int filter_size = filter_height * filter_width;

  int output_ldm = C * filter_size;

  int input_channel_size = H * W;

  for (int output_h = 0; output_h < output_height; ++output_h) {
    for (int output_w = 0; output_w < output_width; ++output_w) {
      for (int filter_h = 0; filter_h < filter_height; ++filter_h) {
        for (int filter_w = 0; filter_w < filter_width; ++filter_w) {
          for (int c = 0; c < C; ++c) {
            int index_h = output_h * stride + filter_h - padding;
            int index_w = output_w * stride + filter_w - padding;
            int input_index = c * input_channel_size + index_h * W + index_w;
            int output_index = output_h * output_width + output_w;
            int output_offset = c * filter_size + filter_h * filter_width + filter_w;
            if (index_h >= 0 && index_h < H && index_w >= 0 && index_w < W) {
              output[output_index * output_ldm + output_offset] = input[input_index];
            } else {
              output[output_index * output_ldm + output_offset] = 0;
            }
          }
        }
      }
    }
  }
}

__global__ void im2col_cuda_kernel(float *input,
                                   float_16 *output,
                                   int C,
                                   int H,
                                   int W,
                                   int filter_height,
                                   int filter_width,
                                   int stride,
                                   int padding) {
  // Use CUDA loop
  int output_height = (H + 2 * padding - filter_height) / stride + 1;
  int output_width = (W + 2 * padding - filter_width) / stride + 1;
  int output_size = output_height * output_width;

  int filter_size = filter_height * filter_width;

  int output_ldm = C * filter_size;

  int input_channel_size = H * W;
  CUDA_KERNEL_LOOP(loop_idx, output_size * filter_size * C) {
    // order: output_h, output_w, filter_h, filter_w, c
    int curr_index = loop_idx;
    int c = curr_index % C;
    curr_index /= C;
    int filter_w = curr_index % filter_width;
    curr_index /= filter_width;
    int filter_h = curr_index % filter_height;
    curr_index /= filter_height;
    int output_w = curr_index % output_width;
    curr_index /= output_width;
    int output_h = curr_index;

    int index_h = output_h * stride + filter_h - padding;
    int index_w = output_w * stride + filter_w - padding;
    int input_index = c * input_channel_size + index_h * W + index_w;
    if (input_index > C * H * W) {
      continue;
    }
    int output_index = output_h * output_width + output_w;
    int output_offset = c * filter_size + filter_h * filter_width + filter_w;
    if (index_h >= 0 && index_h < H && index_w >= 0 && index_w < W) {
      output[output_index * output_ldm + output_offset] = input[input_index];
    } else {
      output[output_index * output_ldm + output_offset] = 0;
    }
  }

  __syncthreads();

}

void im2col_cuda(float *input,
                 float_16 *output,
                 int C,
                 int H,
                 int W,
                 int filter_height,
                 int filter_width,
                 int stride,
                 int padding) {
  // Copy input to device
  float *input_device;
  cudaMalloc(&input_device, sizeof(float) * C * H * W);
  cudaMemcpy(input_device, input, sizeof(float) * C * H * W, cudaMemcpyHostToDevice);

  auto result_size = im2col_result_size(C, H, W, filter_height, filter_width, stride, padding);
  float_16 *output_device;
  cudaMalloc(&output_device, sizeof(float_16) * result_size);

  // Launch CUDA kernel
  im2col_cuda_kernel<<<KERNEL_LOOP_BLOCKS(result_size), KERNEL_LOOP_THREADS>>>(input_device,
                                                                               output_device,
                                                                               C,
                                                                               H,
                                                                               W,
                                                                               filter_height,
                                                                               filter_width,
                                                                               stride,
                                                                               padding);

  // Copy result back to host
  cudaMemcpy(output, output_device, sizeof(float_16) * result_size, cudaMemcpyDeviceToHost);
  // Free
  cudaFree(input_device);
  cudaFree(output_device);
}

void _add(float *Result, const float *adder, int length) {
  for (int i = 0; i < length; ++i) {
    Result[i] += adder[i];
  }
}

[[unused]] void col2im_naive(float *dst, const float *source, int C, int H, int W) {
  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        dst[c * (H * W) + h * W + w] = source[c + (h * W + w) * C];
      }
    }
  }
}

__global__ void col2im_loop(float *dst, const float *src, int C, int H, int W) {
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

void col2im_cuda(float *dst, const float *source, int C, int H, int W) {
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
  auto im2col_result = create_im2col_result_store(C, H, W, kernel_size, kernel_size, stride, padding);
  im2col(input, im2col_result.get(), C, H, W, kernel_size, kernel_size, stride, padding);
  // Multiply with weight
  int output_height = (H + 2 * padding - kernel_size) / stride + 1;
  int output_width = (W + 2 * padding - kernel_size) / stride + 1;
  int im2col_result_height = output_height * output_width;
  int im2col_result_width = C * kernel_size * kernel_size;

  auto gemm_result = std::make_unique<float[]>(im2col_result_height * out_channels);
  gemm_row_major(im2col_result.get(),
                 weight,
                 gemm_result.get(),
                 im2col_result_height,
                 out_channels,
                 im2col_result_width);
  // shape back
  col2im_naive(output, gemm_result.get(), out_channels, output_height, output_width);
  // Expanding bias from (out_channels) to (out_channels, output_height, output_width)
  auto bias_expanded = std::make_unique<float[]>(out_channels * output_height * output_width);

  for (int i = 0; i < out_channels; ++i) {
    for (int j = 0; j < output_height * output_width; ++j) {
      bias_expanded[i * output_height * output_width + j] = bias[i];
    }
  }

  // Add bias
  _add(output, bias_expanded.get(), out_channels * output_height * output_width);
}
