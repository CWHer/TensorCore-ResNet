/** @file im2col.cu
*/

#include <memory>
#include "common.h"
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
 * (C * filter_height * filter_width, output_height * output_width)
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
  int output_ldm = output_size;
  int input_channel_size = H * W;

  for (int filter_h = 0; filter_h < filter_height; ++filter_h) {
    for (int filter_w = 0; filter_w < filter_width; ++filter_w) {
      for (int output_h = 0; output_h < output_height; ++output_h) {
        for (int output_w = 0; output_w < output_width; ++output_w) {
          for (int c = 0; c < C; ++c) {
            int index_h = output_h * stride + filter_h - padding;
            int index_w = output_w * stride + filter_w - padding;
            int input_index = c * input_channel_size + index_h * W + index_w;
            int output_index = c * filter_size + filter_h * filter_width + filter_w;
            int output_offset = output_h * output_width + output_w;
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

  int output_ldm = output_size;

  int input_channel_size = H * W;
  CUDA_KERNEL_LOOP(loop_idx, output_size * filter_size * C) {
    // FIXME: unify the order with im2col_naive
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
    int output_index = c * filter_size + filter_h * filter_width + filter_w;
    int output_offset = output_h * output_width + output_w;
    if (index_h >= 0 && index_h < H && index_w >= 0 && index_w < W) {
      output[output_index * output_ldm + output_offset] = input[input_index];
    } else {
      output[output_index * output_ldm + output_offset] = 0;
    }
  }

  __syncthreads();

}

/** @copydoc im2col_naive
 *
 * @brief im2col, performs as the matlab function, using CUDA
 */
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
