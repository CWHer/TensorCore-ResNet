/** @file im2col.cu
*/

#include <memory>
#include "common.h"
#include "functional/conv2d.hpp"
#include "functional/macros.h"
#include "mem_pool.h"

using namespace Impl;

/* @brief im2col result shape
 */
size_t im2col_result_size(int N, int C, int H, int W, int filter_height, int filter_width, int stride, int padding) {
  int output_height = (H + 2 * padding - filter_height) / stride + 1;
  int output_width = (W + 2 * padding - filter_width) / stride + 1;
  int output_size = output_height * output_width;
  return N * C * filter_height * filter_width * output_size;
}

std::unique_ptr<float_16[]> create_im2col_result_store_host(int N,
                                                            int C,
                                                            int H,
                                                            int W,
                                                            int filter_height,
                                                            int filter_width,
                                                            int stride,
                                                            int padding) {
  // Allocate memory for im2col result.
  auto im2col_size = im2col_result_size(N, C, H, W, filter_height, filter_width, stride, padding);
  return std::make_unique<float_16[]>(im2col_size);
}

std::unique_ptr<float_16[], decltype(&cudaPooledFree)> create_im2col_result_store_device(int N,
                                                                                   int C,
                                                                                   int H,
                                                                                   int W,
                                                                                   int filter_height,
                                                                                   int filter_width,
                                                                                   int stride,
                                                                                   int padding) {
  auto im2col_size = im2col_result_size(N, C, H, W, filter_height, filter_width, stride, padding);
  float_16 *ptr;
  Impl::cudaPooledMalloc(&ptr, im2col_size * sizeof(float_16));
  return std::unique_ptr<float_16[], decltype(&cudaPooledFree)>(ptr, &cudaPooledFree);
}

__global__ static void im2col_cuda_kernel(const float *input,
                                          float_16 *output,
                                          int N,
                                          int C,
                                          int H,
                                          int W,
                                          int kernel_size,
                                          int stride,
                                          int padding) {
  int output_height = (H + 2 * padding - kernel_size) / stride + 1;
  int output_width = (W + 2 * padding - kernel_size) / stride + 1;
  int output_size = output_height * output_width;

  int filter_size = kernel_size * kernel_size;
  int input_channel_size = H * W;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < output_size * filter_size * C * N; i += blockDim.x * gridDim.x)
  {
    // FIXME: unify the order with im2col_naive
    // order: output_h, output_w, filter_h, filter_w, C
    int cur_index = i;
    int cur_n = cur_index % N;
    cur_index /= N;
    int cur_c = cur_index % C;
    cur_index /= C;
    int filter_w = cur_index % kernel_size;
    cur_index /= kernel_size;
    int filter_h = cur_index % kernel_size;
    cur_index /= kernel_size;
    int output_w = cur_index % output_width;
    cur_index /= output_width;
    int output_h = cur_index;

    int index_h = output_h * stride + filter_h - padding;
    int index_w = output_w * stride + filter_w - padding;
    int input_index = (cur_n * C + cur_c) * input_channel_size + index_h * W + index_w;
    // clang-format off
    if (input_index > N * C * H * W) continue;
    // clang-format on

    int output_index = (cur_n * C + cur_c) * filter_size + filter_h * kernel_size + filter_w;
    int output_offset = output_h * output_width + output_w;
    output[output_index * output_size + output_offset] = index_h >= 0 && index_h < H &&
        index_w >= 0 && index_w < W &&
        cur_c < C && cur_n < N
                                                         ? __float2half(input[input_index])
                                                         : float_16(0);
  }
}

/**
 * @copydoc im2col
 */
static void im2col_device_memory(const float *input,
                                 float_16 *output,
                                 int N,
                                 int C,
                                 int H,
                                 int W,
                                 int filter_height,
                                 int filter_width,
                                 int stride,
                                 int padding) {
  auto single_result_size = im2col_result_size(1, C, H, W, filter_height, filter_width, stride, padding);
  // Launch CUDA kernel
  constexpr unsigned long minibatch_size = 2;
  constexpr int stream_num = 8;
  unsigned long minibatches = (N + minibatch_size - 1) / minibatch_size;

  cudaStream_t stream[stream_num];
  for (auto & i : stream) {
    checkCudaErrors(cudaStreamCreate(&i));
  }

  for (unsigned long i = 0; i < minibatches; i++) {
    unsigned long curr_minibatch_size = std::min(minibatch_size, (unsigned long) N - i * minibatch_size);
    unsigned long curr_result_size = curr_minibatch_size * single_result_size;

    im2col_cuda_kernel<<<KERNEL_LOOP_BLOCKS(curr_result_size), KERNEL_LOOP_THREADS, 0, stream[i % stream_num]>>>(
        input + i * minibatch_size * C * H * W,
        output + i * minibatch_size * single_result_size,
        (int) curr_minibatch_size,
        C,
        H,
        W,
        filter_height,
        stride,
        padding);
    checkCudaErrors(cudaPeekAtLastError());
  }

  for (auto & i : stream) {
    cudaStreamSynchronize(i);
    cudaStreamDestroy(i);
  }
}

/**
 * @copydoc im2col
 */
static void im2col_host_memory(const float *input,
                               float_16 *output,
                               int N,
                               int C,
                               int H,
                               int W,
                               int filter_height,
                               int filter_width,
                               int stride,
                               int padding) {
  // Copy input to device
  float *input_device;
  cudaMalloc(&input_device, sizeof(float) * N * C * H * W);
  cudaMemcpy(input_device, input, sizeof(float) * N * C * H * W, cudaMemcpyHostToDevice);

  auto result_size = im2col_result_size(N, C, H, W, filter_height, filter_width, stride, padding);
  float_16 *output_device;
  cudaMalloc(&output_device, sizeof(float_16) * result_size);

  im2col_device_memory(input_device, output_device, N, C, H, W, filter_height, filter_width, stride, padding);

  // Copy result back to host
  cudaMemcpy(output, output_device, sizeof(float_16) * result_size, cudaMemcpyDeviceToHost);
  // Free
  cudaFree(input_device);
  cudaFree(output_device);
}

/**
 * @brief im2col, performs as the matlab function and at::Tensor function
 *
 * @param input float of shape (N, C, H, W)
 * @param output Output should be shaped as:
 * (N, C * filter_height * filter_width, output_height * output_width)
 *
 * where output_height = (H + 2 * padding - filter_height) / stride + 1
 * and output_width = (W + 2 * padding - filter_width) / stride + 1
 *
 * Data are arranged per channel of columns, this results in factor C.
 */
void im2col(const float *input,
            float_16 *output,
            int N,
            int C,
            int H,
            int W,
            int filter_height,
            int filter_width,
            int stride,
            int padding,
            Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU: return im2col_host_memory(input,
                                                        output,
                                                        N,
                                                        C,
                                                        H,
                                                        W,
                                                        filter_height,
                                                        filter_width,
                                                        stride,
                                                        padding);

  case Impl::DeviceType::CUDA: return im2col_device_memory(input,
                                                           output,
                                                           N,
                                                           C,
                                                           H,
                                                           W,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           padding);
  }

}

