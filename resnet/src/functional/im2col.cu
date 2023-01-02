/** @file im2col.cu
*/

#include <memory>
#include "common.hpp"
#include "functional/conv2d.hpp"
#include "functional/macros.hpp"
#include "mem_pool.h"

using namespace Impl;

/* @brief im2col result shape
 */
constexpr size_t im2col_result_size(int N, int C, int H, int W, int filter_height, int filter_width, int stride, int padding) {
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
  return {ptr, &cudaPooledFree};
}


// borrowed from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/im2col.cuh
__global__ static void im2col_cuda_kernel(const float * RESTRICT input,
                                          float_16 * RESTRICT output,
                                          const int num_kernels,
                                          const int height,
                                          const int width,
                                          const int kernel_size,
                                          const int stride,
                                          const int padding,
                                          const int out_height,
                                          const int out_width) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_kernels; i += blockDim.x * gridDim.x) {
    int w_out = i % out_width;
    int index = i / out_width;
    int h_out = index % out_height;
    int channel_in = index / out_height;
    int channel_out = channel_in * kernel_size * kernel_size;

    int h_in = h_out * stride - padding;
    int w_in = w_out * stride - padding;

    output += (channel_out * out_height + h_out) * out_width + w_out;
    input += (channel_in * height + h_in) * width + w_in;

    for (int i = 0; i < kernel_size; ++i)
      for (int j = 0; j < kernel_size; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *output = (h >= 0 && w >= 0 && h < height && w < width) ? __float2half(input[i * width + j]) : float_16(0);
        output += out_height * out_width;
      }
  }
}

static void im2colStream(const float * RESTRICT input, float_16 * RESTRICT output,
                         int channels, int height, int width,
                         int kernel_size, int stride, int padding,
                         int out_height, int out_width,
                         cudaStream_t stream)
{
  // We are going to launch channels * out_height * out_width kernels, each
  // kernel responsible for copying a single-channel grid.
  int num_kernels = channels * out_height * out_width;
  static const int N_THREADS = 128;
  im2col_cuda_kernel<<<(num_kernels - 1) / N_THREADS + 1, N_THREADS, 0, stream>>>(input,
                                                                                  output,
                                                                                  num_kernels,
                                                                                  height,
                                                                                  width,
                                                                                  kernel_size,
                                                                                  stride,
                                                                                  padding,
                                                                                  out_height,
                                                                                  out_width);
  checkCudaErrors(cudaPeekAtLastError());
}

/**
 * @copydoc im2col
 */
static void im2col_device_memory(const float * RESTRICT input,
                                 float_16 * RESTRICT output,
                                 int N,
                                 int C,
                                 int H,
                                 int W,
                                 int filter_height,
                                 int filter_width,
                                 int stride,
                                 int padding) {
  const int output_height = (H + 2 * padding - filter_height) / stride + 1;
  const int output_width = (W + 2 * padding - filter_width) / stride + 1;
  const int output_size = output_height * output_width;
  const auto result_size_per_batch = C * filter_height * filter_width * output_size;

  constexpr int N_STREAMS = 8;
  cudaStream_t stream[N_STREAMS];
  for (auto &i : stream)
    checkCudaErrors(cudaStreamCreate(&i));

  for (int i = 0; i < N; i++) {
    im2colStream(input + i * C * H * W,
                 output + i * result_size_per_batch,
                 C,
                 H,
                 W,
                 filter_height,
                 stride,
                 padding,
                 output_height,
                 output_width,
                 stream[i % N_STREAMS]);
  }

  for (auto &i : stream) {
    checkCudaErrors(cudaStreamSynchronize(i));
    checkCudaErrors(cudaStreamDestroy(i));
  }
}

/**
 * @copydoc im2col
 */
static void im2col_host_memory(const float * RESTRICT input,
                               float_16 * RESTRICT output,
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
void im2col(const float * RESTRICT input,
            float_16 * RESTRICT output,
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
  case Impl::DeviceType::CPU:
    return im2col_host_memory(input,
                              output,
                              N,
                              C,
                              H,
                              W,
                              filter_height,
                              filter_width,
                              stride,
                              padding);

  case Impl::DeviceType::CUDA:
    return im2col_device_memory(input,
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
