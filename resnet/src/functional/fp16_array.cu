/** @file fp16_array.cu
*/
#include "functional/gemm.hpp"
#include "functional/macros.h"

static void check_cuda_error() {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

static __global__ void fp32_to_fp16_kernel(const float_32 *input, float_16 *output, size_t size) {
  CUDA_KERNEL_LOOP(index, size) {
    output[index] = __float2half(input[index]);
  }
  __syncthreads();
}

static __global__ void fp16_to_fp32_kernel(const float_16 *input, float_32 *output, size_t size) {
  CUDA_KERNEL_LOOP(index, size) {
    output[index] = __half2float(input[index]);
  }
  __syncthreads();
}


float_16 *fp32_array_to_fp16_array(const float_32 *fp32_array, size_t size, Impl::DeviceType device_type) {
  float_16 *fp16_array;
  switch (device_type) {
  case Impl::DeviceType::CPU:fp16_array = new float_16[size];
    for (int i = 0; i < size; i++) {
      fp16_array[i] = float_16(fp32_array[i]);
    }
    break;
  case Impl::DeviceType::CUDA:cudaMalloc(&fp16_array, size * sizeof(float_16));
    fp32_to_fp16_kernel<<< KERNEL_LOOP_BLOCKS(size), KERNEL_LOOP_THREADS>>>(fp32_array, fp16_array, size);
    check_cuda_error();
    break;
  }
  return fp16_array;
}

float_32 *fp16_array_to_fp32_array(const float_16 *fp16_array, size_t size, Impl::DeviceType device_type) {
  float_32 *fp32_array;
  switch (device_type) {
  case Impl::DeviceType::CPU:fp32_array = new float_32[size];
    for (int i = 0; i < size; i++) {
      fp32_array[i] = __half2float(fp16_array[i]);
    }
    break;
  case Impl::DeviceType::CUDA:cudaMalloc(&fp32_array, size * sizeof(float_32));
    fp16_to_fp32_kernel<<<KERNEL_LOOP_BLOCKS(size), KERNEL_LOOP_THREADS>>>(fp16_array, fp32_array, size);
    check_cuda_error();

    break;

  }
  return fp32_array;
}
