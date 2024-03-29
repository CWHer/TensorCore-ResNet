/** @file fp16_array.cu
*/
#include "functional/gemm.hpp"
#include "functional/macros.hpp"

static __global__ void fp32_to_fp16_kernel(const float_32 *RESTRICT input, float_16 *RESTRICT output, size_t size) {
  CUDA_KERNEL_LOOP(index, size) {
    if (likely(index < size))
      output[index] = __float2half(input[index]);
  }
  __syncthreads();
}

static __global__ void fp16_to_fp32_kernel(const float_16 *RESTRICT input, float_32 *RESTRICT output, size_t size) {
  CUDA_KERNEL_LOOP(index, size) {
    if (likely(index < size))
      output[index] = __half2float(input[index]);
  }
  __syncthreads();
}

float_16 *fp32_array_to_fp16_array(const float_32 *RESTRICT fp32_array, size_t size, Impl::DeviceType device_type) {
  float_16 *fp16_array = nullptr;
  switch (device_type) {
  case Impl::DeviceType::CPU:fp16_array = new float_16[size];
    for (size_t i = 0; i < size; i++) {
      fp16_array[i] = float_16(fp32_array[i]);
    }
    break;
  case Impl::DeviceType::CUDA:Impl::cudaPooledMalloc(&fp16_array, size * sizeof(float_16));
    fp32_to_fp16_kernel<<< KERNEL_LOOP_BLOCKS(size), KERNEL_LOOP_THREADS>>>(fp32_array, fp16_array, size);
    break;
  }
  return fp16_array;
}

float_32 *fp16_array_to_fp32_array(const float_16 *RESTRICT fp16_array, size_t size, Impl::DeviceType device_type) {
  float_32 *fp32_array = nullptr;
  switch (device_type) {
  case Impl::DeviceType::CPU:fp32_array = new float_32[size];
    for (size_t i = 0; i < size; i++) {
      fp32_array[i] = __half2float(fp16_array[i]);
    }
    break;
  case Impl::DeviceType::CUDA:Impl::cudaPooledMalloc(&fp32_array, size * sizeof(float_32));
    fp16_to_fp32_kernel<<<KERNEL_LOOP_BLOCKS(size), KERNEL_LOOP_THREADS>>>(fp16_array, fp32_array, size);
    break;

  }
  return fp32_array;
}
