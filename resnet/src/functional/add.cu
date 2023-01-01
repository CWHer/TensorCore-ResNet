/** @file add.cu
*/

#include "functional/macros.hpp"
#include "functional/add.hpp"
#include "common.hpp"

__global__ static void cuda_add_(float *Result, const float *adder, size_t length) {
  CUDA_KERNEL_LOOP(i, length) {
    if (i < length) {
      Result[i] += adder[i];
    }
  }
}

__global__ static void cuda_add(float *Result, const float *adder_a, const float *adder_b, size_t length) {
  CUDA_KERNEL_LOOP(i, length) {
    if (i < length) {
      Result[i] = adder_a[i] + adder_b[i];
    }
  }
}

__global__ static void cuda_relu_(float *Result, size_t length) {
  CUDA_KERNEL_LOOP(i, length) {
    if (i < length) {
      Result[i] = Result[i] > 0.0f ? Result[i] : 0.0f;
    }
  }
}

__global__ static void cuda_add_relu_(float *Result, const float *adder, size_t length) {
  CUDA_KERNEL_LOOP(i, length) {
    if (i < length) {
      Result[i] += adder[i];
      Result[i] = Result[i] > 0.0f ? Result[i] : 0.0f;
    }
  }
}

void add_(float *Result, const float *adder, size_t length, Impl::DeviceType device_type, cudaStream_t stream) {
  switch (device_type) {
  case Impl::DeviceType::CPU: {
    for (size_t i = 0; i < length; ++i) {
      Result[i] += adder[i];
    }
  }
    break;
  case Impl::DeviceType::CUDA:
    cuda_add_<<<KERNEL_LOOP_BLOCKS(length), KERNEL_LOOP_THREADS, 0, stream>>>(Result,
                                                                              adder,
                                                                              length);
    break;
  }
}

void add(float *Result, const float *adder_a, const float *adder_b, int length, Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU: {
    for (int i = 0; i < length; ++i) {
      Result[i] = adder_a[i] + adder_b[i];
    }
  }
    break;
  case Impl::DeviceType::CUDA:
    cuda_add<<<KERNEL_LOOP_BLOCKS(length), KERNEL_LOOP_THREADS>>>(Result,
                                                                  adder_a,
                                                                  adder_b,
                                                                  length);
    break;
  }
}

void relu_(float *Result, size_t length, Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU: {
    for (size_t i = 0; i < length; ++i) {
      Result[i] = Result[i] > 0.0f ? Result[i] : 0.0f;
    }
  }
    break;
  case Impl::DeviceType::CUDA:cuda_relu_<<<KERNEL_LOOP_BLOCKS(length), KERNEL_LOOP_THREADS>>>(Result, length);
    break;
  }
}

void add_relu_(float *Result, const float *adder, size_t length, Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU: {
    for (size_t i = 0; i < length; ++i) {
      Result[i] += adder[i];
      Result[i] = Result[i] > 0.0f ? Result[i] : 0.0f;
    }
  }
    break;
  case Impl::DeviceType::CUDA:
    cuda_add_relu_<<<KERNEL_LOOP_BLOCKS(length), KERNEL_LOOP_THREADS>>>(Result,
                                                                        adder,
                                                                        length);
    break;
  }
}
