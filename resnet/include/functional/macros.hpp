/** @file macros.h
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_MACROS_H
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_MACROS_H

#include <stdexcept>
#include <cuda_runtime_api.h>
#include "mem_pool.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-macro-parentheses"
#define CUDA_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#pragma clang diagnostic pop

const int KERNEL_LOOP_THREADS = 512;

// CUDA: number of blocks for threads.
inline constexpr int KERNEL_LOOP_BLOCKS(const int N) {
  return (N + KERNEL_LOOP_THREADS - 1) / KERNEL_LOOP_THREADS;
}

inline constexpr int KERNEL_LOOP_BLOCKS(const size_t N) {
  return (int) (N + KERNEL_LOOP_THREADS - 1) / KERNEL_LOOP_THREADS;
}

#if CUDA_MALLOC_ASYNC
template<class T> inline cudaError_t cudaMallocAsyncIfAvailable(T **ptr, size_t size, cudaStream_t stream) {
  return ::cudaMallocAsync((void **) (void *) ptr, size, stream);
}

inline cudaError_t cudaFreeAsyncIfAvailable(void *devPtr, cudaStream_t hStream) {
  return ::cudaFreeAsync(devPtr, hStream);
}

#else
#pragma message "Stream Ordered Memory Allocator is not available with this nvidia driver version.\n"
template<class T> inline cudaError_t cudaMallocAsyncIfAvailable(T **ptr, size_t size, cudaStream_t stream) {
  return Impl::cudaPooledMallocAsync((void **) (void *) ptr, size, stream);
}

inline cudaError_t cudaFreeAsyncIfAvailable(void *devPtr, cudaStream_t hStream) {
  return Impl::cudaPooledFreeAsync(devPtr, hStream);
}
#endif

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_MACROS_H

