/** @file macros.h
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_MACROS_H
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_MACROS_H

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

const int KERNEL_LOOP_THREADS = 512;

// CUDA: number of blocks for threads.
inline constexpr int KERNEL_LOOP_BLOCKS(const int N) {
  return (N + KERNEL_LOOP_THREADS - 1) / KERNEL_LOOP_THREADS;
}

inline void CheckCUDAError() {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_MACROS_H
