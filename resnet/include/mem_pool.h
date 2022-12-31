/** @file mem_pool.h
*/#ifndef TENSORCORE_RESNET_COMMON_MEM_POOL_H
#define TENSORCORE_RESNET_COMMON_MEM_POOL_H

#include <cuda_runtime.h>

#if not DISABLE_MEM_POOL

namespace Impl {
void init_mem_pool();
void deinit_mem_pool();

cudaError_t cudaPooledMalloc(void **devPtr, size_t size);
template<class T> inline cudaError_t cudaPooledMalloc(T **ptr, size_t size) {
  return cudaPooledMalloc((void **) (void *) ptr, size);
}
cudaError_t cudaPooledFree(void *devPtr);
}

#else

namespace Impl {
inline void init_mem_pool() {}
inline void deinit_mem_pool() {}

inline cudaError_t cudaPooledMalloc(void **devPtr, size_t size) {return ::cudaMalloc(devPtr, size);}
template<class T> inline cudaError_t cudaPooledMalloc(T **ptr, size_t size) {
  return ::cudaMalloc((void **) (void *) ptr, size);
}
inline cudaError_t cudaPooledFree(void *devPtr) {return ::cudaFree(devPtr);}

}

#endif

#endif //TENSORCORE_RESNET_COMMON_MEM_POOL_H
