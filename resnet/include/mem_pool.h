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
cudaError_t cudaPooledMallocAsync(void **ptr, size_t size, cudaStream_t stream);
template<class T> inline cudaError_t cudaPooledMallocAsync(T **ptr, size_t size, cudaStream_t stream) {
  return cudaPooledMallocAsync((void **) (void *) ptr, size, stream);
}
cudaError_t cudaPooledFree(void *devPtr);
cudaError_t cudaPooledFreeAsync(void *ptr, cudaStream_t stream);
void cudaCacheCommit(cudaStream_t stream);
}

#else

namespace Impl {
inline void init_mem_pool() {}
inline void deinit_mem_pool() {}

inline cudaError_t cudaPooledMalloc(void **devPtr, size_t size) {return ::cudaMalloc(devPtr, size);}
template<class T> inline cudaError_t cudaPooledMalloc(T **ptr, size_t size) {
  return ::cudaMalloc((void **) (void *) ptr, size);
}
inline cudaError_t cudaPooledMallocAsync(void **ptr, size_t size, cudaStream_t stream) {return ::cudaMallocAsyncIfAvailable(ptr, size, stream);}
template<class T> inline cudaError_t cudaPooledMallocAsync(T **ptr, size_t size, cudaStream_t stream) {
  return cudaPooledMallocAsync((void **) (void *) ptr, size, stream);
}
inline cudaError_t cudaPooledFree(void *devPtr) {return ::cudaFree(devPtr);}
inline cudaError_t cudaPooledFreeAsync(void *ptr, cudaStream_t stream) {return ::cudaFreeAsyncIfAvailable(ptr, stream);}
inline void cudaCacheCommit(cudaStream_t stream){return;}

}

#endif

#endif //TENSORCORE_RESNET_COMMON_MEM_POOL_H
