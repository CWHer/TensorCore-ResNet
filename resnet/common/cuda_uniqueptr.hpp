/** @file cuda_uniqueptr.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_CUDA_UNIQUEPTR_HPP
#define TENSORCORE_RESNET_COMMON_CUDA_UNIQUEPTR_HPP

#include <memory>
#include "common.h"

template<class T> class unified_deleter {
public:
  void operator()(T *ptr);

};
template<class T> void unified_deleter<T>::operator()(T *ptr) {
  cudaPointerAttributes attributes{};
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
  if (err != cudaSuccess) {
    std::cerr << "cudaPointerGetAttributes failed: " << cudaGetErrorString(err) << std::endl;
    return;
  }
  switch (attributes.type) {
  case cudaMemoryTypeHost:delete[] ptr;
    break;
  case cudaMemoryTypeDevice:cudaFree(ptr);
    break;
  default:throw std::runtime_error("unified_deleter: unknown pointer type");
    break;
  }
}

typedef std::unique_ptr<float_16[], unified_deleter<float_16 []>> unified_unique_ptr_float_16;

#endif //TENSORCORE_RESNET_COMMON_CUDA_UNIQUEPTR_HPP
