/** @file mem_pool.cpp
*/
#if not DISABLE_MEM_POOL
#include <cuda.h>
#include "common.h"
#include "mem_pool.h"

using namespace Impl;
using namespace std;

static std::unordered_map<uint64_t, std::deque<void *>> mem_cache;
static std::unordered_map<void*, uint64_t> mem_size;

namespace Impl {

void init_mem_pool() {}

void deinit_mem_pool() {
  for (auto &kv : mem_cache)
    for (auto &ptr : kv.second)
      checkCudaErrors(cudaFree(ptr));
}

cudaError_t cudaPooledMalloc(void **devPtr, size_t size) {
  if (mem_cache.count(size) > 0 && !mem_cache[size].empty())
  {
    *devPtr = mem_cache[size].front();
    mem_cache[size].pop_front();
    return cudaSuccess;
  }
  else
  {
    auto ret = cudaMalloc(devPtr, size);
    if (ret == cudaSuccess)
    {
      mem_size[*devPtr] = size;
    }
    return ret;
  }
}

cudaError_t cudaPooledFree(void *devPtr) {
  auto size = mem_size[devPtr];
  mem_cache[size].push_back(devPtr);
  return cudaSuccess;
}

}
#endif

