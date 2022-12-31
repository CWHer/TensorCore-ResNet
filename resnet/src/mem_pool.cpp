/** @file mem_pool.cpp
*/
#if not DISABLE_MEM_POOL
#include "common.h"
#include "mem_pool.h"

using namespace Impl;
using namespace std;

// // Total memory size: 2GB
// constexpr size_t max_mempool_size = 1024;
// // prealloc size: 2MB
// constexpr size_t max_prealloc_size = 2 * 1024 * 1024;
// constexpr size_t gap_size = 1024;
// constexpr size_t total_size = max_mempool_size * (max_prealloc_size + gap_size);

// static deque<void *> mem_pool{};
// static unordered_map<void *, size_t> mem_size{};
// static void *map_base = nullptr;

// static int malloc_called = 0;
// static int free_called = 0;

static std::unordered_map<uint64_t, std::deque<void *>> mem_cache;

namespace Impl {

void init_mem_pool() {
  // checkCudaErrors(cudaMalloc(&map_base, total_size));
  // if(map_base == nullptr) {
  //   throw std::runtime_error("memory pool init failed");
  // }
  // void *ptr = map_base;
  // for (int i = 0; i < max_mempool_size; i++) {
  //   mem_pool.push_back(ptr);
  //   mem_size[ptr] = -1;
  //   ptr = (void *) ((char *) ptr + max_prealloc_size + gap_size);
  // }

  // malloc_called = 0;
  // free_called = 0;
}

void deinit_mem_pool() {
  // cudaFree(map_base);
  // mem_pool.clear();

  // if (malloc_called != free_called) {
  //   std::cout << "Memory pool might have memory leak: " << malloc_called << ":" << free_called << std::endl;
  // }
  for (auto &kv : mem_cache)
    for (auto &ptr : kv.second)
      checkCudaErrors(cudaFree(ptr));
}

cudaError_t cudaPooledMalloc(void **devPtr, size_t size) {
  // if (map_base == nullptr) {
  //   cout << "WARNING: Memory pool not initialized" << endl;
  //   init_mem_pool();
  // }

  // if (size > max_prealloc_size) {
  //   return cudaMalloc(devPtr, size);
  // }
  // if (mem_pool.empty()) {
  //   return cudaMalloc(devPtr, size);
  // }
  // *devPtr = mem_pool.front();
  // mem_pool.pop_front();

  // mem_size[*devPtr] = size;

  // malloc_called++;

  // return cudaSuccess;

  if (mem_cache.count(size) > 0 && !mem_cache[size].empty())
  {
    *devPtr = mem_cache[size].front();
    mem_cache[size].pop_front();
    return cudaSuccess;
  }
  else
  {
    return cudaMalloc(devPtr, size);
  }
}

cudaError_t cudaPooledFree(void *devPtr) {
  // if (mem_size.find(devPtr) != mem_size.end()) {
  //   if (mem_size[devPtr] == -1) {
  //     throw std::runtime_error("Memory pool double free");
  //   }
  //   mem_pool.push_back(devPtr);
  //   mem_size[devPtr] = -1;
  //   free_called++;
  //   return cudaSuccess;
  // } else {
  //   return cudaFree(devPtr);
  // }

  size_t size;
  checkCudaErrors(cudaMemGetInfo(nullptr, &size));
  mem_cache[size].push_back(devPtr);
  return cudaSuccess;
}

}
#endif

