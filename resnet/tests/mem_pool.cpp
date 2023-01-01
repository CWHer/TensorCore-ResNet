/** @file mem_pool.cpp
*/
#include <gtest/gtest.h>
#include "mem_pool.h"

using namespace Impl;

TEST(mem_pool, sync) {
  deinit_mem_pool();
  init_mem_pool();
  float *a, *b;
  cudaPooledMalloc(&a, 1024);
  cudaPooledFree(a);
  cudaPooledMalloc(&b, 1024);
  EXPECT_EQ(a, b);
  cudaPooledFree(b);
  deinit_mem_pool();
}

TEST(mem_pool, async) {
  deinit_mem_pool();
  init_mem_pool();
  void *a, *b, *c;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaPooledMallocAsync(&a, 1024, stream);
  cudaPooledFreeAsync(a, stream);
  cudaPooledMalloc(&b, 1024);
  EXPECT_NE(a, b);

  cudaStreamSynchronize(stream);
  cudaCacheCommit(stream);
  cudaPooledMallocAsync(&c, 1024, stream);
  EXPECT_EQ(a, c);

  cudaPooledFree(b);
  cudaPooledFreeAsync(c, stream);

  cudaStreamSynchronize(stream);
  cudaCacheCommit(stream);
  cudaStreamDestroy(stream);

  deinit_mem_pool();
}
