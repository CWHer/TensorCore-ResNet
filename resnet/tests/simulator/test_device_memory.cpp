#include "gtest/gtest.h"
#include "device_memory.hpp"

TEST(simulator, global_memory) {
  Sim::GlobalMemory mem;
  EXPECT_FALSE(mem.isAllocated(nullptr));

  float *ptr = nullptr;
  mem.cudaMalloc((void **) &ptr, 100 * sizeof(float));
  EXPECT_TRUE(mem.isAllocated(ptr));

  *(ptr + 50) = 1.0f;
  auto val_ptr = ptr + 50;
  EXPECT_NEAR(*val_ptr, 1.0f, 0.001);
  // HACK: only check head address
  EXPECT_FALSE(mem.isAllocated(val_ptr));

#pragma GCC diagnostic push
// If cannot find Wuse-after-free, ignore it
// This is needed with GCC 7.5 as it is not able to find use-after-free
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wuse-after-free"
  mem.cudaFree(ptr);
  EXPECT_FALSE(mem.isAllocated(ptr));
#pragma GCC diagnostic pop
}
