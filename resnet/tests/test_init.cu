/**
 * @file test/test_init.cu
 * @brief Test file for basic initialization, including GTest, CUDA functionality.
 */

#include <gtest/gtest.h>

/** @brief Test if GTest is working */
TEST(init, test_init) {
  EXPECT_EQ(1, 1);
}

/** @brief Test if the test environment have CUDA. */
TEST(init, have_CUDA) {
  int cuda_count;
  if (cudaSuccess != cudaGetDeviceCount(&cuda_count))
    FAIL() << "CUDA not available";
  if (!cuda_count)
    FAIL() << "No CUDA devices found";
}