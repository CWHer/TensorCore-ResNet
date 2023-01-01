/**
 * @file test/test_pooling.cpp
 * @brief Test pooling functionality
 */

#include <gtest/gtest.h>
#include "pooling.hpp"
#include <string>

using namespace Impl;
using namespace std;

#if WITH_DATA
#ifdef TEST_DATA_ROOT
TEST(pooling, max_pool) {
  // load_single module
  MaxPool2d max_pool(3, 2, 1);

  // load_single input
  Tensor x;
  string filename = TEST_DATA_ROOT;
  filename += "/test_maxpool_x.bin";
  x.load(filename);
  x.to(DeviceType::CUDA);

  x = max_pool.forward(x);

  // load_single output & check correctness
  Tensor y;
  filename = TEST_DATA_ROOT;
  filename += "/test_maxpool_y.bin";
  y.load(filename);

  x.to(DeviceType::CPU);
  float *naive_res = x.data_ptr();
  float *std_res = y.data_ptr();
  const int N_TEST = 5000;
  for (int i = 0; i < N_TEST; i++)
    EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}
#endif
#endif

#if WITH_DATA
#ifdef TEST_DATA_ROOT
TEST(pooling, avg_pool) {
  // load_single module
  AvgPool2d avg_pool(5, 2, 2);

  // load_single input
  Tensor x;
  string filename = TEST_DATA_ROOT;
  filename += "/test_avgpool_x.bin";
  x.load(filename);
  x.to(DeviceType::CUDA);

  x = avg_pool.forward(x);

  // load_single output & check correctness
  Tensor y;
  filename = TEST_DATA_ROOT;
  filename += "/test_avgpool_y.bin";
  y.load(filename);

  x.to(DeviceType::CPU);
  float *naive_res = x.data_ptr();
  float *std_res = y.data_ptr();
  const int N_TEST = 500;
  for (int i = 0; i < N_TEST; i++)
    EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}
#endif
#endif
