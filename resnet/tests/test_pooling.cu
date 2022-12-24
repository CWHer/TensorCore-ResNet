/**
 * @file test/test_pooling.cpp
 * @brief Test pooling functionality
 */

#include <gtest/gtest.h>
#include "pooling.hpp"

TEST(pooling, max_pool)
{
    // load module
    MaxPool2d max_pool(3, 2, 1);

    // load input
    Tensor x;
    x.load("test_maxpool_x.bin");
    x.to(DeviceType::CUDA);

    x = max_pool.forward(x);

    // load output & check correctness
    Tensor y;
    y.load("test_maxpool_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();
    const int N_TEST = 5000;
    for (int i = 0; i < N_TEST; i++)
        EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}

TEST(pooling, avg_pool)
{
    // load module
    AvgPool2d avg_pool(5, 2, 2);

    // load input
    Tensor x;
    x.load("test_avgpool_x.bin");
    x.to(DeviceType::CUDA);

    x = avg_pool.forward(x);

    // load output & check correctness
    Tensor y;
    y.load("test_avgpool_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();
    const int N_TEST = 500;
    for (int i = 0; i < N_TEST; i++)
        EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}