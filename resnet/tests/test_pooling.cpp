/**
 * @file test/test_pooling.cpp
 * @brief Test pooling functionality
 */

#include <gtest/gtest.h>
#include "pooling.hpp"

#ifdef TEST_DATA_ROOT
TEST(pooling, max_pool)
{
    std::string file_dir = TEST_DATA_ROOT;

    // load module
    MaxPool2d max_pool(3, 2, 1);

    // load input
    Tensor x;
    x.load(file_dir + "/test_maxpool_x.bin");
    x.to(DeviceType::CUDA);

    x = max_pool.forward(x);

    // load output & check correctness
    Tensor y;
    y.load(file_dir + "/test_maxpool_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();
    for (int i = 0; i < x.totalSize(); i++)
        EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}

TEST(pooling, avg_pool)
{
    std::string file_dir = TEST_DATA_ROOT;

    // load module
    AvgPool2d avg_pool(5, 2, 2);

    // load input
    Tensor x;
    x.load(file_dir + "/test_avgpool_x.bin");
    x.to(DeviceType::CUDA);

    x = avg_pool.forward(x);

    // load output & check correctness
    Tensor y;
    y.load(file_dir + "/test_avgpool_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();
    for (int i = 0; i < x.totalSize(); i++)
        EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}
#endif