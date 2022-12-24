/**
 * @file test/test_batchnorm.cpp
 * @brief Test bn functionality
 */

#include <gtest/gtest.h>
#include "batchnorm.hpp"

TEST(batchnorm, bn)
{
    // load module
    BatchNorm2d bn(3);
    bn.loadWeights("test_batchnorm_bn");
    bn.to(DeviceType::CUDA);

    // load input
    Tensor x;
    x.load("test_batchnorm_x.bin");
    x.to(DeviceType::CUDA);

    bn.forward(x);

    // load output & check correctness
    Tensor y;
    y.load("test_batchnorm_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();
    const int N_TEST = 5000;
    for (int i = 0; i < N_TEST; i++)
        EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}