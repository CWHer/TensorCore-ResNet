/**
 * @file test/test_batchnorm.cpp
 * @brief Test bn functionality
 */

#include <gtest/gtest.h>
#include "batchnorm.hpp"

using namespace Impl;

#if WITH_DATA
#ifdef TEST_DATA_ROOT
TEST(batchnorm, bn)
{
    // load_single module
    BatchNorm2d bn(3);
    std::string filename = TEST_DATA_ROOT;
    filename += "/test_batchnorm_bn";
    bn.loadWeights(filename);
    bn.to(DeviceType::CUDA);

    // load_single input
    Tensor x;
    filename = TEST_DATA_ROOT;
    filename += "/test_batchnorm_x.bin";
    x.load(filename);
    x.to(DeviceType::CUDA);

    x = bn.forward(x);

    // load_single output & check correctness
    Tensor y;
    filename = TEST_DATA_ROOT;
    filename += "/test_batchnorm_y.bin";
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
