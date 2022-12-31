/**
 * @file test/test_batchnorm.cpp
 * @brief Test bn functionality
 */

#include <gtest/gtest.h>
#include "batchnorm.hpp"

#ifdef TEST_DATA_ROOT
TEST(batchnorm, bn)
{
    std::string file_dir = TEST_DATA_ROOT;

    // load module
    BatchNorm2d bn(3);
    bn.loadWeights(file_dir + "/test_batchnorm_bn");
    bn.to(DeviceType::CUDA);

    // load input
    Tensor x;
    x.load(file_dir + "/test_batchnorm_x.bin");
    x.to(DeviceType::CUDA);

    x = bn.forward(x);

    // load output & check correctness
    Tensor y;
    y.load(file_dir + "/test_batchnorm_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();
    for (int i = 0; i < x.totalSize(); i++)
        EXPECT_NEAR(naive_res[i], std_res[i], 1e-5);
}
#endif