/**
 * @file test/test_batchnorm.cpp
 * @brief Test bn functionality
 */

#include <gtest/gtest.h>
#include "batchnorm.cu"

TEST(batchnorm, bn)
{
    const float eps = 1e-5;

    // load data
    std::string weight_fn = "test_batchnorm_bn_weight.bin";
    std::string bias_fn = "test_batchnorm_bn_bias.bin";
    std::string mean_fn = "test_batchnorm_bn_running_mean.bin";
    std::string var_fn = "test_batchnorm_bn_running_var.bin";
    Tensor weight;
    weight.load(weight_fn);
    Tensor bias;
    bias.load(bias_fn);
    Tensor running_mean;
    running_mean.load(mean_fn);
    Tensor running_var;
    running_var.load(var_fn);

    // execution on gpu
    std::string input_fn = "test_batchnorm_x.bin";
    Tensor x;
    x.load(input_fn);
    BatchNorm2d bn2d(weight, bias, running_mean, running_var, eps);
    bn2d.forward(x);

    // check correctness
    std::string output_fn = "test_batchnorm_y.bin";
    Tensor y;
    y.load(output_fn);
    float *res = x.data_ptr();
    const unsigned int feature_sizes = x.sizes()[1];
    float *ref = y.data_ptr();
    for (int i = 0; i < feature_sizes; i++)
    {
        EXPECT_NEAR(res[i], ref[i], 1e-3);
    }
}