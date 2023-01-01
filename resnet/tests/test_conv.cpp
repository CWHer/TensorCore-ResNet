#include <gtest/gtest.h>
#include "conv.hpp"

#ifdef TEST_DATA_ROOT
TEST(conv, conv2d)
{
    std::string file_dir = TEST_DATA_ROOT;

    // load module
    Conv2d conv2d(7, 3, 5, 2, 3);
    conv2d.loadWeights(file_dir + "/test_conv_conv2d");
    conv2d.to(DeviceType::CUDA);

    // load input
    Tensor x;
    x.load(file_dir + "/test_conv2d_x.bin");
    x.to(DeviceType::CUDA);

    x = conv2d.forward(x);

    // load output & check correctness
    Tensor y;
    y.load(file_dir + "/test_conv2d_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();
    for (int i = 0; i < x.totalSize(); i++)
    {
        auto diff_ratio = std_res[i] != 0
                              ? std::fabs(naive_res[i] - std_res[i]) / std::fabs(std_res[i])
                              : 0;
        EXPECT_LE(diff_ratio, 1e-3);
        EXPECT_GE(diff_ratio, 0);
    }
}
#endif
