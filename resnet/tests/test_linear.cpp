#include <gtest/gtest.h>
#include "linear.hpp"

#ifdef TEST_DATA_ROOT
TEST(linear, fc)
{
    std::string file_dir = TEST_DATA_ROOT;

    auto in_features = 117;
    auto out_features = 16;

    // load module
    Linear fc(in_features, out_features);
    fc.loadWeights(file_dir + "/test_linear_fc");
    fc.to(DeviceType::CUDA);

    // load input
    Tensor x;
    x.load(file_dir + "/test_linear_x.bin");
    x.to(DeviceType::CUDA);

    x = fc.forward(x);

    // load output & check correctness
    Tensor y;
    y.load(file_dir + "/test_linear_y.bin");

    x.to(DeviceType::CPU);
    float *naive_res = x.data_ptr();
    float *std_res = y.data_ptr();

    double max_diff_ratio = 0;
    double avg_diff_ratio = 0;
    for (int i = 0; i < x.totalSize(); i++)
    {
        double diff_ratio = std_res[i] != 0
                                ? std::fabs(naive_res[i] - std_res[i]) / std::fabs(std_res[i])
                                : 0;
        max_diff_ratio = std::max(max_diff_ratio, diff_ratio);
        avg_diff_ratio += diff_ratio / x.totalSize();
    }
    EXPECT_LE(avg_diff_ratio, 1e-3);
    std::cout << "Max diff ratio: " << max_diff_ratio << std::endl;
    std::cout << "Avg diff ratio: " << avg_diff_ratio << std::endl;
}
#endif