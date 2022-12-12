#include "gtest/gtest.h"
#include "tensor.hpp"

TEST(tensor, basic)
{
    Tensor x({2, 3, 4});

    EXPECT_EQ(x.sizes()[0], 2);
    EXPECT_EQ(x.sizes()[1], 3);
    EXPECT_EQ(x.sizes()[2], 4);
    EXPECT_EQ(x.totalSize(), 24);

    // FIXME: this is fake as the value is not initialized
    auto value = x.index({1, 2, 3});
    x.view({2, 12});
    EXPECT_EQ(x.index({1, 6}), value);
    x.view({24});
    EXPECT_EQ(x.index({18}), value);

    Tensor y;
    EXPECT_EQ(y.empty(), true);
}

TEST(tensor, load)
{
    // NOTE: pretrained resnet18 model
    std::string filename = "conv1_weight.bin";
    Tensor x;
    x.load(filename);

    EXPECT_EQ(x.sizes()[0], 64);
    EXPECT_EQ(x.sizes()[1], 3);

    const float value1 = -1.0419e-02;
    const float value2 = -6.1356e-03;
    const float value3 = -1.8098e-03;
    const float eps = 1e-6;
    EXPECT_NEAR(x.index({0, 0, 0, 0}), value1, eps);
    EXPECT_NEAR(x.index({0, 0, 0, 1}), value2, eps);
    EXPECT_NEAR(x.index({0, 0, 0, 2}), value3, eps);
}

TEST(tensor, device)
{
    // NOTE: pretrained resnet18 model
    std::string filename = "conv1_weight.bin";
    Tensor x;
    x.load(filename);
    EXPECT_EQ(x.getDevice(), DeviceType::CPU);

    x.to(DeviceType::CUDA);
    EXPECT_EQ(x.getDevice(), DeviceType::CUDA);

    Tensor y = x.clone();
    EXPECT_EQ(y.getDevice(), DeviceType::CUDA);
    const float value1 = -1.0419e-02;
    const float value2 = -6.1356e-03;
    const float value3 = -1.8098e-03;
    const float eps = 1e-3;
    EXPECT_NEAR(x.index({0, 0, 0, 0}), value1, eps);
    EXPECT_NEAR(y.index({0, 0, 0, 1}), value2, eps);
    EXPECT_NEAR(y.index({0, 0, 0, 2}), value3, eps);
}
