#include "gtest/gtest.h"
#include "tensor.hpp"

using namespace Impl;

TEST(tensor, basic) {
  float data_[24] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
  };
  auto data = new float[24];
  std::copy(data_, data_ + 24, data);
  Tensor x({2, 3, 4}, Impl::DeviceType::CPU, data);

  EXPECT_EQ(x.sizes()[0], 2);
  EXPECT_EQ(x.sizes()[1], 3);
  EXPECT_EQ(x.sizes()[2], 4);
  EXPECT_EQ(x.totalSize(), 24);

  auto value = x.index({1, 2, 3});
  x.reshape({2, 12});
  EXPECT_EQ(x.index({1, 11}), value);
  x.reshape({24});
  EXPECT_EQ(x.index({23}), value);

  Tensor y;
  EXPECT_EQ(y.empty(), true);
}

#if WITH_DATA
#ifdef RESNET18_ROOT
TEST(tensor, load) {
  // NOTE: pretrained resnet18 model
  std::string filename = RESNET18_ROOT;
  filename += "/resnet18_conv1_weight.bin";
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
#endif
#endif

#if WITH_DATA
#ifdef RESNET18_ROOT
TEST(tensor, device) {
  // Test if device loaded as same as host loads
  std::string filename = RESNET18_ROOT;
  filename += "/resnet18_conv1_weight.bin";
  Tensor x;
  x.load(filename);
  EXPECT_EQ(x.getDevice(), Impl::DeviceType::CPU);

  x.to(Impl::DeviceType::CUDA);
  EXPECT_EQ(x.getDevice(), Impl::DeviceType::CUDA);

  Tensor y = x.clone();
  EXPECT_EQ(y.getDevice(), Impl::DeviceType::CUDA);

  // flatten x and y
  x.reshape({(int) x.totalSize()});
  y.reshape({(int) y.totalSize()});

  // total sizes should be the same
  EXPECT_EQ(x.totalSize(), y.totalSize());

  const float eps = 1e-3;
  for (auto idx = 0; idx < x.totalSize(); ++idx) {
    EXPECT_NEAR(x.index({0, 0, 0, idx}), y.index({0, 0, 0, idx}), eps);
  }
}
#endif
#endif
