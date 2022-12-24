#include "gtest/gtest.h"
#include "ops.hpp"
#include <random>

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

using namespace Impl;
using namespace std;

TEST(tensor_ops, argmax_sumequal) {

  float data_[24] = {
      1, 2, 3, 4, 4, 3, 2, 1, 5, 5, 5, 5, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0, 0
  };
  float *data = new float[24];
  std::copy(data_, data_ + 24, data);
  Tensor x({2, 3, 4}, Impl::DeviceType::CPU, data);

  const float eps = 1e-3;
  Tensor y = TensorOps::argmax(x, 1);
  EXPECT_EQ(y.sizes()[0], 2);
  EXPECT_EQ(y.sizes()[1], 4);
  EXPECT_NEAR(y.index({0, 0}), 2, eps);
  EXPECT_NEAR(y.index({0, 3}), 2, eps);
  EXPECT_NEAR(y.index({1, 0}), 1, eps);
  EXPECT_NEAR(y.index({1, 3}), 0, eps);
  Tensor z = TensorOps::argmax(x, 2);
  EXPECT_EQ(z.sizes()[0], 2);
  EXPECT_EQ(z.sizes()[1], 3);
  EXPECT_NEAR(z.index({0, 0}), 3, eps);
  EXPECT_NEAR(z.index({0, 1}), 0, eps);
  EXPECT_NEAR(z.index({0, 2}), 0, eps);

  float label_data_[6] = {3, 0, 0, 3, 0, 0};
  float *label_data = new float[6];
  std::copy(label_data_, label_data_ + 6, label_data);
  Tensor label = Tensor({2, 3}, Impl::DeviceType::CPU, label_data);

  EXPECT_NEAR(TensorOps::sum_equal(z, label), 6, eps);
}

TEST(tensor_ops, relu_) {
  auto t = Tensor({4, 5, 6}, Impl::DeviceType::CPU);

  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < t.totalSize(); i++) {
    t.data_ptr()[i] = matrix_dist(generator);
  }

  auto t_orig = t;

  TensorOps::relu_(t);

  for (int i = 0; i < t.totalSize(); i++) {
    EXPECT_GE(t.data_ptr()[i], 0);
    if (t_orig.data_ptr()[i] < 0) {
      EXPECT_NEAR(t.data_ptr()[i], 0, 1e-6);
    } else {
      EXPECT_NEAR(t.data_ptr()[i], t_orig.data_ptr()[i], 1e-6);
    }
  }

  t = t_orig;
  t.to(Impl::DeviceType::CUDA);

  TensorOps::relu_(t);

  t.to(Impl::DeviceType::CPU);

  for (int i = 0; i < t.totalSize(); i++) {
    EXPECT_GE(t.data_ptr()[i], 0);
    if (t_orig.data_ptr()[i] < 0) {
      EXPECT_NEAR(t.data_ptr()[i], 0, 1e-6);
    } else {
      EXPECT_NEAR(t.data_ptr()[i], t_orig.data_ptr()[i], 1e-6);
    }
  }
}

TEST(tensor_ops, add_) {
  auto t1 = Tensor({4, 5, 6}, Impl::DeviceType::CPU);
  auto t2 = Tensor({4, 5, 6}, Impl::DeviceType::CPU);

  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < t1.totalSize(); i++) {
    t1.data_ptr()[i] = matrix_dist(generator);
    t2.data_ptr()[i] = matrix_dist(generator);
  }

  auto t1_orig = t1;
  auto t2_orig = t2;

  TensorOps::add_(t1, t2);

  for (int i = 0; i < t1.totalSize(); i++) {
    EXPECT_NEAR(t1.data_ptr()[i], t1_orig.data_ptr()[i] + t2_orig.data_ptr()[i], 1e-6);
  }

  t1 = t1_orig;
  t2 = t2_orig;
  t1.to(Impl::DeviceType::CUDA);
  t2.to(Impl::DeviceType::CUDA);

  TensorOps::add_(t1, t2);

  t1.to(Impl::DeviceType::CPU);

  for (int i = 0; i < t1.totalSize(); i++) {
    EXPECT_NEAR(t1.data_ptr()[i], t1_orig.data_ptr()[i] + t2_orig.data_ptr()[i], 1e-6);
  }
}


