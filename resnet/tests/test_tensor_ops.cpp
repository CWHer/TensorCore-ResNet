#include "gtest/gtest.h"
#include "ops.hpp"

TEST(tensor_ops, cpu_ops)
{

    float data_[24] = {1, 2, 3, 4, 4, 3, 2, 1, 5, 5, 5, 5,
                       1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0, 0};
    float *data = new float[24];
    std::copy(data_, data_ + 24, data);
    Tensor x({2, 3, 4}, DeviceType::CPU, data);

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
    Tensor label = Tensor({2, 3}, DeviceType::CPU, label_data);

    EXPECT_NEAR(TensorOps::sum_equal(z, label), 6, eps);
}