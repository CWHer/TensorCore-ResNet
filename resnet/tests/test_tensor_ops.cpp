#include "gtest/gtest.h"
#include "ops.hpp"

TEST(tensor_ops, cpu_ops)
{
    Tensor x({2, 3, 4});

    const float value1 = 1.0;
    const float value2 = 1.0 / 3;
    const float eps = 1e-6;
    TensorOps::add_(x, 1.0);
    EXPECT_NEAR(x.index({0, 0, 0}), value1, eps);
    TensorOps::mul_(x, 1.0 / 3);
    EXPECT_NEAR(x.index({0, 0, 0}), value2, eps);
}