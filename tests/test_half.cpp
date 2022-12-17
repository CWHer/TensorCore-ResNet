#include <gtest/gtest.h>
#include "half.hpp"

TEST(half, conversion)
{
    Sim::Half::FP32 x;
    Sim::Half h;

    const float eps = 1e-4f;

    // normalized value
    x.f = 10.0f;
    h.fromFloat(x.f);
    EXPECT_NEAR(x.f, h.toFloat(), x.f * eps);

    // normalized value: overflow
    x.f = std::numeric_limits<float>::max();
    h.fromFloat(x.f);
    EXPECT_TRUE(std::isinf(h.toFloat()));

    // normalized value: underflow
    x.f = std::numeric_limits<float>::min();
    h.fromFloat(x.f);
    EXPECT_EQ(0, h.toFloat());

    // normalized value: underflow
    x.f = 1.0f / (1 << 20);
    h.fromFloat(x.f);
    EXPECT_NEAR(x.f, h.toFloat(), x.f * eps);

    // denormalized value
    x.f = std::numeric_limits<float>::denorm_min();
    h.fromFloat(x.f);
    EXPECT_EQ(0, h.toFloat());

    // infinity
    x.f = std::numeric_limits<float>::infinity();
    h.fromFloat(x.f);
    EXPECT_TRUE(std::isinf(h.toFloat()));

    // NaN
    x.f = std::numeric_limits<float>::quiet_NaN();
    h.fromFloat(x.f);
    EXPECT_TRUE(std::isnan(h.toFloat()));
}

TEST(half, operators)
{
    Sim::Half h1, h2;

    const float eps = 1e-10f;

    h1.fromFloat(1.0f);
    h2.fromFloat(20000.0f);

    EXPECT_NEAR(20001.0f, h1 + h2, eps);
    EXPECT_NEAR(20000.0f, h1 * h2, eps);
}