#include <gtest/gtest.h>
#include "half.hpp"

TEST(half, conversion) {
  Sim::f32 f;
  Sim::f16 h;

  const float eps = 1e-4f;

  // normalized value
  f = 10.0f;
  h = Sim::__float2half(f);
  EXPECT_NEAR(f, Sim::__half2float(h), f * eps);

// HACK: when optimizing with -ffast-math, inf, nan, etc.
// checks like isinf and isnan will not be available.
#if not defined (__FAST_MATH__)
  // normalized value: overflow
  f = std::numeric_limits<float>::max();
  h = Sim::__float2half(f);
  EXPECT_TRUE(std::isinf(Sim::__half2float(h)));
#else
#pragma message "fast math is enabled, overflow test will not be valid."
#endif

  // normalized value: underflow
  f = std::numeric_limits<float>::min();
  h = Sim::__float2half(f);
  EXPECT_EQ(0, Sim::__half2float(h));

  // normalized value: underflow
  f = 1.0f / (1 << 20);
  h = Sim::__float2half(f);
  EXPECT_NEAR(f, Sim::__half2float(h), f * eps);

  // denormalized value
  f = std::numeric_limits<float>::denorm_min();
  h = Sim::__float2half(f);
  EXPECT_EQ(0, Sim::__half2float(h));

#if not defined (__FAST_MATH__)
  // infinity
  f = std::numeric_limits<float>::infinity();
  h = Sim::__float2half(f);
  EXPECT_TRUE(std::isinf(Sim::__half2float(h)));

  // NaN
  f = std::numeric_limits<float>::quiet_NaN();
  h = Sim::__float2half(f);
  EXPECT_TRUE(std::isnan(Sim::__half2float(h)));
#endif
}

TEST(half, operators) {
  Sim::f16 h1, h2;

  const float eps = 1e-10f;

  h1 = Sim::__float2half(1.0f);
  h2 = Sim::__float2half(20000.0f);

  EXPECT_NEAR(20000.0f, h1 * h2, eps);
}
