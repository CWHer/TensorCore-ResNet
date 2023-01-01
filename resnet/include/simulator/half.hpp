#pragma once

#include "sim_common.hpp"

namespace Sim {

// Reference: https://gist.github.com/rygorous/2156668

// NOTE: from msb to lsb
//  f32 -> 1 bit sign + 8 bits exponent + 23 bits fraction
//  f16 -> 1 bit sign + 5 bits exponent + 10 bits fraction

// NOTE: B = 2 ^ (e - 1) - 1
//          15 = 2 ^ (5 - 1) - 1
//          127 = 2 ^ (8 - 1) - 1
//  1. normalized value:    1.ffff * 2 ^ (e - B)
//  2. denormalized value:  0.ffff * 2 ^ (1 - B)
//  3. special values:      infinity, NaN

union FP32 {
  u32 u;
  f32 f;
  struct {
    u32 fraction: 23;
    u32 exponent: 8;
    u32 sign: 1;
  };

  FP32() = default;
  explicit FP32(f32 f) : f(f) {}
};

union FP16 {
  u16 u;
  struct {
    u16 fraction: 10;
    u16 exponent: 5;
    u16 sign: 1;
  };

  FP16() = default;
  [[maybe_unused]] explicit FP16(u16 u) : u(u) {}
};

using f16 = FP16;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"
// NOTE: round-to-even
f16 __float2half(f32 z);

f32 __half2float(f16 x);

f32 __unsigned2float(u32 x);

#pragma clang diagnostic pop

f32 operator*(const f16 &lhs, const f16 &rhs);

}
