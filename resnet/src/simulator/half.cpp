#include "half.h"

namespace Sim
{

    // NOTE: round-to-even
    f16 __float2half(f32 z)
    {
        FP32 y = {z};
        f16 x = {0};
        x.sign = y.sign;

        // normalized value
        if (y.exponent != 0 && y.exponent != 0b11111111)
        {
            i32 new_exp = y.exponent - 127 + 15;
            if (new_exp > 0b11111)
            {
                // 1. overflow
                x.exponent = 0b11111;
                x.fraction = 0;
            }
            else if (new_exp <= 0)
            {
                // 2. underflow
                if (14 - new_exp <= 24)
                {
                    // denormalized value
                    u32 fraction = y.fraction | 0x800000;
                    u32 shift = 14 - new_exp;
                    x.fraction = fraction >> shift;
                    x.exponent = 0;

                    u32 low_fraction = fraction & ((1 << shift) - 1);
                    u32 halfway = 1 << (shift - 1);

                    // if above halfway point or unrounded result is odd
                    // NOTE: might overflow to inf, this is OK
                    if (low_fraction & halfway) // check for rounding
                        x.u += low_fraction > halfway || (x.fraction & 1);
                }
            }
            else
            {
                x.exponent = new_exp;
                x.fraction = y.fraction >> (23 - 10);
                // if above halfway point or unrounded result is odd
                // NOTE: might overflow to inf, this is OK
                if (y.fraction & 0x1000) // check for rounding
                    x.u += ((y.fraction & 0x1fff) > 0x1000) || (x.fraction & 1);
            }
        }

        // denormalized value
        else if (y.exponent == 0)
        {
            // NOTE: this value will underflow
        }

        else if (y.exponent == 0b11111111)
        {
            x.exponent = 0b11111;
            x.fraction = y.fraction == 0
                             ? 0      // 1. infinity
                             : 0x3ff; // 2. NaN
        }

        return x;
    }

    f32 __half2float(f16 x)
    {
        FP32 y(0);
        y.sign = x.sign;

        // normalized value
        if (x.exponent != 0 && x.exponent != 0b11111)
        {
            y.exponent = x.exponent - 15 + 127;
            y.fraction = x.fraction << (23 - 10);
        }

        // denormalized value
        else if (x.exponent == 0)
        {
            // NOTE: re-normalize
            if (x.fraction > 0)
            {
                u32 fraction = x.fraction, shift = 0;
                for (;; fraction <<= 1, ++shift)
                    if (fraction & 0x400)
                        break;

                y.exponent = 1 - 15 + 127 - shift;
                y.fraction = (fraction & 0x3ff) << (23 - 10);
            }
        }

        else if (x.exponent == 0b11111)
        {
            y.exponent = 0b11111111;
            y.fraction = x.fraction == 0
                             ? 0         // 1. infinity
                             : 0x7fffff; // 2. NaN
        }

        return y.f;
    }

    f32 __unsigned2float(u32 x)
    {
        FP32 y;
        y.u = x;
        return y.f;
    }

    f32 operator*(const f16 &lhs, const f16 &rhs)
    {
        return __half2float(lhs) * __half2float(rhs);
    }

}