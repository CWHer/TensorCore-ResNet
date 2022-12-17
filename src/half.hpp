#pragma once

#include "common.h"

namespace Sim
{
    class Half
    {
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

        friend float operator*(const Half &lhs,
                               const Half &rhs)
        {
            return lhs.toFloat() * rhs.toFloat();
        }

        friend float operator+(const Half &lhs,
                               const Half &rhs)
        {
            return lhs.toFloat() + rhs.toFloat();
        }

    public:
        union FP32
        {
            u32 u;
            f32 f;
            struct
            {
                u32 fraction : 23;
                u32 exponent : 8;
                u32 sign : 1;
            };
        };

        union FP16
        {
            u16 u;
            struct
            {
                u16 fraction : 10;
                u16 exponent : 5;
                u16 sign : 1;
            };
        };

    private:
        FP16 x;

    public:
        Half() = default;

        // NOTE: round-to-even
        void fromFloat(f32 z)
        {
            FP32 y;
            y.f = z, x.u = 0;
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
        }

        f32 toFloat() const
        {
            FP32 y = {0};
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
    };

    using f16 = Half;
}