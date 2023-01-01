#pragma once

#include "common.h"
#include "tensor.hpp"

Tensor makeIm2colResult(int n, int c, int h, int w,
                        int kernel_size, int stride, int padding);

void im2col(float *input, f16 *output,
            int n, int c, int h, int w,
            int kernel_size, int stride, int padding);
