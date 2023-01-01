#pragma once

#include "common.h"

void conv2d(float *input, float *output, f16 *weight,
            int n, int c, int h, int w,
            int out_channels, int out_height, int out_width,
            int kernel_size, int stride, int padding);