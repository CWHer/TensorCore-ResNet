#pragma once

#include "common.h"

void linear(f16 *input, float *output,
            f16 *weight, float *bias,
            int batch_size, int input_channel, int output_channel);

void makeLinearWeight(float *input, f16 *output, int row, int col);