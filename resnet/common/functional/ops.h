#pragma once

#include "common.h"

__global__ void kernelAdd_(float *result, float *adder, int length);

__global__ void kernelRelu_(float *result, int length);

void hostAdd_(float *result, float *adder_a, int length);

void hostRelu_(float *result, int length);