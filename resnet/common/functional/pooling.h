#pragma once

#include "common.h"

template <int block_size>
__global__ void deviceMaxPool2dKernel(float *input_data, int height, int width,
                                      float *output_data, int out_height, int out_width,
                                      int kernel_size, int padding, int stride);

void hostMaxPool2d(int batch_size, int num_channels,
                   float *input_data, int height, int width,
                   float *output_data, int out_height, int out_width,
                   int kernel_size, int padding, int stride);

__global__ void deviceAvgPool2dKernel(float *input_data, int height, int width,
                                      float *output_data, int out_height, int out_width,
                                      int kernel_size, int padding, int stride);

void hostAvgPool2d(int batch_size, int num_channels,
                   float *input_data, int height, int width,
                   float *output_data, int out_height, int out_width,
                   int kernel_size, int padding, int stride);