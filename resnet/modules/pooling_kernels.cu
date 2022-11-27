#pragma once

#include "../common/common.h"
#include "module.hpp"
#include <float.h>

__global__ void MaxPool2dKernel(const float* in_data, const int in_height, const int in_width,
		                        float* out_data, const int out_height, const int out_width,
								const int kernel_size, const int stride, const int padding)
{
	int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	int out_y = blockIdx.y * blockDim.y + threadIdx.y;
	int in_start_x = stride * out_x;
	int in_start_y = stride * out_y;
	float max_val = -FLT_MAX;
	for (int x = in_start_x; x < in_start_x + kernel_size && x < in_width + padding; x++)
	{
		for (int y = in_start_y; y < in_start_y + kernel_size && in_height + padding; y++)
		{
			int cur = in_data[x * in_width + y];
			max_val = cur > max_val ? cur : max_val;
		}
	}
	out_data[out_x * out_width + out_y] = max_val;
}


__global__ void AdaptiveAvgPool2dKernel(const float* in_data, const int in_height, const int in_width,
		                        float* out_data, const int out_height, const int out_width,
								const int kernel_size_h, const int kernel_size_w, const int stride)
{
	int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	int out_y = blockIdx.y * blockDim.y + threadIdx.y;
	int in_start_x = stride * out_x;
	int in_start_y = stride * out_y;
	float sum = 0;
	int num = 0;
	for (int x = in_start_x; x < in_start_x + kernel_size_w && x < in_width; x++)
	{
		for (int y = in_start_y; y < in_start_y + kernel_size_h && in_height; y++)
		{
			sum += in_data[x * in_width + y];
			num++;
		}
	}
	out_data[out_x * out_width + out_y] = sum / num;
}
