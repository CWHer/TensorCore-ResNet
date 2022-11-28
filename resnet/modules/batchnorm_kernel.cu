#include "common.h"
#include "module.hpp"
#include <numeric>

__global__ void BatchNorm2dKernel(float* input_data, int num_features, int num_elements, float eps)
{
	int t_id = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	float* data_start = input_data + t_id;
	float* data_end = nullptr;
	if (t_id + num_elements <= num_features)
		data_end = data_start + num_elements;
	else
		data_end = input_data + num_features;

	float sum = std::accumulate(data_start, data_end, 0);
	// calculate mean
	float mean = sum / num_elements;
	// calculate population standard deviation and sample standard deviation
	float diff_pow_sum = 0.0;
	float* data_cur = data_start;
	while (data_cur != data_end)
	{
		float diff = *data_cur - mean;
		diff_pow_sum += diff * diff;
		data_cur++;
	}
	float population_std = diff_pow_sum / num_elements;

	// update input data
	data_cur = data_start;
	float denominator = population_std + eps;
	while (data_cur != data_end)
	{
		*data_cur = (*data_cur - mean) / denominator;
		data_cur++;
	}
}
