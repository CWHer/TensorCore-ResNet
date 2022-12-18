/**
 * @file test/test_pooling.cpp
 * @brief Test pooling functionality
 */

#include <gtest/gtest.h>
#include "pooling.hpp"
#include <float.h>

static void max_pool_2d_naive(float* input_data, int in_height, int in_width, int channel,
		                      float* output_data, int out_height, int out_width,
		                      int kernel_size = 3, int stride = 2, int padding = 0,
		                      int dilation = 1, bool ceil_mode = false)
{
	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < out_height; i++)
		{
			int i_start = stride * i;
			for (int j = 0; j < out_width; j++)
			{
				int j_start = stride * j;
				float max_val = -FLT_MAX;
				for (int ii = i_start; ii < in_height && ii < i_start + kernel_size; ii++)
				{
					for (int jj = j_start; jj < in_width && jj < j_start + kernel_size; jj++)
					{
						float cur = input_data[ii * in_width + jj];
						max_val = cur > max_val ? cur : max_val;
					}
				}
				output_data[c * out_height * out_width + i * out_width + j] = max_val;
			}
		}
	}
}

static void adaptive_avg_pool_2d_naive(float* input_data, int in_height, int in_width, int channel,
		                               float* output_data, int out_height, int out_width,
		                               int kernel_size_h, int kernel_size_w, int stride = 1)
{
	for (int c = 0; c < channel; c++)
	{
		for (int i = 0; i < out_height; i++)
		{
			int i_start = stride * i;
			for (int j = 0; j < out_width; j++)
			{
				int j_start = stride * j;
				float sum = 0;
				int num = 0;
				for (int ii = i_start; ii < in_height && ii < i_start + kernel_size_h; ii++)
				{
					for (int jj = j_start; jj < in_width && jj < j_start + kernel_size_w; jj++)
					{
						sum += input_data[ii * in_width + jj];
						num++;
					}
				}
				output_data[c * out_height * out_width + i * out_width + j] = sum / num;
			}
		}
	}
}

TEST(pooling, max_pool_2d)
{
	// Test if MaxPool2d works properly
	float data_[24] = {1, 2, 3, 4, 4, 3, 2, 1, 5, 5, 5, 5,
                       1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0, 0};
	float *data = new float[24];
	std::copy(data_, data_ + 24, data);
	Tensor x({2,3,4}, Impl::DeviceType::CPU, data);

	// cuda implementation
	int kernel_size = 3, stride = 2, padding = 0, dilation = 1, ceil_mode = false;
	MaxPool2d mp2d(kernel_size, stride, padding, dilation, ceil_mode);
	Tensor res = mp2d.forward(x);
	float* cuda_out_data = res.data_ptr();

	// cpu implementation
	int in_channel = x.sizes()[0];
	int in_height = x.sizes()[1];
	int in_width = x.sizes()[2];
	int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
	int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
	float* naive_out_data = new float[in_channel * in_height * in_width];
	max_pool_2d_naive(data, in_height, in_width, in_channel,
			          naive_out_data, out_height, out_width,
					  kernel_size, stride, padding,
					  dilation, ceil_mode);

	const float eps = 1e-3;
	int result_size = in_channel * out_height * out_width;
	for (int i = 0; i < result_size; i++)
	{
		EXPECT_NEAR(naive_out_data[i], cuda_out_data[i], eps);
	}
}

TEST(pooling, adaptive_avg_pool_2d)
{
	// Test if AdaptiveAvgPool2d works properly
	float data_[24] = {1, 2, 3, 4, 4, 3, 2, 1, 5, 5, 5, 5,
                       1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0, 0};
	float *data = new float[24];
	std::copy(data_, data_ + 24, data);
	Tensor x({2,3,4}, Impl::DeviceType::CPU, data);
	int out_height = 2;
	int out_width = 2;
	int stride = 1;

	// cuda implementation
	AdaptiveAvgPool2d aap2d(out_height, out_width);
	Tensor res = aap2d.forward(x);
	float* cuda_out_data = res.data_ptr();

	// cpu implementation
	int in_channel = x.sizes()[0];
	int in_height = x.sizes()[1];
	int in_width = x.sizes()[2];
    int kernel_size_h = (in_height + out_height - 1) / out_height;
    int kernel_size_w = (in_width + out_width - 1) / out_width;
	float* naive_out_data = new float[in_channel * in_height * in_width];
	adaptive_avg_pool_2d_naive(data, in_height, in_width, in_channel,
			                   naive_out_data, out_height, out_width,
							   kernel_size_h, kernel_size_w, stride);

	const float eps = 1e-3;
	int result_size = in_channel * out_height * out_width;
	for (int i = 0; i < result_size; i++)
	{
		EXPECT_NEAR(naive_out_data[i], cuda_out_data[i], eps);
	}
}
