#pragma once

#include "common.h"
#include "module.hpp"
#include "pooling_kernels.cu"

class MaxPool2d : public Module
{
private:
    int kernel_size, stride, padding, dilation, ceil_mode;
	Tensor output;

public:
    MaxPool2d(int kernel_size = 3, int stride = 2, int padding = 0,
              int dilation = 1, bool ceil_mode = false)
        : kernel_size(kernel_size), stride(stride), padding(padding),
          dilation(dilation), ceil_mode(ceil_mode)
    {
        // TODO
    }

    Tensor forward(Tensor x) override
    {
		int in_channel = x.sizes()[0];
		int in_height = x.sizes()[1];
		int in_width = x.sizes()[2];
        int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
		float* intput_data = x.data_ptr();
		float* output_data = new float[in_channel * out_height * out_width];

		// gpu execution
		int in_single_slice_size = in_height * in_width;
		float* in_data_device_1_slice;
		cudaMalloc(&in_data_device_1_slice, in_single_slice_size * sizeof(float));
		int out_single_slice_size = out_height * out_width;
		float* out_data_device_1_slice;
		cudaMalloc(&out_data_device_1_slice, out_single_slice_size * sizeof(float));
		dim3 dim_grid(1, 1);
		dim3 dim_block(out_height, out_width);
		for (int i = 0; i < in_channel; i++)
		{
			// copy one single slice to device
			cudaMemcpy(in_data_device_1_slice, intput_data + i * in_single_slice_size, in_single_slice_size * sizeof(float), cudaMemcpyHostToDevice);
			// do the computation, compute one single slice for a time
			AdaptiveAvgPool2dKernel<<<dim_grid, dim_block>>>(in_data_device_1_slice, in_height, in_width, out_data_device_1_slice, out_height, out_width, kernel_size_h, kernel_size_w, stride);
			// copy the result to output
			cudaMemcpy(output_data + i * out_single_slice_size, out_data_device_1_slice, out_single_slice_size * sizeof(float), cudaMemcpyDeviceToHost);
		}
        output = Tensor({in_channel, out_height, out_width}, DeviceType::CPU, output_data);
		return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":MaxPool2d" << std::endl;
    }
};

class AdaptiveAvgPool2d : public Module
{
private:
    int kernel_size_h, kernel_size_w;
	int stride = 1;
    int out_height;
    int out_width;
	Tensor output;

public:
    AdaptiveAvgPool2d(int output_height, int output_width) : out_height(output_height), out_width(output_width)
    {
        // TODO
    }

    Tensor forward(Tensor x) override
    {
		int in_channel = x.sizes()[0];
		int in_height = x.sizes()[1];
		int in_width = x.sizes()[2];
		kernel_size_h = (in_height + out_height - 1) / out_height;
		kernel_size_w = (in_width + out_width - 1) / out_width;
		float* input_data = x.data_ptr();
		float* output_data = new float[in_channel * out_height * out_width];

		// gpu execution
		int in_single_slice_size = in_height * in_width;
		float* in_data_device_1_slice;
		cudaMalloc(&in_data_device_1_slice, in_single_slice_size * sizeof(float));
		int out_single_slice_size = out_height * out_width;
		float* out_data_device_1_slice;
		cudaMalloc(&out_data_device_1_slice, out_single_slice_size * sizeof(float));
		dim3 dim_grid(1, 1);
		dim3 dim_block(out_height, out_width);
		for (int i = 0; i < in_channel; i++)
		{
			// calculate the bias of the array
			int bias = i * single_slice_size;
			// copy one single slice to device
			cudaMemcpy(in_data_device_1_slice, intput_data + bias, in_single_slice_size * sizeof(float), cudaMemcpyHostToDevice);
			// do the computation, compute one single slice for a time
			MaxPool2dKernel<<<dim_grid, dim_block>>>(in_data_device_1_slice, in_height, in_width, out_data_device_1_slice, out_height, out_width, kernel_size, stride, padding);
			// copy the result to output
			cudaMemcpy(output_data + bias, out_data_device_1_slice, out_single_slice_size * sizeof(float), cudaMemcpyDeviceToHost);
		}
		AdaptiveAvgPool2dKernel<<<dim_grid, dim_block>>>(input_data, in_height, in_width, output_data, output_size, output_size, kernel_size_h, kernel_size_w, stride);
        output = Tensor({in_channel, out_height, out_width}, DeviceType::CPU, out_data);
		return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":AdaptiveAvgPool2d" << std::endl;
    }
};
