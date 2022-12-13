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
        int out_height = (x.sizes()[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int out_width = (x.sizes()[3] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        output = Tensor({x.sizes()[0], x.sizes()[1], out_height, out_width});
		float* intput_data = x.data_ptr();
		float* output_data = output.data_ptr();

		// gpu execution
		dim3 dim_grid(1, 1);
		dim3 dim_block(out_height, out_width);
		MaxPool2dKernel<<<dim_grid, dim_block>>>(input_data, x.sizes()[2], x.sizes()[3], output_data, out_height, out_width, kernel_size, stride, padding);
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
    int output_size;
	Tensor output;

public:
    AdaptiveAvgPool2d(int output_size) : output_size(output_size)
    {
        // TODO
    }

    Tensor forward(Tensor x) override
    {
        output = Tensor({x.sizes()[0], x.sizes()[1], output_size, output_size});
		float* input_data = x.data_ptr();
		float* output_data = output.data_ptr();
		kernel_size_h = (x.sizes()[2] + output_size - 1) / output_size;
		kernel_size_w = (x.sizes()[3] + output_size - 1) / output_size;

		// gpu execution
		dim3 dim_grid(1, 1);
		dim3 dim_block(output_size, output_size);
		AdaptiveAvgPool2dKernel<<<dim_grid, dim_block>>>(input_data, x.sizes()[2], x.sizes()[3], output_data, output_size, output_size, kernel_size_h, kernel_size_w, stride);
		return output;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":AdaptiveAvgPool2d" << std::endl;
    }
};
