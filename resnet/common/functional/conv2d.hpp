/** @file conv2d.hpp
*/
#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP

#include "common.h"
#include <memory>

size_t im2col_result_size(int N, int C, int H, int W, int filter_height, int filter_width, int stride, int padding);

std::unique_ptr<float_16[]> create_im2col_result_store(int N,
                                                       int C,
                                                       int H,
                                                       int W,
                                                       int filter_height,
                                                       int filter_width,
                                                       int stride,
                                                       int padding);

void im2col(const float *input,
            float_16 *output,
            int N,
            int C,
            int H,
            int W,
            int filter_height,
            int filter_width,
            int stride,
            int padding,
            Impl::DeviceType device_type);

int conv2d_output_sizes(int N, int C, int H, int W, int out_channels, int kernel_size, int stride, int padding);

std::vector<int> conv2d_result_shape(int N,
                                     int C,
                                     int H,
                                     int W,
                                     int out_channels,
                                     int kernel_size,
                                     int stride,
                                     int padding);

void conv2d(const float *input,
            float *output,
            const float_16 *weight,
            const float *bias,
            int N,
            int C,
            int H,
            int W,
            int out_channels,
            int kernel_size,
            int stride,
            int padding);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP
