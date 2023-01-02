/** @file conv2d.hpp
*/
#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP

#include "common.hpp"
#include <memory>
#include "mem_pool.h"

/* @brief im2col result shape
 */
constexpr size_t im2col_result_size(int N, int C, int H, int W, int filter_height, int filter_width, int stride, int padding) {
  int output_height = (H + 2 * padding - filter_height) / stride + 1;
  int output_width = (W + 2 * padding - filter_width) / stride + 1;
  int output_size = output_height * output_width;
  return N * C * filter_height * filter_width * output_size;
}

std::unique_ptr<float_16[]> create_im2col_result_store_host(int N,
                                                            int C,
                                                            int H,
                                                            int W,
                                                            int filter_height,
                                                            int filter_width,
                                                            int stride,
                                                            int padding);

std::unique_ptr<float_16[], decltype(&Impl::cudaPooledFree)> create_im2col_result_store_device(int N,
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
            int padding,
            Impl::DeviceType device_type);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP
