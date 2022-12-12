/** @file conv2d.hpp
*/
#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP

#include "common.h"
#include <memory>

std::unique_ptr<float_16[]> create_im2col_result_store(int C,
                                                       int H,
                                                       int W,
                                                       int filter_height,
                                                       int filter_width,
                                                       int stride,
                                                       int padding);

void im2col_naive(float *input,
                  float_16 *output,
                  int C,
                  int H,
                  int W,
                  int filter_height,
                  int filter_width,
                  int stride,
                  int padding);

void im2col_cuda(float *input,
                 float_16 *output,
                 int C,
                 int H,
                 int W,
                 int filter_height,
                 int filter_width,
                 int stride,
                 int padding);

/**
 * @copydoc im2col_naive
 */
inline void im2col(float *input,
                   float_16 *output,
                   int C,
                   int H,
                   int W,
                   int filter_height,
                   int filter_width,
                   int stride,
                   int padding) {
  return im2col_cuda(input, output, C, H, W, filter_height, filter_width, stride, padding);
}

size_t im2col_result_size(int C, int H, int W, int filter_height, int filter_width, int stride, int padding);

void conv2d(float *input,
            float *output,
            float_16 *weight,
            float *bias,
            int C,
            int H,
            int W,
            int out_channels,
            int kernel_size,
            int stride,
            int padding);

std::vector<int> conv2d_result_shape(int C, int H, int W, int out_channels, int kernel_size, int stride, int padding);
#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_CONV2D_HPP
