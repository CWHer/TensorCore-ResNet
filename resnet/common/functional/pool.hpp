/** @file pool.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_POOL_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_POOL_HPP

void maxpool2d(const float *input_data,
               int batch_size,
               int num_channels,
               int height,
               int width,
               float *output_data,
               int out_height,
               int out_width,
               int kernel_size,
               int padding,
               int stride);

void avgpool2d(const float *input_data,
               int batch_size,
               int num_channels,
               int height,
               int width,
               float *output_data,
               int out_height,
               int out_width,
               int kernel_size,
               int padding,
               int stride);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_POOL_HPP
