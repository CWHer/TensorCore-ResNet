/** @file linear.hpp
*/
#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_LINEAR_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_LINEAR_HPP

#include "common.h"

void linear(const float_16 *input,
            float *output,
            const float_16 *weight,
            const float *bias,
            int batch_size,
            int input_channel,
            int output_channel,
            Impl::DeviceType device_type);

void prepare_linear_weight(const float *input, float_16 *output, int row, int col, Impl::DeviceType device_type);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_LINEAR_HPP
