/** @file add.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_ADD_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_ADD_HPP

#include "common.h"

void add_(float *Result, const float *adder, int length, Impl::DeviceType device_type);
void add(float *Result, const float *adder_a, const float *adder_b, int length, Impl::DeviceType device_type);
void relu_(float *Result, int length, Impl::DeviceType device_type);
#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_ADD_HPP
