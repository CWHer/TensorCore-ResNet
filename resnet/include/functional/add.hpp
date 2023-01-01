/** @file add.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_ADD_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_ADD_HPP

#include "common.hpp"

void add_(float *Result,
          const float *adder,
          size_t length,
          Impl::DeviceType device_type,
          cudaStream_t stream = cudaStream_t(nullptr));
[[maybe_unused]] void add(float *Result,
                          const float *adder_a,
                          const float *adder_b,
                          int length,
                          Impl::DeviceType device_type);
void relu_(float *Result, size_t length, Impl::DeviceType device_type);
// Fuse
void add_relu_(float *Result, const float *adder, size_t length, Impl::DeviceType device_type);
#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_ADD_HPP
