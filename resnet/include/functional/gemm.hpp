/** @file gemm_col_major.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP

#include "common.hpp"

namespace GEMM {
enum Major {
  col_major, row_major
};

}

// functional/fp16_array.cu
float_32 *fp16_array_to_fp32_array(const float_16 *fp16_array, size_t size, Impl::DeviceType device_type);
float_16 *fp32_array_to_fp16_array(const float_32 *fp32_array, size_t size, Impl::DeviceType device_type);

// functional/gemm.cu or simulator/gemm.cpp
void gemm_stream(const float_16 *A,
                 const float_16 *B,
                 float_32 *C,
                 size_t M,
                 size_t N,
                 size_t K,
                 GEMM::Major major,
                 Impl::DeviceType device_type,
                 cudaStream_t &stream);

// functional/gemm_helpers.cu
void gemm(const float_16 *A,
          const float_16 *B,
          float_32 *C,
          size_t M,
          size_t N,
          size_t K,
          GEMM::Major major = GEMM::Major::row_major,
          Impl::DeviceType device_type = Impl::DeviceType::CPU);
void gemm_batched_B(const float_16 *A,
                    const float_16 *B,
                    float_32 *C,
                    size_t M,
                    size_t N,
                    size_t K,
                    size_t batch_size,
                    GEMM::Major major,
                    Impl::DeviceType device_type);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
