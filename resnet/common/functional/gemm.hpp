/** @file gemm_col_major.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP

#include "common.h"

namespace GEMM {
enum Major {
  col_major, row_major
};

}

void gemm_host_memory(const float_16 *A, const float_16 *B, float_32 *Result, size_t M, size_t N, size_t K);
void gemm_device_memory(const float_16 *A, const float_16 *B, float_32 *C, size_t M, size_t N, size_t K);
void gemm(const float_16 *A,
          const float_16 *B,
          float_32 *C,
          size_t M,
          size_t N,
          size_t K,
          GEMM::Major major = GEMM::Major::row_major,
          Impl::DeviceType device_type = Impl::DeviceType::CPU);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
