/** @file gemm_col_major.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP

#include "common.h"
#include "tensor.hpp"

void gemm_naive(const float_16 *A, const float_16 *B, float_32 *C, size_t M, size_t N, size_t K);

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ArgumentSelectionDefects"
// Currently redirect GEMM to naive implementation.
inline void gemm_col_major(const float_16 *A, const float_16 *B, float_32 *C, size_t M, size_t N, size_t K) {
  gemm_naive(A, B, C, M, N, K);
}

// GEMM row-major
inline void gemm_row_major(const float_16 *A, const float_16 *B, float_32 *C, size_t M, size_t N, size_t K) {
  gemm_naive(B, A, C, N, M, K);
}
#pragma clang diagnostic pop

Tensor Mul(const Tensor &A, const Tensor &B);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
