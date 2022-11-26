/** @file gemm.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP

#include "common.h"
#include "tensor.hpp"

// Remember that all the functions in this file are column-major
// This is resolvable by transposing the input matrices:
// $B^T A^T = C^T$

void gemm_naive(const float_16 *, const float_16 *, float_32 *, size_t, size_t, size_t);

// Currently redirect GEMM to naive implementation.
inline void gemm(const float_16 *A, const float_16 *B, float_32 *C, size_t M, size_t N, size_t K) {
  gemm_naive(A, B, C, M, N, K);
}

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_GEMM_HPP
