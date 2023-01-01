/** @file gemm_helpers.cu
*/

#include "common.hpp"
#include "functional/gemm.hpp"
#include "mem_pool.h"

using namespace Impl;

void gemm(const float_16 *A,
          const float_16 *B,
          float_32 *C,
          size_t M,
          size_t N,
          size_t K,
          const GEMM::Major major,
          const Impl::DeviceType device_type) {
  cudaStream_t stream = nullptr;
  gemm_stream(A, B, C, M, N, K, major, device_type, stream);
  cudaStreamSynchronize(stream);
  cudaCacheCommit(stream);
}
