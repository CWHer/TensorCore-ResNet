/** @file gemm_helpers.cu
*/

#include "common.h"
#include "functional/gemm.hpp"

void gemm(const float_16 *A,
          const float_16 *B,
          float_32 *C,
          size_t M,
          size_t N,
          size_t K,
          const GEMM::Major major,
          const Impl::DeviceType device_type) {
  cudaStream_t stream = nullptr;
  return gemm_stream(A, B, C, M, N, K, major, device_type, stream);
}

void gemm_batched_B(const float_16 *A,
                    const float_16 *B,
                    float_32 *C,
                    size_t M,
                    size_t N,
                    size_t K,
                    size_t batch_size,
                    GEMM::Major major,
                    Impl::DeviceType device_type) {
#if DEBUG
  if (major != GEMM::Major::row_major) {
    // Batches does not support col-major
    throw std::runtime_error("Batches does not support col-major");
  }
#endif

  constexpr int stream_count = 8;
  cudaStream_t streams[stream_count];
  for (auto & stream : streams) {
    cudaStreamCreate(&stream);
  }

  for (int i = 0; i < batch_size; i++) {
    auto B_ptr = B + i * K * N;
    auto C_ptr = C + i * M * N;
    gemm_stream(A, B_ptr, C_ptr, M, N, K, major, device_type, streams[i % stream_count]);
  }

  for (auto & stream : streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
}
