/** @file gemm.cu
*/
#include <mma.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_gl_interop.h>
#include "common.hpp"
#include "functional/macros.hpp"
#include "functional/gemm.hpp"

using namespace nvcuda;
using namespace Impl;

static const int warp_size = 32;

// For Volta architecture, only FP16 of 16x16x16 is supported.
static const int volta_m_factor = 16;
static const int volta_n_factor = 16;
static const int volta_k_factor = 16;

template<int block_col_warps, int block_row_warps> static __global__ void gemm_naive_kernel(const float_16 *A,
                                                                                            const float_16 *B,
                                                                                            float_32 *Result,
                                                                                            size_t M,
                                                                                            size_t N,
                                                                                            size_t K) {

  constexpr int block_threads = warp_size * (block_row_warps * block_col_warps);

  constexpr int tile_m = block_row_warps * volta_m_factor;
  constexpr int tile_n = block_col_warps * volta_n_factor;
  constexpr int tile_k = tile_m;

  __shared__ float_16 As[tile_k][tile_m];
  __shared__ float_16 Bs[tile_n][tile_k];

  const auto tid = threadIdx.y * blockDim.x + threadIdx.x;

  const auto aRow = blockIdx.x * tile_m;
  const auto bCol = blockIdx.y * tile_n;

  wmma::fragment<wmma::matrix_a, volta_m_factor, volta_n_factor, volta_k_factor, float_16, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, volta_m_factor, volta_n_factor, volta_k_factor, float_16, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, volta_m_factor, volta_n_factor, volta_k_factor, float_32> result_frag;
  wmma::fill_fragment(result_frag, 0.0f);

  for (int k = 0; k < K; k += tile_k) {
    // Parallel loading
    for (int i = 0; i < tile_m * tile_k; i += block_threads) {
      const auto idx = (tid + i);

      const auto As_row = idx % tile_m;
      const auto As_col = idx / tile_m;
      const auto Bs_row = idx % tile_k;
      const auto Bs_col = idx / tile_k;

      if (likely(aRow + As_row < M && k + As_col < K)) {
        As[As_col][As_row] = A[(k + As_col) * M + aRow + As_row];
      } else {
        As[As_col][As_row] = 0;
      }

      if (likely(k + Bs_row < K && bCol + Bs_col < N)) {
        Bs[Bs_col][Bs_row] = B[(bCol + Bs_col) * K + k + Bs_row];
      } else {
        Bs[Bs_col][Bs_row] = 0;
      }

    }
    __syncthreads();

    for (int i = 0; i < tile_k; i += volta_k_factor) {
      const auto As_offset = i * tile_m + volta_m_factor * (threadIdx.x / warp_size);
      const auto Bs_offset = volta_n_factor * threadIdx.y * tile_k + i;

      wmma::load_matrix_sync(a_frag, (half *) As + As_offset, tile_m);
      wmma::load_matrix_sync(b_frag, (half *) Bs + Bs_offset, tile_k);

      wmma::mma_sync(result_frag, a_frag, b_frag, result_frag);
    }
  }

  const auto cRow = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size * volta_m_factor;
  const auto cCol = (blockIdx.y * blockDim.y + threadIdx.y) * volta_n_factor;
  const auto c_offset = cRow + cCol * M;

  if (likely(cRow < M && cCol < N)) {
    wmma::store_matrix_sync(Result + c_offset, result_frag, M, wmma::mem_col_major);
  }
}

template<int block_col_warps, int block_row_warps> static void gemm_naive_caller(const float_16 *A,
                                                                                 const float_16 *B,
                                                                                 float_32 *Result,
                                                                                 size_t M,
                                                                                 size_t N,
                                                                                 size_t K,
                                                                                 cudaStream_t &stream) {
  constexpr int tile_m = block_row_warps * volta_m_factor;
  constexpr int tile_n = block_col_warps * volta_n_factor;

  dim3 grid((M + (tile_m - 1)) / tile_m, (N + (tile_n - 1)) / tile_n);
  dim3 block(block_row_warps * warp_size, block_col_warps);

  gemm_naive_kernel<block_col_warps, block_row_warps><<<grid, block, 0, stream>>>(A, B, Result, M, N, K);
}

template<typename T, cudaMemcpyKind memcpy_kind, bool require_copy> static T *gemm_padding_col_major(const T * RESTRICT source,
                                                                                                     size_t row,
                                                                                                     size_t col,
                                                                                                     size_t pad_row,
                                                                                                     size_t pad_col,
                                                                                                     cudaStream_t &stream,
                                                                                                     T * padded = nullptr) {
  constexpr bool same_device = (memcpy_kind == cudaMemcpyHostToHost || memcpy_kind == cudaMemcpyDeviceToDevice);
  if ((col == pad_col) && (row == pad_row) && same_device)
    return (T *) source;

  if (padded == nullptr) {
    checkCudaErrors(cudaMallocAsyncIfAvailable(&padded, sizeof(T) * pad_col * pad_row, stream));
  }

  if (require_copy) {
    checkCudaErrors(cudaMemsetAsync((void *) padded, 0, sizeof(T) * pad_col * pad_row, stream));
    checkCudaErrors(cudaMemcpy2DAsync(padded,
                                      sizeof(T) * pad_col,
                                      source,
                                      sizeof(T) * col,
                                      sizeof(T) * col,
                                      row,
                                      memcpy_kind,
                                      stream));
  }

  return padded;
}

template<typename T, cudaMemcpyKind memcpy_kind> static void gemm_unpad_col_major(T * RESTRICT source,
                                                                                  T * RESTRICT padded,
                                                                                  size_t row,
                                                                                  size_t col,
                                                                                  size_t pad_row,
                                                                                  size_t pad_col,
                                                                                  cudaStream_t &stream) {
  constexpr bool same_device = (memcpy_kind == cudaMemcpyHostToHost || memcpy_kind == cudaMemcpyDeviceToDevice);
  if (source == padded && same_device) {
    return;
  }

  if ((col == pad_col) && (row == pad_row)) {
    checkCudaErrors(cudaMemcpyAsync(source, padded, sizeof(T) * col * row, memcpy_kind, stream));
  } else {
    checkCudaErrors(cudaMemcpy2DAsync(source,
                                      sizeof(T) * col,
                                      padded,
                                      sizeof(T) * pad_col,
                                      sizeof(T) * col,
                                      row,
                                      memcpy_kind,
                                      stream));
  }
}

static void gemm_device_memory(const float_16 *A,
                               const float_16 *B,
                               float_32 *Result,
                               size_t M,
                               size_t N,
                               size_t K,
                               cudaStream_t &stream) {
  float_16 *padded_A;
  float_16 *padded_B;
  float_32 *padded_C;

  // If M and N are not by 16, we need to pad them.
  auto padded_M = (M + (volta_m_factor - 1)) / volta_m_factor * volta_m_factor;
  auto padded_N = (N + (volta_n_factor - 1)) / volta_n_factor * volta_n_factor;

  // Copy A and B by padding to device
  // B needs padding, with a leading dimension of N

  padded_A = gemm_padding_col_major<float_16, cudaMemcpyDeviceToDevice, true>(A, K, M, K, padded_M, stream);
  padded_B = gemm_padding_col_major<float_16, cudaMemcpyDeviceToDevice, true>(B, N, K, padded_N, K, stream);
  padded_C =
      gemm_padding_col_major<float_32, cudaMemcpyDeviceToDevice, false>(Result, N, M, padded_N, padded_M, stream);

  gemm_naive_caller<2, 2>(padded_A, padded_B, padded_C, padded_M, padded_N, K, stream);

  if (padded_A != A)
    checkCudaErrors(cudaFreeAsyncIfAvailable(padded_A, stream));

  if (padded_B != B)
    checkCudaErrors(cudaFreeAsyncIfAvailable(padded_B, stream));

  if (padded_C != Result) {
    gemm_unpad_col_major<float_32, cudaMemcpyDeviceToDevice>(Result, padded_C, N, M, padded_N, padded_M, stream);
    checkCudaErrors(cudaFreeAsyncIfAvailable(padded_C, stream));
  }

}

static void gemm_host_memory(const float_16 * RESTRICT A,
                             const float_16 * RESTRICT B,
                             float_32 * RESTRICT Result,
                             size_t M,
                             size_t N,
                             size_t K,
                             cudaStream_t &stream) {
  float_16 *padded_A;
  float_16 *padded_B;
  float_32 *padded_C;

  // If M and N are not by 16, we need to pad them.
  auto padded_M = (M + (volta_m_factor - 1)) / volta_m_factor * volta_m_factor;
  auto padded_N = (N + (volta_n_factor - 1)) / volta_n_factor * volta_n_factor;

  // Copy A and B by padding to device
  // Copy A and B by padding to device
  // B needs padding, with a leading dimension of N
  padded_A = gemm_padding_col_major<float_16, cudaMemcpyHostToDevice, true>(A, K, M, K, padded_M, stream);
  padded_B = gemm_padding_col_major<float_16, cudaMemcpyHostToDevice, true>(B, N, K, padded_N, K, stream);
  padded_C = gemm_padding_col_major<float_32, cudaMemcpyHostToDevice, false>(Result, N, M, padded_N, padded_M, stream);

  gemm_device_memory(padded_A, padded_B, padded_C, padded_M, padded_N, K, stream);

  gemm_unpad_col_major<float_32, cudaMemcpyDeviceToHost>(Result, padded_C, N, M, padded_N, padded_M, stream);

  checkCudaErrors(cudaFreeAsyncIfAvailable(padded_A, stream));
  checkCudaErrors(cudaFreeAsyncIfAvailable(padded_B, stream));
  checkCudaErrors(cudaFreeAsyncIfAvailable(padded_C, stream));
}

void gemm_stream(const float_16 * RESTRICT A,
                 const float_16 *RESTRICT B,
                 float_32 * RESTRICT C,
                 size_t M,
                 size_t N,
                 size_t K,
                 const GEMM::Major major,
                 const Impl::DeviceType device_type,
                 cudaStream_t &stream) {
  switch (device_type) {
  case Impl::DeviceType::CPU:
    switch (major) {
    case GEMM::Major::col_major:gemm_host_memory(A, B, C, M, N, K, stream);
      break;
    case GEMM::Major::row_major:gemm_host_memory(B, A, C, N, M, K, stream);
      break;
    }
    break;
  case Impl::DeviceType::CUDA:
    switch (major) {
    case GEMM::Major::col_major:gemm_device_memory(A, B, C, M, N, K, stream);
      break;
    case GEMM::Major::row_major:gemm_device_memory(B, A, C, N, M, K, stream);
      break;
    }
    break;
  }
}

void gemm_batched_B(const float_16 * RESTRICT A,
                    const float_16 * RESTRICT B,
                    float_32 * RESTRICT C,
                    size_t M,
                    size_t N,
                    size_t K,
                    size_t batch_size,
                    [[maybe_unused]] GEMM::Major major,
                    Impl::DeviceType device_type) {
#if DEBUG
  if (major != GEMM::Major::row_major) {
    // Batches does not support col-major
    throw std::runtime_error("Batches does not support col-major");
  }
#endif

  constexpr int stream_count = 8;
  cudaStream_t streams[stream_count];
  for (auto &stream : streams) {
    cudaStreamCreate(&stream);
  }

  for (size_t i = 0; i < batch_size; i++) {
    auto B_ptr = B + i * K * N;
    auto C_ptr = C + i * M * N;
    gemm_stream(A, B_ptr, C_ptr, M, N, K, GEMM::Major::row_major, device_type, streams[i % stream_count]);
  }

  for (auto &stream : streams) {
    // Only sync if we are using async memory allocation
    // Otherwise, we are already synced in GEMM execution
    cudaStreamSynchronize(stream);
    cudaCacheCommit(stream);
    cudaStreamDestroy(stream);
  }
}


void gemm(const float_16 * RESTRICT A,
          const float_16 * RESTRICT B,
          float_32 * RESTRICT C,
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
