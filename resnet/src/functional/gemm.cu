/** @file gemm.cu
*/
#include <mma.h>
#include <cublas_v2.h>
#include "common.h"
#include "functional/macros.h"
#include "functional/gemm.hpp"

using namespace nvcuda;
using namespace Impl;

static void check_cuda_error() {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

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

  const int block_threads = warp_size * (block_row_warps * block_col_warps);

  constexpr int tile_m = block_row_warps * volta_m_factor;
  constexpr int tile_n = block_col_warps * volta_n_factor;
  constexpr int tile_k = tile_m;

  __shared__ float_16 As[tile_k][tile_m];
  __shared__ float_16 Bs[tile_n][tile_k];

  auto tid = threadIdx.y * blockDim.x + threadIdx.x;

  auto aRow = blockIdx.x * tile_m;
  auto bCol = blockIdx.y * tile_n;

  wmma::fragment<wmma::matrix_a, volta_m_factor, volta_n_factor, volta_k_factor, float_16, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, volta_m_factor, volta_n_factor, volta_k_factor, float_16, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, volta_m_factor, volta_n_factor, volta_k_factor, float_32> result_frag;
  wmma::fill_fragment(result_frag, 0.0f);

  for (int k = 0; k < K; k += tile_k) {
    // Parallel loading
    for (int i = 0; i < tile_m * tile_k; i += block_threads) {
      auto idx = (tid + i);

      auto As_row = idx % tile_m;
      auto As_col = idx / tile_m;
      auto Bs_row = idx % tile_k;
      auto Bs_col = idx / tile_k;

      if (aRow + As_row < M && k + As_col < K) {
        As[As_col][As_row] = A[(k + As_col) * M + aRow + As_row];
      } else {
        As[As_col][As_row] = 0;
      }

      if (k + Bs_row < K && bCol + Bs_col < N) {
        Bs[Bs_col][Bs_row] = B[(bCol + Bs_col) * K + k + Bs_row];
      } else {
        Bs[Bs_col][Bs_row] = 0;
      }

    }
    __syncthreads();

    for (int i = 0; i < tile_k; i += volta_k_factor) {
      auto As_offset = i * tile_m + volta_m_factor * (threadIdx.x / warp_size);
      auto Bs_offset = volta_n_factor * threadIdx.y * tile_k + i;

      wmma::load_matrix_sync(a_frag, (half *) As + As_offset, tile_m);
      wmma::load_matrix_sync(b_frag, (half *) Bs + Bs_offset, tile_k);

      wmma::mma_sync(result_frag, a_frag, b_frag, result_frag);
    }
  }

  auto cRow = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size * volta_m_factor;
  auto cCol = (blockIdx.y * blockDim.y + threadIdx.y) * volta_n_factor;
  auto c_offset = cRow + cCol * M;

  if (cRow < M && cCol < N) {
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

template<typename T, cudaMemcpyKind memcpy_kind, bool require_copy> static T *gemm_padding_col_major(const T *source,
                                                                                                     size_t row,
                                                                                                     size_t col,
                                                                                                     size_t pad_row,
                                                                                                     size_t pad_col,
                                                                                                     cudaStream_t &stream) {
  if ((col == pad_col) && (row == pad_row)
      && (memcpy_kind == cudaMemcpyHostToHost || memcpy_kind == cudaMemcpyDeviceToDevice))
    return (T *) source;

  T *padded;
  checkCudaErrors(cudaMallocAsyncIfAvailable(&padded, sizeof(T) * pad_col * pad_row, stream));

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

template<typename T, cudaMemcpyKind memcpy_kind> static void gemm_unpad_col_major(T *source,
                                                                                  T *padded,
                                                                                  size_t row,
                                                                                  size_t col,
                                                                                  size_t pad_row,
                                                                                  size_t pad_col,
                                                                                  cudaStream_t &stream) {
  if (source == padded && (memcpy_kind == cudaMemcpyHostToHost || memcpy_kind == cudaMemcpyDeviceToDevice)) {
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

  // Fixme: this template parameter is adjustable
  gemm_naive_caller<4, 4>(padded_A, padded_B, padded_C, padded_M, padded_N, K, stream);

  if (padded_C != Result) {
    gemm_unpad_col_major<float_32, cudaMemcpyDeviceToDevice>(Result, padded_C, N, M, padded_N, padded_M, stream);

  }

#if not CUDA_MALLOC_ASYNC
    checkCudaErrors(cudaStreamSynchronize(stream));
#endif

  if (padded_A != A)
    checkCudaErrors(cudaFreeAsyncIfAvailable(padded_A, stream));

  if (padded_B != B)
    checkCudaErrors(cudaFreeAsyncIfAvailable(padded_B, stream));

  if (padded_C != Result) {
    checkCudaErrors(cudaFreeAsyncIfAvailable(padded_C, stream));
  }

}

static void gemm_host_memory(const float_16 *A,
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
  // Copy A and B by padding to device
  // B needs padding, with a leading dimension of N
  padded_A = gemm_padding_col_major<float_16, cudaMemcpyHostToDevice, true>(A, K, M, K, padded_M, stream);
  padded_B = gemm_padding_col_major<float_16, cudaMemcpyHostToDevice, true>(B, N, K, padded_N, K, stream);
  padded_C = gemm_padding_col_major<float_32, cudaMemcpyHostToDevice, false>(Result, N, M, padded_N, padded_M, stream);

  // Fixme: this template parameter is adjustable
  gemm_device_memory(padded_A, padded_B, padded_C, padded_M, padded_N, K, stream);

  gemm_unpad_col_major<float_32, cudaMemcpyDeviceToHost>(Result, padded_C, N, M, padded_N, padded_M, stream);

#if not CUDA_MALLOC_ASYNC
  checkCudaErrors(cudaStreamSynchronize(stream));
#endif

  checkCudaErrors(cudaPooledFree(padded_A));
  checkCudaErrors(cudaPooledFree(padded_B));
  checkCudaErrors(cudaPooledFree(padded_C));
}

void gemm_stream(const float_16 *A,
                 const float_16 *B,
                 float_32 *C,
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


