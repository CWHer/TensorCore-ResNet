/** @file gemm.cu
*/
#include <mma.h>
#include <cublas_v2.h>
#include "common.h"

using namespace nvcuda;

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

template<int block_col_warps, int block_row_warps>
void gemm_naive_caller(const float_16 *A, const float_16 *B, float_32 *Result, size_t M, size_t N, size_t K) {
  constexpr int tile_m = block_row_warps * volta_m_factor;
  constexpr int tile_n = block_col_warps * volta_n_factor;

  dim3 grid((M + (tile_m - 1)) / tile_m, (N + (tile_n - 1)) / tile_n);
  dim3 block(block_row_warps * warp_size, block_col_warps);

  check_cuda_error();

  gemm_naive_kernel<block_col_warps, block_row_warps><<<grid, block>>>(A, B, Result, M, N, K);

  check_cuda_error();
}

void gemm_naive(const float_16 *A, const float_16 *B, float_32 *Result, size_t M, size_t N, size_t K) {
  float_16 *d_A, *d_B;
  float_32 *d_C;

  // If M and N are not by 16, we need to pad them.
  auto padded_M = (M + (volta_m_factor - 1)) / volta_m_factor * volta_m_factor;
  auto padded_N = (N + (volta_n_factor - 1)) / volta_n_factor * volta_n_factor;

  cudaMalloc(&d_A, padded_M * K * sizeof(float_16));
  cudaMalloc(&d_B, K * padded_N * sizeof(float_16));
  cudaMalloc(&d_C, padded_M * padded_N * sizeof(float_32));

  // Copy A and B by padding to device
  // A needs padding
  if (padded_M != M) {
    cudaMemset(d_A, 0, padded_M * K * sizeof(float_16));
    for (int i = 0; i < K; i++) {
      cudaMemcpy(d_A + i * padded_M, A + i * M, M * sizeof(float_16), cudaMemcpyHostToDevice);
    }
  } else {
    cudaMemcpy(d_A, A, M * K * sizeof(float_16), cudaMemcpyHostToDevice);
  }
  // B have same leading dimension
  cudaMemcpy(d_B, B, K * N * sizeof(float_16), cudaMemcpyHostToDevice);


  // Fixme: this template parameter is adjustable
  gemm_naive_caller<4, 4>(d_A, d_B, d_C, padded_M, padded_N, K);

  if (padded_N == N && padded_M == M) {
    cudaMemcpy(Result, d_C, M * N * sizeof(float_32), cudaMemcpyDeviceToHost);
  } else {
    for (int i = 0; i < N; i++) {
      cudaMemcpy(Result + i * M, d_C + i * padded_M, M * sizeof(float_32), cudaMemcpyDeviceToHost);
    }
  }
  check_cuda_error();

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  check_cuda_error();
}
