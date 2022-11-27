/**
 * @file test/gemm.cu
 * @brief Test GEMM functionality.
 */

#include <gtest/gtest.h>
#include "functional/gemm.hpp"
#include <random>
#include <cublas_v2.h>

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

using namespace std;

__attribute((unused)) static void gemm_CPU_reference_row_major(const float *A,
                                                               const float *B,
                                                               float *Result,
                                                               size_t M,
                                                               size_t N,
                                                               size_t K) {
// This is row-major
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      Result[i * N + j] = sum;
    }
  }
}

static void gemm_CPU_reference(const float *A, const float *B, float *Result, size_t M, size_t N, size_t K) {
  // This is column-major
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum += A[k * M + i] * B[j * K + k];
      }
      Result[j * M + i] = sum;
    }
  }
}

void gemm_cuBLAS_reference(const float *A, const float *B, float *Result, size_t M, size_t N, size_t K) {
  float *d_A, *d_B, *d_Result;
  auto start = std::chrono::high_resolution_clock::now();
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_Result, M * N * sizeof(float));

  cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasSgemm(handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              (int) M,
              (int) N,
              (int) K,
              &alpha,
              d_A,
              (int) M,
              d_B,
              (int) K,
              &beta,
              d_Result,
              (int) M);

  cublasDestroy(handle);
  cudaMemcpy(Result, d_Result, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_Result);

}

TEST(gemm, test_gemm_cuBLAS_param) {
  // Test if we entered the correct cuBLAS parameters
  auto *A = new float[12 * 34];
  auto *B = new float[34 * 56];

  // Randomly initialize A, B
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < 12 * 34; i++) {
    A[i] = matrix_dist(generator);
  }
  for (int i = 0; i < 34 * 56; i++) {
    B[i] = matrix_dist(generator);
  }

  // Create float matrix C
  auto *C = new float[12 * 56];
  auto *C_cublas = new float[12 * 56];

  // Compute C = A * B
  gemm_CPU_reference(A, B, C, 12, 56, 34);
  gemm_cuBLAS_reference(A, B, C_cublas, 12, 56, 34);

  // Check if C is correct
  for (int i = 0; i < 12 * 56; i++) {
    EXPECT_NEAR(C[i], C_cublas[i], 1.0f);
  }

}

TEST(gemm, test_gemm_naive_basic) {
  // Create float matrices A, B
  auto *A = new float[256 * 256];
  auto *B = new float[256 * 256];
  auto *A_float_16 = new float_16[256 * 256];
  auto *B_float_16 = new float_16[256 * 256];

  // Randomly initialize A, B
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < 256 * 256; i++) {
    // Make sure the matrix have the same float
    float_16 a_float_16 = __float2half(matrix_dist(generator));
    float_16 b_float_16 = __float2half(matrix_dist(generator));
    A[i] = __half2float(a_float_16);
    B[i] = __half2float(b_float_16);
    A_float_16[i] = a_float_16;
    B_float_16[i] = b_float_16;
  }

  // Create float matrix C
  auto *C = new float[256 * 256];
  auto *C_cublas = new float[256 * 256];

  // Compute C = A * B
  gemm_naive(A_float_16, B_float_16, C, 256, 256, 256);
  gemm_cuBLAS_reference(A, B, C_cublas, 256, 256, 256);

  // Check if C is correct
  for (int i = 0; i < 256 * 256; i++) {
    ASSERT_NEAR(C[i], C_cublas[i], 1.0f);
  }
}

TEST(gemm, test_gemm_naive_rectangular) {
  // Create float matrices A, B
  auto *A = new float[256 * 64];
  auto *B = new float[64 * 256];
  auto *A_float_16 = new float_16[256 * 64];
  auto *B_float_16 = new float_16[64 * 256];

  // Randomly initialize A, B
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < 64 * 256; i++) {
    // Make sure the matrix have the same float
    float_16 a_float_16 = __float2half(matrix_dist(generator));
    float_16 b_float_16 = __float2half(matrix_dist(generator));

    A[i] = __half2float(a_float_16);
    B[i] = __half2float(b_float_16);

    A_float_16[i] = a_float_16;
    B_float_16[i] = b_float_16;
  }

  // Create float matrix C
  auto *C = new float[256 * 256];
  auto *C_cublas = new float[256 * 256];

  // Compute C = A * B
  gemm_naive(A_float_16, B_float_16, C, 256, 256, 64);
  gemm_cuBLAS_reference(A, B, C_cublas, 256, 256, 64);

  // Check if C is correct
  for (int i = 0; i < 256 * 256; i++) {
    ASSERT_NEAR(C[i], C_cublas[i], 1.0f);
  }

  delete[] A;
  delete[] B;
  delete[] A_float_16;
  delete[] B_float_16;
  delete[] C;
  delete[] C_cublas;

}

TEST(gemm, test_gemm_naive_irregular) {
  const int M = 7;
  const int N = 5;
  const int K = 3;
  // Create float matrices A, B
  auto *A = new float[M * K];
  auto *B = new float[K * N];
  auto *A_float_16 = new float_16[M * K];
  auto *B_float_16 = new float_16[K * N];

  // Randomly initialize A, B
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < M * K; i++) {
    // Make sure the matrix have the same float
    float_16 a_float_16 = __float2half(matrix_dist(generator));
    A[i] = __half2float(a_float_16);
    A_float_16[i] = a_float_16;
  }
  for (int i = 0; i < K * N; i++) {
    float_16 b_float_16 = __float2half(matrix_dist(generator));
    B[i] = __half2float(b_float_16);
    B_float_16[i] = b_float_16;
  }

  // Create float matrix C
  auto *C = new float[M * N];
  auto *C_cublas = new float[M * N];

  // Compute C = A * B
  gemm_naive(A_float_16, B_float_16, C, M, N, K);
  gemm_cuBLAS_reference(A, B, C_cublas, M, N, K);

  // Check if C is correct
  for (int i = 0; i < M * N; i++) {
    ASSERT_NEAR(C[i], C_cublas[i], 1.0f);
  }
  delete[] A;
  delete[] B;
  delete[] A_float_16;
  delete[] B_float_16;
  delete[] C;
  delete[] C_cublas;

}

