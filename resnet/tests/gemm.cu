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

static void gemm_CPU_reference_row_major(const float *A, const float *B, float *Result, size_t M, size_t N, size_t K) {
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

static void gemm_CPU_reference_col_major(const float *A, const float *B, float *Result, size_t M, size_t N, size_t K) {
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

static void gemm_cuBLAS_reference_col_major(const float *A,
                                            const float *B,
                                            float *Result,
                                            size_t M,
                                            size_t N,
                                            size_t K) {
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

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ArgumentSelectionDefects"
static void gemm_cuBLAS_reference_row_major(const float *A,
                                            const float *B,
                                            float *Result,
                                            size_t M,
                                            size_t N,
                                            size_t K) {
  return gemm_cuBLAS_reference_col_major(B, A, Result, N, M, K);
}
#pragma clang diagnostic pop

typedef void (*gemm32_func)(const float *A, const float *B, float *Result, size_t M, size_t N, size_t K);
typedef void (*gemm16_func)(const float_16 *A, const float_16 *B, float_32 *Result, size_t M, size_t N, size_t K);

static void func_test_32(size_t M, size_t N, size_t K, gemm32_func func1, gemm32_func func2, float eps = 5.0) {
  // Test if we entered the correct cuBLAS parameters
  auto *A = new float[M * K];
  auto *B = new float[K * N];

  // Randomly initialize A, B
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < M * K; i++) {
    A[i] = matrix_dist(generator);
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = matrix_dist(generator);
  }

  // Create float matrix C
  auto *C = new float[M * N];
  auto *C_ref = new float[M * N];

  // Compute C = A * B
  func1(A, B, C, M, N, K);
  func2(A, B, C_ref, M, N, K);

  // Check if C is correct
  for (int i = 0; i < M * N; i++) {
    if (fabs(C[i] - C_ref[i]) > eps) {
      throw runtime_error("Incorrect result");
    }
  }

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C_ref;
}

static void func_test_16(size_t M, size_t N, size_t K, gemm16_func gemm16, gemm32_func gemm32, float eps = 5.0) {
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
  auto *C_ref = new float[M * N];

  // Compute C = A * B
  gemm16(A_float_16, B_float_16, C, M, N, K);
  gemm32(A, B, C_ref, M, N, K);

  // Check if C is correct
  for (int i = 0; i < M * N; i++) {
    if (fabs(C[i] - C_ref[i]) > eps) {
      float excess = fabs(C[i] - C_ref[i]);
      char buffer[100];
      sprintf(buffer, "Incorrect result: %f vs %f at %d, excess: %f", C[i], C_ref[i], i, excess);

      throw runtime_error(buffer);
    }
  }
  delete[] A;
  delete[] B;
  delete[] A_float_16;
  delete[] B_float_16;
  delete[] C;
  delete[] C_ref;
}

TEST(gemm_col_major, cuBLAS_param_correctness) {
  ASSERT_NO_THROW(func_test_32(12, 34, 56, gemm_cuBLAS_reference_col_major, gemm_CPU_reference_col_major));
}

TEST(gemm_col_major, gemm_col_major_square) {
  ASSERT_NO_THROW(func_test_16(256, 256, 256, gemm_col_major, gemm_cuBLAS_reference_col_major));
}

TEST(gemm_col_major, gemm_col_major_rectangular) {
  ASSERT_NO_THROW(func_test_16(256, 512, 1024, gemm_col_major, gemm_cuBLAS_reference_col_major));
}

TEST(gemm_col_major, gemm_col_major_irregular) {
  ASSERT_NO_THROW(func_test_16(17, 513, 1029, gemm_col_major, gemm_cuBLAS_reference_col_major));
}

TEST(gemm_row_major, cuBLAS_param_correctness) {
  ASSERT_NO_THROW(func_test_32(12, 34, 56, gemm_cuBLAS_reference_row_major, gemm_CPU_reference_row_major));
}

TEST(gemm_row_major, gemm_row_major_square) {
  ASSERT_NO_THROW(func_test_16(256, 256, 256, gemm_row_major, gemm_cuBLAS_reference_row_major));
}

TEST(gemm_row_major, gemm_row_major_rectangular) {
  ASSERT_NO_THROW(func_test_16(256, 512, 1024, gemm_row_major, gemm_cuBLAS_reference_row_major));
}

TEST(gemm_row_major, gemm_row_major_irregular) {
  ASSERT_NO_THROW(func_test_16(17, 513, 1029, gemm_row_major, gemm_cuBLAS_reference_row_major));
}

