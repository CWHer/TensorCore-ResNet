/**
 * @file test/gemm.cpp
 * @brief Test GEMM functionality.
 */

#include <gtest/gtest.h>
#include "functional/gemm.hpp"
#include <random>

#if WITH_CUBLAS
#include <cublas_v2.h>
#endif

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

using namespace std;

static void check_cuda_error() {
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

static void gemm_CPU_reference(const float *A,
                               const float *B,
                               float *Result,
                               size_t M,
                               size_t N,
                               size_t K,
                               GEMM::Major major,
                               const Impl::DeviceType device_type = Impl::DeviceType::CPU) {
  if (device_type != Impl::DeviceType::CPU) {
    throw std::runtime_error("CPU reference only.");
  }

  switch (major) {
  case GEMM::Major::col_major: {
// This is column-major
    for (size_t I = 0; I < M; I++) {
      for (size_t J = 0; J < N; J++) {
        float Sum = 0;
        for (size_t K2 = 0; K2 < K; K2++) {
          Sum += A[K2 * M + I] * B[J * K + K2];
        }
        Result[J * M + I] = Sum;
      }
    }
  }
    break;
  case GEMM::Major::row_major: {
// This is row-major
    for (size_t I = 0; I < M; I++) {
      for (size_t J = 0; J < N; J++) {
        float Sum = 0;
        for (size_t K2 = 0; K2 < K; K2++) {
          Sum += A[I * K + K2] * B[K2 * N + J];
        }
        Result[I * N + J] = Sum;
      }
    }
  }
    break;
  }

}

#if WITH_CUBLAS
#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
static void gemm_cuBLAS_reference(const float *A,
                                  const float *B,
                                  float *Result,
                                  size_t M,
                                  size_t N,
                                  size_t K,
                                  GEMM::Major major,
                                  const Impl::DeviceType device_type = Impl::DeviceType::CPU) {
  switch (device_type) {
  case Impl::DeviceType::CPU: {
    float *DA, *DB, *DResult;
    cudaMalloc(&DA, M * K * sizeof(float));
    cudaMalloc(&DB, K * N * sizeof(float));
    cudaMalloc(&DResult, M * N * sizeof(float));
    cudaMemcpy(DA, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    gemm_cuBLAS_reference(DA, DB, DResult, M, N, K, major, Impl::DeviceType::CUDA);

    cudaMemcpy(Result, DResult, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DResult);
    return;
  }
    break;
  case Impl::DeviceType::CUDA: break;
  }

  switch (major) {
  case GEMM::Major::col_major: {
    cublasHandle_t Handle;
    cublasCreate(&Handle);
    float Alpha = 1.0f;
    float Beta = 0.0f;

    cublasSgemm(Handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                (int) M,
                (int) N,
                (int) K,
                &Alpha,
                A,
                (int) M,
                B,
                (int) K,
                &Beta,
                Result,
                (int) M);

    cublasDestroy(Handle);
  }
    break;
  case GEMM::Major::row_major: {
    return gemm_cuBLAS_reference(B, A, Result, N, M, K, GEMM::Major::col_major, device_type);
  }
    break;
  }
}
#pragma clang diagnostic pop
#endif

typedef decltype(gemm_CPU_reference) gemm32_new_func;
typedef decltype(gemm) gemm16_new_func;

static void func_test(size_t M,
                      size_t N,
                      size_t K,
                      gemm32_new_func func1,
                      gemm32_new_func func2,
                      GEMM::Major major = GEMM::Major::row_major,
                      Impl::DeviceType device_type = Impl::DeviceType::CPU,
                      float eps = 1e-2) {
  // Test if we entered the correct cuBLAS parameters
  auto A = make_unique<float[]>(M * K);
  auto B = make_unique<float[]>(K * N);

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
  auto C = make_unique<float[]>(M * N);
  auto C_ref = make_unique<float[]>(M * N);

  switch (device_type) {
  case Impl::DeviceType::CPU: {
    func1(A.get(), B.get(), C_ref.get(), M, N, K, major, device_type);
    func2(A.get(), B.get(), C.get(), M, N, K, major, device_type);
  }
    break;
  case Impl::DeviceType::CUDA: {
    float *DA, *DB, *DC;
    cudaMalloc(&DA, M * K * sizeof(float));
    cudaMalloc(&DB, K * N * sizeof(float));
    cudaMalloc(&DC, M * N * sizeof(float));
    cudaMemcpy(DA, A.get(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B.get(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    func1(DA, DB, DC, M, N, K, major, device_type);
    cudaMemcpy(C_ref.get(), DC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    func2(DA, DB, DC, M, N, K, major, device_type);
    cudaMemcpy(C.get(), DC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
  }
    break;
  }

  // Check if C is correct
  double average_diff = 0;
  for (int i = 0; i < M * N; i++) {
    auto diff_ratio = C_ref[i] != 0 ? fabs(C[i] - C_ref[i]) / fabs(C_ref[i]) : 0;
    average_diff += diff_ratio / (float) (M * N);
  }

  if (average_diff > eps) {
    stringstream ss;
    ss << "Incorrect result: " << average_diff << " > " << eps << endl;
    throw runtime_error(ss.str());
  }
}

static void func_test(size_t M,
                      size_t N,
                      size_t K,
                      gemm16_new_func gemm16,
                      gemm32_new_func gemm32,
                      GEMM::Major major,
                      Impl::DeviceType device_type = Impl::DeviceType::CPU,
                      float eps = 1e-1) {
  // Create float matrices A, B
  auto A = make_unique<float[]>(M * K);
  auto B = make_unique<float[]>(K * N);
  auto A_float_16 = make_unique<float_16[]>(M * K);
  auto B_float_16 = make_unique<float_16[]>(K * N);


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

  auto C = make_unique<float[]>(M * N);
  auto C_ref = make_unique<float[]>(M * N);

  switch (device_type) {
  case Impl::DeviceType::CPU:gemm32(A.get(), B.get(), C_ref.get(), M, N, K, major, device_type);
    gemm16(A_float_16.get(), B_float_16.get(), C.get(), M, N, K, major, device_type);
    break;
  case Impl::DeviceType::CUDA: {
    float *DA, *DB, *DC, *DC_ref;
    float_16 *DA_fp16, *DB_fp16;
    cudaMalloc(&DA, M * K * sizeof(float));
    cudaMalloc(&DB, K * N * sizeof(float));
    cudaMalloc(&DA_fp16, M * K * sizeof(float_16));
    cudaMalloc(&DB_fp16, K * N * sizeof(float_16));

    cudaMalloc(&DC, M * N * sizeof(float));
    cudaMalloc(&DC_ref, M * N * sizeof(float));

    check_cuda_error();

    cudaMemcpy(DA, A.get(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B.get(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DA_fp16, A_float_16.get(), M * K * sizeof(float_16), cudaMemcpyHostToDevice);
    cudaMemcpy(DB_fp16, B_float_16.get(), K * N * sizeof(float_16), cudaMemcpyHostToDevice);

    check_cuda_error();

    gemm32(DA, DB, DC_ref, M, N, K, major, device_type);

    cudaMemcpy(C_ref.get(), DC_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda_error();
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC_ref);

    check_cuda_error();

    gemm16(DA_fp16, DB_fp16, DC, M, N, K, major, device_type);
    cudaMemcpy(C.get(), DC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    check_cuda_error();
    cudaFree(DA_fp16);
    cudaFree(DB_fp16);
    cudaFree(DC);

    check_cuda_error();
  }
    break;
  }

  // Check if C is correct
  double average_diff = 0;
  for (int i = 0; i < M * N; i++) {
    auto diff_ratio = C_ref[i] != 0 ? fabs(C[i] - C_ref[i]) / fabs(C_ref[i]) : 0;
    average_diff += diff_ratio / (float) (M * N);
  }

  if (average_diff > eps) {
    stringstream ss;
    ss << "Incorrect result: " << average_diff << " > " << eps << endl;
    throw runtime_error(ss.str());
  }
}

static void unequal_B(int M, int N, int K, int batch_size) {
  auto eps = 1e-1;

  // Test if we entered the correct cuBLAS parameters
  auto A = make_unique<float_16[]>(M * K);
  auto B = make_unique<float_16[]>(batch_size * K * N);

  // Randomly initialize A, B
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < M * K; i++) {
    A[i] = matrix_dist(generator);
  }
  for (int i = 0; i < batch_size * K * N; i++) {
    B[i] = matrix_dist(generator);
  }

  // Create float matrix C
  auto C = make_unique<float[]>(batch_size * M * N);
  auto C_ref = make_unique<float[]>(batch_size * M * N);

  gemm_batched_B(A.get(), B.get(), C.get(), M, N, K, batch_size, GEMM::Major::row_major, Impl::DeviceType::CPU);

  for (int batch = 0; batch < batch_size; batch++) {
    gemm(A.get(),
         B.get() + batch * K * N,
         C_ref.get() + batch * M * N,
         M,
         N,
         K,
         GEMM::Major::row_major,
         Impl::DeviceType::CPU);
  }

  // Check if C is correct
  double average_diff = 0;
  for (int i = 0; i < batch_size * M * N; i++) {
    auto diff_ratio = C_ref[i] != 0 ? fabs(C[i] - C_ref[i]) / fabs(C_ref[i]) : 0;
    average_diff += diff_ratio / (float) (batch_size * M * N);
  }

  if (average_diff > eps) {
    stringstream ss;
    ss << "Incorrect result: " << average_diff << " > " << eps << endl;
    throw runtime_error(ss.str());
  }
}

#if WITH_CUBLAS
TEST(gemm, cuBLAS_param_correctness) {
  ASSERT_NO_THROW(func_test(12,
                            34,
                            56,
                            gemm_cuBLAS_reference,
                            gemm_CPU_reference,
                            GEMM::Major::col_major,
                            Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(12,
                            34,
                            56,
                            gemm_cuBLAS_reference,
                            gemm_CPU_reference,
                            GEMM::Major::row_major,
                            Impl::DeviceType::CPU));
}
#endif

TEST(gemm, small_matrix_CPU) {
  ASSERT_NO_THROW(func_test(16, 16, 16, gemm, gemm_CPU_reference, GEMM::Major::row_major, Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(16, 16, 16, gemm, gemm_CPU_reference, GEMM::Major::col_major, Impl::DeviceType::CPU));
}

#if WITH_CUBLAS
TEST(gemm, square_matrix) {
  ASSERT_NO_THROW(func_test(256, 256, 256, gemm, gemm_cuBLAS_reference, GEMM::Major::col_major, Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(256, 256, 256, gemm, gemm_cuBLAS_reference, GEMM::Major::row_major, Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(256,
                            256,
                            256,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::col_major,
                            Impl::DeviceType::CUDA));
  ASSERT_NO_THROW(func_test(256,
                            256,
                            256,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::row_major,
                            Impl::DeviceType::CUDA));
}

TEST(gemm, rectangular_matrix) {
  ASSERT_NO_THROW(func_test(256,
                            512,
                            1024,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::col_major,
                            Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(256,
                            512,
                            1024,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::row_major,
                            Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(256,
                            512,
                            1024,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::col_major,
                            Impl::DeviceType::CUDA));
  ASSERT_NO_THROW(func_test(256,
                            512,
                            1024,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::row_major,
                            Impl::DeviceType::CUDA));
}

TEST(gemm, irregular_matrix) {
  ASSERT_NO_THROW(func_test(17, 513, 1029, gemm, gemm_cuBLAS_reference, GEMM::Major::col_major, Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(17, 513, 1029, gemm, gemm_cuBLAS_reference, GEMM::Major::row_major, Impl::DeviceType::CPU));
  ASSERT_NO_THROW(func_test(17,
                            513,
                            1029,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::col_major,
                            Impl::DeviceType::CUDA));
  ASSERT_NO_THROW(func_test(17,
                            513,
                            1029,
                            gemm,
                            gemm_cuBLAS_reference,
                            GEMM::Major::row_major,
                            Impl::DeviceType::CUDA));
}

TEST(gemm, unaligned_pointer) {
  int M = 17;
  int N = 513;
  int K = 1029;

  auto eps = 1e-2;

  auto major = GEMM::Major::row_major;
  auto device_type = Impl::DeviceType::CPU;

  auto gemm16 = gemm;

  // GEMM should support non-aligned pointers
  // Create float matrices A_loc, B_loc
  auto A_loc = make_unique<float_16[]>(M * K);
  auto B_loc = make_unique<float_16[]>(K * N);
  auto A_float_16 = make_unique<float_16[]>(M * K + 1);
  auto B_float_16 = make_unique<float_16[]>(K * N + 3);


  // Randomly initialize A_loc, B_loc
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  for (int i = 0; i < M * K; i++) {
    // Make sure the matrix have the same float
    float_16 a_float_16 = __float2half(matrix_dist(generator));
    A_loc[i] = __half2float(a_float_16);
    A_float_16[i + 1] = a_float_16;
  }
  for (int i = 0; i < K * N; i++) {
    float_16 b_float_16 = __float2half(matrix_dist(generator));
    B_loc[i] = __half2float(b_float_16);
    B_float_16[i + 3] = b_float_16;
  }

  auto C = make_unique<float[]>(M * N + 7);
  auto C_ref = make_unique<float[]>(M * N);

  gemm16(A_loc.get(), B_loc.get(), C_ref.get(), M, N, K, major, device_type);
  gemm16(A_float_16.get() + 1, B_float_16.get() + 3, C.get() + 7, M, N, K, major, device_type);

  // Check if C is correct
  double average_diff = 0;
  for (int i = 0; i < M * N; i++) {
    auto diff_ratio = C_ref[i] != 0 ? fabs(C[i + 7] - C_ref[i]) / fabs(C_ref[i]) : 0;
    average_diff += diff_ratio / (float) (M * N);
  }

  if (average_diff > eps) {
    stringstream ss;
    ss << "Incorrect result: " << average_diff << " > " << eps << endl;
    throw runtime_error(ss.str());
  }

}
#endif

TEST(gemm, batchedB) {
  EXPECT_NO_THROW(unequal_B(16, 16, 16, 1));
  EXPECT_NO_THROW(unequal_B(16, 16, 16, 16));
  EXPECT_NO_THROW(unequal_B(16, 16, 17, 16));
  EXPECT_NO_THROW(unequal_B(16, 17, 16, 16));
  EXPECT_NO_THROW(unequal_B(17, 16, 16, 16));
  EXPECT_NO_THROW(unequal_B(16, 17, 17, 16));
  EXPECT_NO_THROW(unequal_B(17, 17, 16, 16));
  EXPECT_NO_THROW(unequal_B(17, 16, 17, 16));
  EXPECT_NO_THROW(unequal_B(17, 17, 17, 16));

  EXPECT_NO_THROW(unequal_B(17, 513, 1029, 15));
  EXPECT_NO_THROW(unequal_B(17, 513, 1029, 27));
}
