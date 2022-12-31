/**
 * @file test/gemm.cpp
 * @brief Test GEMM functionality.
 */

#include <gtest/gtest.h>
#include "functional/gemm.h"

static void checkLastCudaError()
{
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

static void gemmOnHost(const float *a, const float *b, float *result,
                       size_t m, size_t n, size_t k,
                       MatrixMajor major)
{
    switch (major)
    {
    case MatrixMajor::ColMajor:
        // This is column-major
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++)
            {
                float sum = 0;
                for (size_t l = 0; l < k; l++)
                    sum += a[l * m + i] * b[j * k + l];
                result[j * m + i] = sum;
            }
        break;
    case MatrixMajor::RowMajor:
        // This is row-major
        for (size_t i = 0; i < m; i++)
            for (size_t j = 0; j < n; j++)
            {
                float sum = 0;
                for (size_t l = 0; l < k; l++)
                    sum += a[i * k + l] * b[l * n + j];
                result[i * n + j] = sum;
            }
        break;
    }
}

typedef decltype(gemmOnHost) gemm32Func;
typedef decltype(gemm) gemm16Func;

static void funcTest(size_t m, size_t n, size_t k,
                     gemm16Func gemm16, gemm32Func gemm32,
                     MatrixMajor major, float eps = 1e-3)
{
    // Create float matrices A, B
    auto a = std::make_unique<float[]>(m * k);
    auto b = std::make_unique<float[]>(k * n);
    auto a_f16 = std::make_unique<f16[]>(m * k);
    auto b_f16 = std::make_unique<f16[]>(k * n);

    // Randomly initialize A, B
    for (int i = 0; i < m * k; i++)
    {
        // Make sure the matrix have the same float
        f16 h = __float2half(randomFloat(-1.0e2, 1.0e2));
        a[i] = __half2float(h);
        a_f16[i] = h;
    }
    for (int i = 0; i < k * n; i++)
    {
        f16 h = __float2half(randomFloat(-1.0e2, 1.0e2));
        b[i] = __half2float(h);
        b_f16[i] = h;
    }

    auto c = std::make_unique<float[]>(m * n);
    auto c_ref = std::make_unique<float[]>(m * n);

    f16 *d_a_f16, *d_b_f16;
    float *d_c;
    checkCudaErrors(cudaMalloc(&d_a_f16, m * k * sizeof(f16)));
    checkCudaErrors(cudaMalloc(&d_b_f16, k * n * sizeof(f16)));
    checkCudaErrors(cudaMalloc(&d_c, m * n * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_a_f16, a_f16.get(), m * k * sizeof(f16), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b_f16, b_f16.get(), k * n * sizeof(f16), cudaMemcpyHostToDevice));

    gemm32(a.get(), b.get(), c_ref.get(), m, n, k, major);
    gemm16(d_a_f16, d_b_f16, d_c, m, n, k, major);
    checkLastCudaError();

    checkCudaErrors(cudaMemcpy(c.get(), d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_a_f16));
    checkCudaErrors(cudaFree(d_b_f16));
    checkCudaErrors(cudaFree(d_c));

    // for (int i = 0; i < 10; i++)
    //     std::cout << c[i] << " ";
    // std::cout << std::endl;
    // for (int i = 0; i < 10; i++)
    //     std::cout << c_ref[i] << " ";
    // std::cout << std::endl;

    // Check if C is correct
    double average_diff = 0;
    for (int i = 0; i < m * n; i++)
    {
        auto diff_ratio = c_ref[i] != 0 ? fabs(c[i] - c_ref[i]) / fabs(c_ref[i]) : 0;
        average_diff += diff_ratio / (float)(m * n);
    }

    if (average_diff > eps)
    {
        std::stringstream ss;
        ss << "Incorrect result: " << average_diff << " > " << eps << std::endl;
        throw std::runtime_error(ss.str());
    }
}

static void unequal_B(int m, int n, int k, int batch_size)
{
    auto eps = 5e-3;

    // Test if we entered the correct cuBLAS parameters
    auto a = std::make_unique<f16[]>(m * k);
    auto b = std::make_unique<f16[]>(batch_size * k * n);

    // Randomly initialize A, B
    for (int i = 0; i < m * k; i++)
        a[i] = randomFloat(-1.0e2, 1.0e2);
    for (int i = 0; i < batch_size * k * n; i++)
        b[i] = randomFloat(-1.0e2, 1.0e2);

    // Create float matrix C
    auto c = std::make_unique<float[]>(batch_size * m * n);
    auto c_ref = std::make_unique<float[]>(batch_size * m * n);

    f16 *d_a, *d_b;
    float *d_c, *d_c_ref;

    checkCudaErrors(cudaMalloc(&d_a, m * k * sizeof(f16)));
    checkCudaErrors(cudaMalloc(&d_b, batch_size * k * n * sizeof(f16)));
    checkCudaErrors(cudaMalloc(&d_c, batch_size * m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c_ref, batch_size * m * n * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_a, a.get(), m * k * sizeof(f16), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b.get(), batch_size * k * n * sizeof(f16), cudaMemcpyHostToDevice));

    gemmBatched(d_a, d_b, d_c, m, n, k, batch_size, MatrixMajor::RowMajor);
    checkLastCudaError();

    checkCudaErrors(cudaMemcpy(c.get(), d_c, batch_size * m * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_c));

    for (int t = 0; t < batch_size; t++)
    {
        gemm(d_a, d_b + t * k * n, d_c_ref + t * m * n,
             m, n, k, MatrixMajor::RowMajor);
        checkLastCudaError();
        checkCudaErrors(cudaMemcpy(c_ref.get() + t * m * n, d_c_ref + t * m * n,
                                   m * n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c_ref));

    // Check if C is correct
    double average_diff = 0;
    for (int i = 0; i < batch_size * m * n; i++)
    {
        auto diff_ratio = c_ref[i] != 0 ? fabs(c[i] - c_ref[i]) / fabs(c_ref[i]) : 0;
        average_diff += diff_ratio / (float)(batch_size * m * n);
    }

    if (average_diff > eps)
    {
        std::stringstream ss;
        ss << "Incorrect result: " << average_diff << " > " << eps << std::endl;
        throw std::runtime_error(ss.str());
    }
}

TEST(gemm, small_matrix)
{
    ASSERT_NO_THROW(funcTest(16, 16, 16, gemm, gemmOnHost, MatrixMajor::RowMajor));
    ASSERT_NO_THROW(funcTest(16, 16, 16, gemm, gemmOnHost, MatrixMajor::ColMajor));
}

TEST(gemm, square_matrix)
{
    ASSERT_NO_THROW(funcTest(256, 256, 256, gemm, gemmOnHost, MatrixMajor::ColMajor));
    ASSERT_NO_THROW(funcTest(256, 256, 256, gemm, gemmOnHost, MatrixMajor::RowMajor));
}

TEST(gemm, rectangular_matrix)
{
    ASSERT_NO_THROW(funcTest(256, 512, 1024, gemm, gemmOnHost, MatrixMajor::ColMajor));
    ASSERT_NO_THROW(funcTest(256, 512, 1024, gemm, gemmOnHost, MatrixMajor::RowMajor));
}

TEST(gemm, irregular_matrix)
{
    ASSERT_NO_THROW(funcTest(17, 513, 1029, gemm, gemmOnHost, MatrixMajor::ColMajor));
    ASSERT_NO_THROW(funcTest(17, 513, 1029, gemm, gemmOnHost, MatrixMajor::RowMajor));
}

TEST(gemm, unaligned_pointer)
{
    int m = 17, n = 513, k = 1029;
    auto eps = 1e-3;

    // GEMM should support non-aligned pointers
    // Create float matrices A_loc, B_loc
    auto a_loc = std::make_unique<f16[]>(m * k);
    auto b_loc = std::make_unique<f16[]>(k * n);
    auto a_f16 = std::make_unique<f16[]>(m * k + 1);
    auto b_f16 = std::make_unique<f16[]>(k * n + 3);

    // Randomly initialize A_loc, B_loc
    for (int i = 0; i < m * k; i++)
    {
        // Make sure the matrix have the same float
        f16 h = __float2half(randomFloat(-1.0e2, 1.0e2));
        a_loc[i] = __half2float(h);
        a_f16[i + 1] = h;
    }
    for (int i = 0; i < k * n; i++)
    {
        f16 h = __float2half(randomFloat(-1.0e2, 1.0e2));
        b_loc[i] = __half2float(h);
        b_f16[i + 3] = h;
    }

    auto c = std::make_unique<float[]>(m * n + 7);
    auto c_ref = std::make_unique<float[]>(m * n);

    f16 *d_a_loc, *d_b_loc;
    f16 *d_a_f16, *d_b_f16;
    float *d_c, *d_c_ref;
    checkCppErrors(cudaMalloc(&d_a_loc, m * k * sizeof(f16)));
    checkCppErrors(cudaMalloc(&d_b_loc, k * n * sizeof(f16)));
    checkCppErrors(cudaMalloc(&d_c_ref, m * n * sizeof(float)));
    checkCppErrors(cudaMalloc(&d_a_f16, (m * k + 1) * sizeof(f16)));
    checkCppErrors(cudaMalloc(&d_b_f16, (k * n + 3) * sizeof(f16)));
    checkCppErrors(cudaMalloc(&d_c, (m * n + 7) * sizeof(float)));

    checkCppErrors(cudaMemcpy(d_a_loc, a_loc.get(), m * k * sizeof(f16), cudaMemcpyHostToDevice));
    checkCppErrors(cudaMemcpy(d_b_loc, b_loc.get(), k * n * sizeof(f16), cudaMemcpyHostToDevice));
    checkCppErrors(cudaMemcpy(d_a_f16 + 1, a_f16.get() + 1, m * k * sizeof(f16), cudaMemcpyHostToDevice));
    checkCppErrors(cudaMemcpy(d_b_f16 + 3, b_f16.get() + 3, k * n * sizeof(f16), cudaMemcpyHostToDevice));

    gemm(d_a_loc, d_b_loc, d_c_ref, m, n, k, MatrixMajor::RowMajor);
    gemm(d_a_f16 + 1, d_b_f16 + 3, d_c + 7, m, n, k, MatrixMajor::RowMajor);

    checkCppErrors(cudaMemcpy(c_ref.get(), d_c_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCppErrors(cudaMemcpy(c.get() + 7, d_c + 7, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    checkCppErrors(cudaFree(d_a_loc));
    checkCppErrors(cudaFree(d_b_loc));
    checkCppErrors(cudaFree(d_c_ref));
    checkCppErrors(cudaFree(d_a_f16));
    checkCppErrors(cudaFree(d_b_f16));
    checkCppErrors(cudaFree(d_c));

    // Check if C is correct
    double average_diff = 0;
    for (int i = 0; i < m * n; i++)
    {
        auto diff_ratio = c_ref[i] != 0 ? fabs(c[i + 7] - c_ref[i]) / fabs(c_ref[i]) : 0;
        average_diff += diff_ratio / (float)(m * n);
    }

    if (average_diff > eps)
    {
        std::stringstream ss;
        ss << "Incorrect result: " << average_diff << " > " << eps << std::endl;
        throw std::runtime_error(ss.str());
    }
}

TEST(gemm, batched_b)
{
    EXPECT_NO_THROW(unequal_B(16, 16, 16, 1));
    EXPECT_NO_THROW(unequal_B(16, 16, 16, 16));
    EXPECT_NO_THROW(unequal_B(16, 16, 17, 16));
    EXPECT_NO_THROW(unequal_B(16, 17, 16, 16));
    EXPECT_NO_THROW(unequal_B(17, 16, 16, 16));
    EXPECT_NO_THROW(unequal_B(16, 17, 17, 16));
    EXPECT_NO_THROW(unequal_B(17, 17, 16, 16));
    EXPECT_NO_THROW(unequal_B(17, 16, 17, 16));
    EXPECT_NO_THROW(unequal_B(17, 17, 17, 16));

    for (int i = 0; i < 10; ++i)
    {
        EXPECT_NO_THROW(unequal_B(17, 1007, 29, 15));
        EXPECT_NO_THROW(unequal_B(1331, 31, 13, 27));
        EXPECT_NO_THROW(unequal_B(17, 513, 411, 15));
        EXPECT_NO_THROW(unequal_B(171, 513, 129, 41));
    }
}
