#include <gtest/gtest.h>
#include "functions.h"

// NOTE: C = A * B.T
void naiveMatmul(const float *a, const float *b, float *c,
                 int n, int m, int k)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            c[i * m + j] = 0;
            for (int l = 0; l < k; l++)
                c[i * m + j] += a[i * k + l] * b[l + m * j];
        }
}

TEST(functions, gemm)
{
    Sim::GlobalMemory mem;
    Sim::RegisterFile reg_file;
    Sim::PRegisterFile preg_file;
    Sim::SRegisterFile sreg_file;

    Sim::GPUSimulator sim(mem, reg_file, preg_file, sreg_file);

    int n = 16, m = 16, k = 16;
    Sim::f32 a[n * k], b[k * m], c[n * m], std_c[n * m];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            a[i * k + j] = i * k + j;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++)
            b[i + j * m] = i + j * m;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
            std::cout << a[i * k + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < m; j++)
            std::cout << b[i + j * m] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;

    Sim::host_gemm(a, b, c, n, m, k, sim);
    naiveMatmul(a, b, std_c, n, m, k);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
            std::cout << std_c[i * k + j] << ' ';
        std::cout << std::endl;
    }
    std::cout << std::endl;

    const float eps = 1e-5;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            EXPECT_NEAR(c[i * m + j], std_c[i * m + j], eps);
}

// TEST(functions, gemm_irregular)
// {
//     Sim::GlobalMemory mem;
//     Sim::RegisterFile reg_file;
//     Sim::PRegisterFile preg_file;
//     Sim::SRegisterFile sreg_file;

//     Sim::GPUSimulator sim(mem, reg_file, preg_file, sreg_file);

//     // int n = 171, m = 131, k = 91;
//     int n = 16, m = 16, k = 16;
//     Sim::f32 a[n * k], b[k * m], c[n * m], std_c[n * m];

//     for (int i = 0; i < 16; i++)
//         for (int j = 0; j < 16; j++)
//         {
//             a[i * 16 + j] = i * 16 + j;
//             b[i + j * 16] = i + j * 16;
//         }
//     for (int i = 0; i < 16; i++)
//         for (int j = 0; j < 16; j++)
//             std::cout << a[i * 16 + j]
//                       << (j == 15 ? '\n' : ' ');
//     std::cout << std::endl;
//     for (int i = 0; i < 16; i++)
//         for (int j = 0; j < 16; j++)
//             std::cout << b[i + j * 16]
//                       << (j == 15 ? '\n' : ' ');
//     std::cout << std::endl;

//     Sim::host_gemm(a, b, c, n, m, k, sim);
//     naiveMatmul(a, b, std_c, n, m, k);

//     const float eps = 1e-5;
//     for (int i = 0; i < n; i++)
//         for (int j = 0; j < m; j++)
//             EXPECT_NEAR(c[i * m + j], std_c[i * m + j], eps);
// }