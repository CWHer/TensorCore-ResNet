#include <gtest/gtest.h>
#include "functions.h"

void naiveMatmul(const float *a, const float *b, float *c,
                 int n, int m, int k)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        {
            float sum = 0;
            for (int l = 0; l < k; l++)
                sum += a[i * k + l] * b[l * m + j];
            c[i * m + j] = sum;
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
            b[i * m + j] = i + j * k;

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < k; j++)
    //         std::cout << a[i * k + j] << ' ';
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < k; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //         std::cout << b[i * m + j] << ' ';
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    Sim::hostGEMM(a, b, c, n, m, k, sim);
    naiveMatmul(a, b, std_c, n, m, k);

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //         std::cout << std_c[i * m + j] << ' ';
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    const float eps = 1e-5;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            EXPECT_NEAR(c[i * m + j], std_c[i * m + j], eps);
}

TEST(functions, gemm_irregular)
{
    Sim::GlobalMemory mem;
    Sim::RegisterFile reg_file;
    Sim::PRegisterFile preg_file;
    Sim::SRegisterFile sreg_file;

    Sim::GPUSimulator sim(mem, reg_file, preg_file, sreg_file);

    int n = 171, m = 131, k = 91;
    Sim::f32 a[n * k], b[k * m], c[n * m], std_c[n * m];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            a[i * k + j] = std::sin(i * 2.0 + j * 3.0);
    for (int i = 0; i < k; i++)
        for (int j = 0; j < m; j++)
            b[i * m + j] = std::cos(i * 3.0 + j * 2.0);

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < k; j++)
    //         std::cout << a[i * k + j] << ' ';
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < k; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //         std::cout << b[i * m + j] << ' ';
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    Sim::hostGEMM(a, b, c, n, m, k, sim);
    naiveMatmul(a, b, std_c, n, m, k);

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //         std::cout << std_c[i * m + j] << ' ';
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    const float eps = 1e-2;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            EXPECT_NEAR(c[i * m + j], std_c[i * m + j], eps);
}