#include <gtest/gtest.h>
#include "simulator.hpp"

TEST(gpu_simulator, memory)
{
    Sim::GlobalMemory mem;
    Sim::RegisterFile reg_file;
    Sim::PRegisterFile preg_file;

    Sim::GPUSimulator sim(mem, reg_file, preg_file);

    const float eps = 1e-6f;
    float *h_ptr = nullptr, *d_ptr = nullptr;
    sim.cudaMalloc((void **)&d_ptr, 100 * sizeof(float));
    d_ptr[50] = 2.0f;
    h_ptr = new float[100];
    h_ptr[50] = 1.0f;

    sim.cudaMemcpy(d_ptr, h_ptr, 100 * sizeof(float),
                   Sim::CUDAMemcpyType::MemcpyHostToDevice);
    EXPECT_NEAR(d_ptr[50], 1.0f, eps);

    d_ptr[50] = 2.0f;
    sim.cudaMemcpy(h_ptr, d_ptr, 100 * sizeof(float),
                   Sim::CUDAMemcpyType::MemcpyDeviceToHost);
    EXPECT_NEAR(h_ptr[50], 2.0f, eps);

    int error_count = 0;
    try
    {
        sim.cudaMemcpy(d_ptr, h_ptr, 100 * sizeof(float),
                       Sim::CUDAMemcpyType::MemcpyDeviceToHost);
    }
    catch (Sim::FatalError &e)
    {
        error_count++;
    }
    try
    {
        sim.cudaMemcpy(h_ptr, d_ptr, 100 * sizeof(float),
                       Sim::CUDAMemcpyType::MemcpyHostToDevice);
    }
    catch (Sim::FatalError &e)
    {
        error_count++;
    }
    EXPECT_EQ(error_count, 2);

    sim.cudaFree(d_ptr);
    EXPECT_FALSE(mem.isAllocated(d_ptr));
    delete[] h_ptr;
}

TEST(gpu_simulator, launch_kernel)
{
    Sim::GlobalMemory mem;
    Sim::RegisterFile reg_file;
    Sim::PRegisterFile preg_file;

    Sim::GPUSimulator sim(mem, reg_file, preg_file);

    auto dummy_kernel = [](Sim::GPUSimulator &sim, const Sim::dim3 &block_dim,
                           const std::array<Sim::unit3, Sim::GPUSimulator::WARP_SIZE> &warp,
                           float *matrix_ptr)
    {
        for (const auto &thread_idx : warp)
        {
            int idx = thread_idx.x + thread_idx.y * block_dim.x;
            matrix_ptr[idx] = 1.0f;
        }
    };

    float h_matrix[16][16] = {0};
    float *d_matrix = nullptr;
    sim.cudaMalloc((void **)&d_matrix, 16 * 16 * sizeof(float));
    Sim::dim3 block_dim(16, 16);
    sim.launchKernel(block_dim, dummy_kernel, d_matrix);
    sim.cudaMemcpy(h_matrix, d_matrix, 16 * 16 * sizeof(float),
                   Sim::CUDAMemcpyType::MemcpyDeviceToHost);

    int error_count = 0;
    try
    {
        sim.launchKernel(block_dim, dummy_kernel, (float *)h_matrix);
    }
    catch (Sim::FatalError &e)
    {
        error_count++;
    }
    EXPECT_EQ(error_count, 1);

    const float eps = 1e-6f;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            EXPECT_NEAR(h_matrix[i][j], 1.0f, eps);
}

TEST(gpu_simulator, instructions)
{
    // TODO
}