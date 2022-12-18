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

TEST(gpu_simulator, instructions)
{
    // TODO
}