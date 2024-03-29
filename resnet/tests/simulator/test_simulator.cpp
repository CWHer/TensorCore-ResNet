#include <gtest/gtest.h>
#include "simulator.hpp"

TEST(gpu_simulator, memory) {
  Sim::GlobalMemory mem;
  Sim::RegisterFile reg_file;
  Sim::PRegisterFile preg_file;
  Sim::SRegisterFile sreg_file;

  Sim::GPUSimulator sim(mem, reg_file, preg_file, sreg_file);

  const float eps = 1e-6f;
  float *h_ptr = nullptr, *d_ptr = nullptr;
  sim.cudaMalloc((void **) &d_ptr, 100 * sizeof(float));
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

  EXPECT_THROW(sim.cudaMemcpy(d_ptr, h_ptr, 100 * sizeof(float),
                              Sim::CUDAMemcpyType::MemcpyDeviceToHost), Sim::FatalError);
  EXPECT_THROW(sim.cudaMemcpy(h_ptr, d_ptr, 100 * sizeof(float),
                              Sim::CUDAMemcpyType::MemcpyHostToDevice), Sim::FatalError);

  sim.cudaFree(d_ptr);
  EXPECT_FALSE(mem.isAllocated(d_ptr));
  delete[] h_ptr;
}

TEST(gpu_simulator, launch_kernel) {
  Sim::GlobalMemory mem;
  Sim::RegisterFile reg_file;
  Sim::PRegisterFile preg_file;
  Sim::SRegisterFile sreg_file;

  Sim::GPUSimulator sim(mem, reg_file, preg_file, sreg_file);

  auto dummy_kernel = [](Sim::GPUSimulator &sim,
                         const Sim::GPUSimulator::ThreadWarp &warp,
                         float *matrix_ptr) {
    for (const auto &thread_num : warp) {
      Sim::unit3 thread_idx;
      Sim::dim3 block_dim;
      std::tie(thread_idx, block_dim) = sim.readThreadInfo(thread_num);
      auto idx = thread_idx.x + thread_idx.y * block_dim.x;
      matrix_ptr[idx] = 1.0f;
    }
  };

  float h_matrix[16][16] = {0};
  float *d_matrix = nullptr;
  sim.cudaMalloc((void **) &d_matrix, 16 * 16 * sizeof(float));
  Sim::dim3 block_dim(16, 16);
  sim.launchKernel(block_dim, dummy_kernel, (float *) d_matrix);
  sim.cudaMemcpy(h_matrix, d_matrix, 16 * 16 * sizeof(float),
                 Sim::CUDAMemcpyType::MemcpyDeviceToHost);
  sim.cudaFree(d_matrix);

  // HACK: will not throw due to old target compiler
#if __GNUC_PREREQ(8, 2)
  EXPECT_THROW(sim.launchKernel(block_dim, dummy_kernel, (float *) d_matrix), Sim::FatalError);
#endif

  const float eps = 1e-6f;
  for (auto &i : h_matrix)
    for (float j : i)
      EXPECT_NEAR(j, 1.0f, eps);
}
