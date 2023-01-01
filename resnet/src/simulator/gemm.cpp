/** @file gemm.cpp
*/

#include "functional/gemm.hpp"
#include "device_memory.hpp"
#include "reg_file.hpp"
#include "simulator.hpp"
#include "functions.hpp"
#include "mem_pool.h"

using namespace Impl;

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
void gemm_stream(const float_16 *A,
                 const float_16 *B,
                 float_32 *C,
                 size_t M,
                 size_t N,
                 size_t K,
                 const GEMM::Major major,
                 const Impl::DeviceType device_type,
                 cudaStream_t &stream [[gnu::unused]]) {
  if (major == GEMM::Major::col_major) {
    return gemm_stream(B, A, C, N, M, K, GEMM::Major::row_major, device_type, stream);
  }

  Sim::GlobalMemory mem;
  Sim::RegisterFile reg_file;
  Sim::PRegisterFile preg_file;
  Sim::SRegisterFile sreg_file;

  Sim::GPUSimulator sim(mem, reg_file, preg_file, sreg_file);

  float *A_host, *B_host, *C_host;

  if (device_type == Impl::DeviceType::CPU) {
    A_host = fp16_array_to_fp32_array(A, M * K, device_type);
    B_host = fp16_array_to_fp32_array(B, K * N, device_type);
    C_host = C;
  } else {
    float *A_dev, *B_dev;
    A_dev = fp16_array_to_fp32_array(A, M * K, device_type);
    B_dev = fp16_array_to_fp32_array(B, K * N, device_type);

    A_host = new float[M * K];
    B_host = new float[K * N];
    C_host = new float[M * N];

    cudaMemcpy(A_host, A_dev, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_host, B_dev, K * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaPooledFree(A_dev);
    cudaPooledFree(B_dev);
  }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ArgumentSelectionDefects"
  Sim::hostGEMM(A_host, B_host, C_host, M, N, K, sim);
#pragma clang diagnostic pop

  if (device_type == Impl::DeviceType::CPU) {
    delete[] A_host;
    delete[] B_host;
  } else {
    cudaMemcpy(C, C_host, M * N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] A_host;
    delete[] B_host;
    delete[] C_host;
  }

}
#pragma clang diagnostic pop

void gemm_batched_B(const float_16 *A,
                    const float_16 *B,
                    float_32 *C,
                    size_t M,
                    size_t N,
                    size_t K,
                    size_t batch_size,
                    GEMM::Major major,
                    Impl::DeviceType device_type) {
#if DEBUG
  if (major != GEMM::Major::row_major) {
    // Batches does not support col-major
    throw std::runtime_error("Batches does not support col-major");
  }
#endif

  constexpr int stream_count = 8;
  cudaStream_t streams[stream_count];
  for (auto &stream : streams) {
    cudaStreamCreate(&stream);
  }

  for (size_t i = 0; i < batch_size; i++) {
    auto B_ptr = B + i * K * N;
    auto C_ptr = C + i * M * N;
    gemm_stream(A, B_ptr, C_ptr, M, N, K, major, device_type, streams[i % stream_count]);
  }

  for (auto &stream : streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaCacheCommit(stream);
  }
}


