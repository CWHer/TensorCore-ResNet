# Simple GPU Simulator

## Description

1. We simulate simplified `global memory`, `register`, `predicate register` and `special register` 

2. We implement `fp16` type with proper rounding mechanism

3. We simulate several SASS instructions, such as `LDG`, `HMMA_STEP`, `IMAD`, ……

4. We use these simulated SASS instructions to compose a GEMM kernel using wmma API

   ```c++
   // NOTE: C = A * B.T
   __global__ void wmma_kernel(half *a, half *b, float *c)
   {
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
   
       wmma::fill_fragment(c_frag, 0.0f);
       wmma::load_matrix_sync(a_frag, a, 16);
       wmma::load_matrix_sync(b_frag, b, 16);
       wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
       wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
   }
   ```

5. We add a wrapper function `hostGEMM` of `wmma_kernel` to handle any shape input and calculate $\bold C = \bold A\times \bold B$.




## Getting Started

- Build and run unit tests

  ```bash
  mkdir build && cd build
  cmake ..
  make
  ./gtest_exc
  ```

- Usage of API

  ```c++
  Sim::GlobalMemory mem;
  Sim::RegisterFile reg_file;
  Sim::PRegisterFile preg_file;
  Sim::SRegisterFile sreg_file;
  // make GPUSimulator
  Sim::GPUSimulator sim(mem, reg_file, preg_file, sreg_file);
  // NOTE: C = A * B (support any shape input)
  Sim::hostGEMM(a, b, c, n, m, k, sim);
  ```

  

## Help

- [x] PTX & SASS

  The device code is first compiled into a device-independent machine-language instruction set architecture known as Parallel Thread eXecution (PTX) before being compiled into device-specific machine code (SASS).

- [x] Tensor Core

  As the underlying design of tensor core has not been publicly described by NVIDIA, we mainly followed the proposed behavioral model of [*Modeling Deep Learning Accelerator Enabled GPUs*](https://arxiv.org/abs/1811.08309) and implemented a ~~very~~ simplified version of it.

- [x] Half: 1 bit sign, 5 bit exponent, 10 bit fraction

- [x] `cuobjdump`

  ```bash
  nvcc -cubin gemm.cu -arch sm_70
  cuobjdump -sass gemm.cubin
  ```

- [x] CUDA GDB (similar to GDB)

  ```bash
  cuda-gdb gemm
  (cuda-gdb) b wmma_kernel(__half*, __half*, float*)
  (cuda-gdb) r
  (cuda-gdb) display /10i $pc
  (cuda-gdb) info registers
  ```
  
  By filling input matrix with different numbers, we can figure out the behavior of threads in each `HMMA` instruction.

