# Simple GPU Simulator

## Description

- [ ] PTX & Compilation Trajectory

- [ ] CUDA Memory Model

- [ ] SASS

- [ ] Tensor Core

    As the underlying design of tensor core has not been publicly described by NVIDIA, we mainly followed the proposed architectural model of [*Modeling Deep Learning Accelerator Enabled GPUs*](https://arxiv.org/abs/1811.08309) and implemented a simplified version of it.

- [ ] Half

## Tutorial

- [x] cuobjdump

    ```bash
    nvcc -cubin gemm.cu -arch sm_70
    cuobjdump -sass gemm.cubin
    ```

- [ ] NVBit

## Getting Started

- [ ] How to build