#pragma once

#include "common.h"
#include "simulator.h"

namespace Sim
{

    void wmma_kernel(GPUSimulator &sim,
                     const GPUSimulator::ThreadWarp &warp,
                     f16 *a, f16 *b, f32 *c);

    void device_gemm(GPUSimulator &sim,
                     const GPUSimulator::ThreadWarp &warp,
                     const f32 *a, const f32 *b, f32 *c, int m, int n, int k);

    void host_gemm(const f32 *a, const f32 *b, f32 *c,
                   int m, int n, int k, GPUSimulator &sim);

}