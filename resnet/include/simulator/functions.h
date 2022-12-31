#pragma once

#include "sim_common.h"
#include "simulator.h"

namespace Sim
{

    void wmmaKernel(GPUSimulator &sim,
                    const GPUSimulator::ThreadWarp &warp,
                    f16 *a, f16 *b, f32 *c);

    void deviceGEMM(GPUSimulator &sim,
                    const GPUSimulator::ThreadWarp &warp,
                    const f32 *a, const f32 *b, f32 *c, u32 n, u32 m, u32 k);

    void hostGEMM(const f32 *a, const f32 *b, f32 *c,
                  u32 n, u32 m, u32 k, GPUSimulator &sim);

}
