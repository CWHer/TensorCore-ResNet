#pragma once

#include "simulator.hpp"

namespace Sim
{

    // SASS instructions
    // https://stackoverflow.com/questions/35055014/how-to-understand-the-result-of-sass-analysis-in-cuda-gpu
    // basic reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

    // NOTE: HACK: FIXME: these instructions are implemented without full functional support,
    //  we only validate their correctness by composing a simple gemm kernel

    void GPUSimulator::LDG_INSTR(const GPUSimulator::ThreadWarp &warp,
                                 u32 n_bits, u32 Rd, u32 Ra, u64 imm)
    {
        // 9.7.8.8. Data Movement and Conversion Instructions: ld

        printCppError(n_bits % 8 != 0 || n_bits % 32 != 0,
                      "Global memory is currently byte-addressable, and each register is 32-bit wide",
                      __FILE__, __LINE__);
        for (const auto &thread_num : warp)
        {
            // HACK: 64-bit address
            u64 addr = imm + reg_file.read(thread_num, Ra) +
                       ((u64)reg_file.read(thread_num, Ra + 1) << 32);
            std::vector<u8> values = global_memory.read(reinterpret_cast<void *>(addr), n_bits / 8);
            for (int i = 0; i < n_bits / 32; i++)
            {
                u32 value = 0;
                for (int j = 0; j < 4; j++)
                    value |= values[i * 4 + j] << (j * 8);
                reg_file.write(thread_num, Rd + i, value);
            }
        }
    }

    void GPUSimulator::STG_INSTR(const GPUSimulator::ThreadWarp &warp,
                                 u32 n_bits, u32 Rd, u64 imm, u32 Ra)
    {
        printCppError(n_bits % 8 != 0 || n_bits % 32 != 0,
                      "Global memory is currently byte-addressable, and each register is 32-bit wide",
                      __FILE__, __LINE__);
        for (const auto &thread_num : warp)
        {
            // HACK: 64-bit address
            u64 addr = imm + reg_file.read(thread_num, Rd) +
                       ((u64)reg_file.read(thread_num, Rd + 1) << 32);
            std::vector<u8> values;
            for (int i = 0; i < n_bits / 32; i++)
            {
                u32 value = reg_file.read(thread_num, Ra + i);
                for (int j = 0; j < 4; j++)
                    values.push_back((value >> (j * 8)) & 0xff);
            }
            global_memory.write(reinterpret_cast<void *>(addr), values);
        }
    }

    void GPUSimulator::S2R_INSTR(const GPUSimulator::ThreadWarp &warp, u32 Rd, u32 Sb)
    {
        for (const auto &thread_num : warp)
        {
            u32 value = sreg_file.read(thread_num, Sb);
            reg_file.write(thread_num, Rd, value);
        }
    }
    void GPUSimulator::EXIT_INSTR(const GPUSimulator::ThreadWarp &warp)
    {
        // HACK: noop
        exit_thread_count += warp.size();
    }
}