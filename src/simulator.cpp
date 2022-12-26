#include "simulator.h"

namespace Sim
{

    std::array<f16, 1 * 4> GPUSimulator::loadReg(u32 thread_num, u32 R)
    {
        std::array<f16, 1 * 4> t;
        for (int i = 0; i < 2; i++)
        {
            u32 value = reg_file.read(thread_num, R + i);
            t[i << 1] = f16(value & 0xffff);
            t[i << 1 | 1] = f16(value >> 16);
        }
        return t;
    }

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
                                 u32 n_bits, u32 Rd, u32 Ra, u64 imm)
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

    void GPUSimulator::HMMA_INSTR_STEP0(const GPUSimulator::ThreadWarp &warp,
                                        u32 Rd, u32 Ra, u32 Rb, u32 Rc)
    {
        // Since the only documentation of HMMA is "Modeling Deep Learning Accelerator Enabled GPUs",
        //  and it only describes the behavior of threadgroup not thread.
        // Thus, we would like to implement this instruction in threadgroup level.
        // (p.s., threadgroup is a group of four consecutive threads in a warp)
        // (p.s., in HMMA instruction a thread may read other threads' register)

        for (int t = 0; t < WARP_SIZE; t += THREAD_PER_GROUP)
        {
            std::vector<u32> thread_nums;
            for (int i = 0; i < THREAD_PER_GROUP; i++)
                thread_nums.push_back(warp[t + i]);

            std::array<f16, 2 * 4> a;
            std::array<f16, 4 * 4> b;
            std::array<f32, 2 * 4> c;
            for (int i = 0; i < 2; ++i)
            {
                auto t = loadReg(thread_nums[i], Ra);
                std::copy(t.begin(), t.end(), a.begin() + i * 4);
            }
            for (int i = 0; i < 4; ++i)
            {
                auto t = loadReg(thread_nums[i], Rb);
                std::copy(t.begin(), t.end(), b.begin() + i * 4);
            }
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j)
                    c[i * 2 + j] = reg_file.read(thread_nums[i], Rc + j);

            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                {
                    f32 sum = 0;
                    for (int k = 0; k < 4; ++k)
                        sum += a[i * 4 + k] * b[k * 4 + j];
                    c[i * 4 + j] += sum;
                }

            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j)
                    reg_file.write(thread_nums[i], Rd + j, c[i * 2 + j]);
        }
    }

    void GPUSimulator::HMMA_INSTR_STEP1(const GPUSimulator::ThreadWarp &warp,
                                        u32 Rd, u32 Ra, u32 Rb, u32 Rc)
    {
    }

    void GPUSimulator::HMMA_INSTR_STEP2(const GPUSimulator::ThreadWarp &warp,
                                        u32 Rd, u32 Ra, u32 Rb, u32 Rc)
    {
    }

    void GPUSimulator::HMMA_INSTR_STEP3(const GPUSimulator::ThreadWarp &warp,
                                        u32 Rd, u32 Ra, u32 Rb, u32 Rc)
    {
    }

    void GPUSimulator::S2R_INSTR(const GPUSimulator::ThreadWarp &warp, u32 Rd, u32 Sb)
    {
        for (const auto &thread_num : warp)
        {
            u32 value = sreg_file.read(thread_num, Sb);
            reg_file.write(thread_num, Rd, value);
        }
    }

    void GPUSimulator::IMAD_INSTR(const GPUSimulator::ThreadWarp &warp, bool wide,
                                  u32 Rd, u64 Ra, u64 Sb, u64 Sc, u32 options)
    {
        // 9.7.1.4. Integer Arithmetic Instructions: mad
        // https://stackoverflow.com/questions/59777333/combined-format-of-sass-instructions

        // options: [Ra, Sb, Sc]
        //  e.g., 0b110 means Ra and Sb are immediate values
        for (const auto &thread_num : warp)
        {
            Sc = !(options & 0x1) ? reg_file.read(thread_num, Sc) +
                                        wide * ((u64)reg_file.read(thread_num, Sc + 1) << 32)
                                  : Sc;
            Sb = !(options & 0x2) ? reg_file.read(thread_num, Sb) : Sb;
            Ra = !(options & 0x4) ? reg_file.read(thread_num, Ra) : Ra;

            // clang-format off
            u64 value = Ra * Sb + Sc;
            reg_file.write(thread_num, Rd, value & 0xffffffff);
            if (wide) reg_file.write(thread_num, Rd + 1, value >> 32);
            // clang-format on
        }
    }

    void GPUSimulator::LOP3_INSTR(const GPUSimulator::ThreadWarp &warp,
                                  u32 Rd, u64 Ra, u64 Sb, u64 Sc, u32 imm, u32 options)
    {
        // 9.7.7.6. Logic and Shift Instructions: lop3

        // options: [Ra, Sb, Sc]
        //  e.g., 0b110 means Ra and Sb are immediate values
        for (const auto &thread_num : warp)
        {
            Sc = !(options & 0x1) ? reg_file.read(thread_num, Sc) : Sc;
            Sb = !(options & 0x2) ? reg_file.read(thread_num, Sb) : Sb;
            Ra = !(options & 0x4) ? reg_file.read(thread_num, Ra) : Ra;
            // clang-format off
            u32 result = 0;
            if (imm & 0x01) result |= (~Ra) & (~Sb) & (~Sc);
            if (imm & 0x02) result |= (~Ra) & (~Sb) & (Sc);
            if (imm & 0x04) result |= (~Ra) & (Sb) & (~Sc);
            if (imm & 0x08) result |= (~Ra) & (Sb) & (Sc);
            if (imm & 0x10) result |= (Ra) & (~Sb) & (~Sc);
            if (imm & 0x20) result |= (Ra) & (~Sb) & (Sc);
            if (imm & 0x40) result |= (Ra) & (Sb) & (~Sc);
            if (imm & 0x80) result |= (Ra) & (Sb) & (Sc);
            reg_file.write(thread_num, Rd, result);
            // clang-format on
        }
    }

    void GPUSimulator::SHF_INSTR(const GPUSimulator::ThreadWarp &warp,
                                 bool left, u32 Rd, u32 Ra, u32 Sb, u32 Sc)
    {
        // 9.7.7.7. Logic and Shift Instructions: shf

        // HACK: DO NOT support .HI and I32
        for (const auto &thread_num : warp)
        {
            u64 value = reg_file.read(thread_num, Ra) +
                        ((u64)reg_file.read(thread_num, Sc) << 32);
            value = left ? value << Sb : value >> Sb;
            reg_file.write(thread_num, Rd, value & 0xffffffff);
        }
    }

    // void GPUSimulator::LEA_INSTR(const GPUSimulator::ThreadWarp &warp,
    //                              bool hi, bool x, u32 Rd, u32 Ra, u32 Sb, u32 Sc, u32 imm, u32 Pd0, u32 Ps0)
    // {
    //     for (const auto &thread_num : warp)
    //     {
    //         u64 value = reg_file.read(thread_num, Ra) +
    //                     ((u64)reg_file.read(thread_num, Sc) << 32);

    //         value = hi ? value >> (32 - imm) : value << imm;
    //         value += x * preg_file.read(thread_num, Ps0);
    //         preg_file.write(thread_num, Pd0, (value >> 32) & 0x1);
    //         reg_file.write(thread_num, Rd, value & 0xffffffff);
    //     }
    // }

    void GPUSimulator::EXIT_INSTR(const GPUSimulator::ThreadWarp &warp)
    {
        // HACK: noop
    }

    std::tuple<unit3, dim3> GPUSimulator::readThreadInfo(u32 thread_num)
    {
        return std::make_tuple(
            unit3(sreg_file.read(thread_num, static_cast<u32>(SRegisterType::SR_TID_X)),
                  sreg_file.read(thread_num, static_cast<u32>(SRegisterType::SR_TID_Y)),
                  sreg_file.read(thread_num, static_cast<u32>(SRegisterType::SR_TID_Z))),
            dim3(sreg_file.read(thread_num, static_cast<u32>(SRegisterType::SR_NTID_X)),
                 sreg_file.read(thread_num, static_cast<u32>(SRegisterType::SR_NTID_Y)),
                 sreg_file.read(thread_num, static_cast<u32>(SRegisterType::SR_NTID_Z))));
    }

    // Memory management
    void GPUSimulator::cudaMalloc(void **ptr, size_t size)
    {
        global_memory.cudaMalloc(ptr, size);
    }

    void GPUSimulator::cudaMemcpy(void *dst, void *src, size_t count, CUDAMemcpyType type)
    {
        switch (type)
        {
        case CUDAMemcpyType::MemcpyDeviceToHost:
            printCppError(!global_memory.isAllocated(src),
                          "Wrong address for cudaMemcpy",
                          __FILE__, __LINE__);
            break;
        case CUDAMemcpyType::MemcpyHostToDevice:
            printCppError(!global_memory.isAllocated(dst),
                          "Wrong address for cudaMemcpy",
                          __FILE__, __LINE__);
            break;
        default:
            printCppError(true, "Not implemented cudaMemcpy type",
                          __FILE__, __LINE__);
        }
        std::copy((char *)src, (char *)src + count, (char *)dst);
    }

    void GPUSimulator::cudaFree(void *ptr)
    {
        global_memory.cudaFree(ptr);
    }

}