#pragma once

#include "sim_common.h"
#include "half.h"
#include "device_memory.hpp"
#include "reg_file.h"

namespace Sim
{

    struct dim3
    {
        u32 x, y, z;
        constexpr dim3(u32 x = 1,
                       u32 y = 1,
                       u32 z = 1)
            : x(x), y(y), z(z) {}
    };

    using unit3 = dim3;

    class GPUSimulator
    {
    public:
        static const u32 WARP_SIZE = 32;

        using ThreadWarp = std::array<u32, WARP_SIZE>;

    private:
        // threadgroup utilities
        static const u32 THREAD_PER_GROUP = 4;

        std::array<f16, 1 * 4> loadReg(u32 thread_num, u32 R);

    private:
        GlobalMemory &global_memory;
        RegisterFile &reg_file;
        PRegisterFile &preg_file;
        SRegisterFile &sreg_file;

    public:
        // frankly speaking, DeviceMemory & Register should be internal modules of GPUSimulator
        //  but make them external can reduce the difficulty of writing test cases
        GPUSimulator(GlobalMemory &global_memory,
                     RegisterFile &reg_file, PRegisterFile &preg_file, SRegisterFile &sreg_file)
            : global_memory(global_memory),
              reg_file(reg_file), preg_file(preg_file), sreg_file(sreg_file) {}

        // SASS instructions
        // NOTE: HACK: DO NOT support BRANCH & SIMT STACK instructions
        void LDG_INSTR(const GPUSimulator::ThreadWarp &warp, u32 n_bits, u32 Rd, u32 Ra, u64 imm);
        void STG_INSTR(const GPUSimulator::ThreadWarp &warp, u32 n_bits, u32 Rd, u32 Ra, u64 imm);
        void HMMA_INSTR_STEP0(const GPUSimulator::ThreadWarp &warp, u32 Rd, u32 Ra, u32 Rb, u32 Rc);
        void HMMA_INSTR_STEP1(const GPUSimulator::ThreadWarp &warp, u32 Rd, u32 Ra, u32 Rb, u32 Rc);
        void HMMA_INSTR_STEP2(const GPUSimulator::ThreadWarp &warp, u32 Rd, u32 Ra, u32 Rb, u32 Rc);
        void HMMA_INSTR_STEP3(const GPUSimulator::ThreadWarp &warp, u32 Rd, u32 Ra, u32 Rb, u32 Rc);
        void S2R_INSTR(const GPUSimulator::ThreadWarp &warp, u32 Rd, u32 Sb);
        void IMAD_INSTR(const GPUSimulator::ThreadWarp &warp, bool wide, u32 Rd, u64 Ra, u64 Sb, u64 Sc, u32 options);
        void LOP3_INSTR(const GPUSimulator::ThreadWarp &warp, u32 Rd, u64 Ra, u64 Sb, u64 Sc, u32 imm, u32 options);
        void SHF_INSTR(const GPUSimulator::ThreadWarp &warp, bool left, u32 Rd, u32 Ra, u32 Sb, u32 Sc);
        void LEA_INSTR(const GPUSimulator::ThreadWarp &warp, bool hi, bool x,
                       u32 Rd, u32 Ra, u32 Sb, u32 Sc, u32 options, u32 imm, u32 Pd0, u32 Ps0);
        void EXIT_INSTR(const GPUSimulator::ThreadWarp &warp);

        // Launch kernel
        // NOTE: to simplify the simulator, we assume one SM is all-mighty
        //     that is, we fix the grid_dim to (1, 1, 1)
        template <typename Fn, typename... Args>
        void launchKernel(const dim3 &block_dim,
                          Fn &&kernel_func, Args &&...args)
        {
            // NOTE: memory validation
            // clang-format off
            ([&]{
                // HACK: FIXME: to avoid T *&x case
                printCppError(std::is_reference<Args>::value,
                              "DO NOT support reference type for kernel function arguments",
                              __FILE__, __LINE__);
                if (std::is_pointer<Args>::value)
                    printCppError(!global_memory.isAllocated((void *)args),
                                  "Wrong address for kernel function arguments",
                                  __FILE__, __LINE__);
            }(), ...);
            // clang-format on

            // TODO: move this to EXIT_INSTR
            reg_file.reset(), preg_file.reset(), sreg_file.reset();

            // HACK: FIXME: to simplify the thread-to-warp partition,
            //   we only support block_dim.x * block_dim.y * block_dim.z % WARP_SIZE == 0 case
            printCppError(block_dim.x * block_dim.y * block_dim.z % WARP_SIZE != 0,
                          "# threads in a block must be a multiple of WARP_SIZE",
                          __FILE__, __LINE__);
            // NOTE: wmma instruction requires co-operation from all threads in a warp.

            auto thread_num_count = 0;
            std::vector<u32> thread_num_list;
            thread_num_list.reserve(block_dim.x * block_dim.y * block_dim.z);
            for (u32 i = 0; i < block_dim.x; i++)
                for (u32 j = 0; j < block_dim.y; j++)
                    for (u32 k = 0; k < block_dim.z; k++)
                    {
                        auto thread_num = thread_num_count++;
                        thread_num_list.push_back(thread_num);

                        // NOTE: initialize special registers
                        sreg_file.write(thread_num, static_cast<u32>(SRegisterType::SR_LANEID), thread_num % WARP_SIZE, true);
                        sreg_file.write(thread_num, static_cast<u32>(SRegisterType::SR_TID_X), i, true);
                        sreg_file.write(thread_num, static_cast<u32>(SRegisterType::SR_TID_Y), j, true);
                        sreg_file.write(thread_num, static_cast<u32>(SRegisterType::SR_TID_Z), k, true);
                        sreg_file.write(thread_num, static_cast<u32>(SRegisterType::SR_NTID_X), block_dim.x, true);
                        sreg_file.write(thread_num, static_cast<u32>(SRegisterType::SR_NTID_Y), block_dim.y, true);
                        sreg_file.write(thread_num, static_cast<u32>(SRegisterType::SR_NTID_Z), block_dim.z, true);
                    }

            ThreadWarp warp;
            for (u32 i = 0; i < thread_num_list.size(); i += WARP_SIZE)
            {
                std::copy(thread_num_list.begin() + i,
                          thread_num_list.begin() + i + WARP_SIZE,
                          warp.begin());
                kernel_func(*this, warp, std::forward<Args>(args)...);
            }
        }

        std::tuple<unit3, dim3> readThreadInfo(u32 thread_num);

        // Memory management
        void cudaMalloc(void **ptr, size_t size);
        void cudaMemcpy(void *dst, void *src, size_t count, CUDAMemcpyType type);
        void cudaFree(void *ptr);
    };

}
