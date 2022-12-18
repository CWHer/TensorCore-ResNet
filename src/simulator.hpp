#pragma once

#include "common.h"
#include "device_memory.hpp"
#include "reg_file.hpp"

namespace Sim
{

    // TODO: What is this?
    enum s_reg_t
    {
        SRZ = 0,
        SR_LAINID,
        SR_TID_X,
        SR_TID_Y,
        SR_CTAID_X,
        SR_CTAID_Y
    };

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

    private:
        GlobalMemory &global_memory;
        RegisterFile &reg_file;
        PRegisterFile &preg_file;

    public:
        GPUSimulator(GlobalMemory &global_memory,
                     RegisterFile &reg_file, PRegisterFile &preg_file)
            : global_memory(global_memory),
              reg_file(reg_file), preg_file(preg_file) {}

        // SASS instructions
        // NOTE: HACK: DO NOT support BRANCH & SIMT STACK instructions
        void LDG_INSTR();
        void STG_INSTR();
        void HMMA_INSTR_STEP0();
        void HMMA_INSTR_STEP1();
        void HMMA_INSTR_STEP2();
        void HMMA_INSTR_STEP3();
        void S2R_INSTR();
        void IMAD_INSTR();
        void LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                        unsigned imm);
        void SHF_INSTR();
        void CS2R_INSTR();
        void LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                       unsigned imm, unsigned Pd0 = 7, unsigned Ps0 = 7);
        void EXIT_INSTR();

        // Launch kernel
        // NOTE: to simplify the simulator, we assume one SM is all-mighty
        //     that is, we fix the grid_dim to (1, 1, 1)
        template <typename Fn, typename... Args>
        void launchKernel(const dim3 &block_dim,
                          Fn &&kernel_func, Args &&...args)
        {
            // NOTE: memory validation
            ([&]
             {
                 if (std::is_pointer<Args>::value)
                     printCppError(!global_memory.isAllocated(args),
                                   "Wrong address for kernel function arguments",
                                   __FILE__, __LINE__); }(),
             ...);

            reg_file.reset(), preg_file.reset();

            std::vector<unit3> thread_idx_list;
            thread_idx_list.reserve(block_dim.x * block_dim.y * block_dim.z);
            for (u32 i = 0; i < block_dim.x; i++)
                for (u32 j = 0; j < block_dim.y; j++)
                    for (u32 k = 0; k < block_dim.z; k++)
                        thread_idx_list.emplace_back(i, j, k);

            std::array<unit3, WARP_SIZE> warp;
            for (u32 i = 0; i < thread_idx_list.size(); i += WARP_SIZE)
            {
                std::copy(thread_idx_list.begin() + i,
                          thread_idx_list.begin() + i + WARP_SIZE,
                          warp.begin());
                kernel_func(*this, block_dim, warp, std::forward<Args>(args)...);
            }
        }

        // Memory management
        void cudaMalloc(void **ptr, size_t size)
        {
            global_memory.cudaMalloc(ptr, size);
        }

        void cudaMemcpy(void *dst, void *src, size_t count, CUDAMemcpyType type)
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

        template <typename T>
        void cudaFree(T *ptr)
        {
            global_memory.cudaFree(ptr);
        }
    };

}