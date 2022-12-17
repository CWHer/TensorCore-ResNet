#pragma once

#include "common.h"
#include "device_memory.hpp"
#include "reg_file.hpp"
#include "preg_file.hpp"

namespace Sim
{

    // What is this?
    enum s_reg_t
    {
        SRZ = 0,
        SR_LAINID,
        SR_TID_X,
        SR_TID_Y,
        SR_CTAID_X,
        SR_CTAID_Y
    };

    struct Dim3
    {
        unsigned int x, y, z;
        constexpr Dim3(unsigned int vx = 1,
                       unsigned int vy = 1,
                       unsigned int vz = 1)
            : x(vx), y(vy), z(vz) {}
    };

    class GPUSimulator
    {
    private:
        static const u32 WARP_SIZE = 32;

        GlobalMemory &global_memory;
        RegisterFile &reg_file;
        PRegisterFile &preg_file;

    public:
        GPUSimulator(GlobalMemory &global_memory,
                     RegisterFile &reg_file, PRegisterFile &preg_file)
            : global_memory(global_memory),
              reg_file(reg_file), preg_file(preg_file) {}

        // SASS instructions
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
                printCppError(global_memory.isAllocated(src),
                              "Wrong address for cudaMemcpy",
                              __FILE__, __LINE__);
                break;
            case CUDAMemcpyType::MemcpyHostToDevice:
                printCppError(global_memory.isAllocated(dst),
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