#pragma once

#include "common.h"
#include "half.hpp"
#include "device_memory.hpp"
#include "reg_file.hpp"

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

        std::tuple<unit3, dim3> readThreadInfo(u32 thread_num)
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