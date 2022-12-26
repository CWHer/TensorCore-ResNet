#pragma once

#include "common.h"

namespace Sim
{

    template <typename T, u32 MAX_REG_NUM_PER, bool read_only>
    class RegisterFileBase
    {
    private:
        static const std::unordered_map<u32, T> SPECIAL_REG;
        std::unordered_map<u32, std::array<T, MAX_REG_NUM_PER>> reg_file;

    public:
        RegisterFileBase() { reset(); }

        void reset() { reg_file.clear(); }

        void write(u32 thread_num, u32 reg_idx, T value, bool force = false)
        {
            printCppError(read_only && !force,
                          "Write to read-only register file",
                          __FILE__, __LINE__);
            if (SPECIAL_REG.count(reg_idx) == 0)
                reg_file[thread_num][reg_idx] = value;
        }

        T read(u32 thread_num, u32 reg_idx)
        {
            return SPECIAL_REG.count(reg_idx) == 0
                       ? reg_file[thread_num][reg_idx]
                       : SPECIAL_REG.at(reg_idx);
        }
    };

    // predicated register file
    using PRegisterFile = RegisterFileBase<bool, 8, false>;

    // register file
    using RegisterFile = RegisterFileBase<u32, 256, false>;

    // special register file
    enum SRegisterType
    {
        // NOTE:
        //  Cooperative thread arrays (CTAs) implement CUDA thread blocks
        //  Clusters implement CUDA thread block clusters.

        // A predefined, read-only special register
        //  that returns the thread's lane within the warp.
        // The lane identifier ranges from zero to WARP_SZ-1.
        SR_LANEID,

        // A predefined, read-only special register
        //  that returns the thread's warp identifier.
        // The warp identifier provides a unique warp number within a CTA
        //  but not across CTAs within a grid.
        // SR_WARPID,

        // A predefined, read-only, per-thread special register
        //  initialized with the thread identifier within the CTA.
        SR_TID_X,
        SR_TID_Y,
        SR_TID_Z,
        // A predefined, read-only special register
        //  initialized with the number of thread ids in each CTA dimension.
        SR_NTID_X,
        SR_NTID_Y,
        SR_NTID_Z,

        // HACK: we fix the grid_dim to (1, 1, 1)
        // A predefined, read-only special register
        //  initialized  with the CTA identifier within the CTA grid.
        // SR_CTAID_X,
        // SR_CTAID_Y,
        // SR_CTAID_Z,

        NUM_SREG
    };
    using SRegisterFile = RegisterFileBase<u32, SRegisterType::NUM_SREG, true>;

}