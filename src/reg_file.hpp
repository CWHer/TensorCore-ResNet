#pragma once

#include "common.h"

namespace Sim
{

    template <typename T, u32 MAX_REG_NUM_PER>
    class RegisterFileBase
    {
    private:
        static const std::unordered_map<u32, T> SPECIAL_REG;
        std::unordered_map<u32, std::array<T, MAX_REG_NUM_PER>> reg_file;

    public:
        RegisterFileBase() { reset(); }

        void reset() { reg_file.clear(); }

        void write(u32 thread_idx, u32 reg_idx, T value)
        {
            if (SPECIAL_REG.count(reg_idx) == 0)
            {
                reg_file[thread_idx][reg_idx] = value;
            }
        }

        T read(u32 thread_idx, u32 reg_idx)
        {
            return SPECIAL_REG.count(reg_idx) == 0
                       ? reg_file[thread_idx][reg_idx]
                       : SPECIAL_REG.at(reg_idx);
        }
    };

    // clang-format off
    // NOTE: R7 is PT, which is always true
    using PRegisterFile = RegisterFileBase<bool, 8>;
    template <> const std::unordered_map<u32, bool> PRegisterFile::SPECIAL_REG = {{7, true}};

    // NOTE: R255 is RZ, which is always 0
    using RegisterFile = RegisterFileBase<u32, 256>;
    template <> const std::unordered_map<u32, u32> RegisterFile::SPECIAL_REG = {{255, 0}};
    // clang-format on

}