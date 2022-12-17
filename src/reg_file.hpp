#pragma once

#include "common.h"

namespace Sim
{

    class RegisterFile
    {
    private:
        // NOTE: R255 is RZ, which is always 0
        static const u32 MAX_REG_NUM_PER = 256;
        // std::unordered_map<u32, std::vector<>> reg_file;
    };

}