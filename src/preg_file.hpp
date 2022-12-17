#pragma once

#include "common.h"

namespace Sim
{

    class PRegisterFile
    {
    private:
        // NOTE: R7 is PT, which is always true
        static const u32 MAX_REG_NUM_PER = 8;
        // std::unordered_map<u32, std::vector<>> reg_file;
    };

}