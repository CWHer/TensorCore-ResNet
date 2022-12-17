#pragma once

#include "common.h"

namespace Sim
{

    // NOTE: HACK: Constant Memory and Shared Memory are not implemented yet

    class GlobalMemory
    {
        friend class GPUSimulator;

    private:
        std::unordered_set<intptr_t> allocated_addrs;

    public:
        GlobalMemory() = default;

        void cudaMalloc(void **ptr, size_t size)
        {
            *ptr = new u8[size];
            allocated_addrs.insert(reinterpret_cast<intptr_t>(*ptr));
        }

        void cudaFree(void *ptr)
        {
            delete[] ptr;
            allocated_addrs.erase(reinterpret_cast<intptr_t>(ptr));
        }

        bool isAllocated(void *ptr)
        {
            return allocated_addrs.count(reinterpret_cast<intptr_t>(ptr));
        }
    };

}
