#pragma once

#include "common.h"

namespace Sim
{

    // NOTE: HACK: Constant Memory and Shared Memory are not implemented yet

    class GlobalMemory
    {
        friend class GPUSimulator;

    private:
        // (address, # bytes)
        std::unordered_map<intptr_t, u64> allocated_addrs;

    public:
        GlobalMemory() = default;

        bool isAllocated(void *ptr)
        {
            auto ptr_addr = reinterpret_cast<intptr_t>(ptr);
            return allocated_addrs.count(ptr_addr) > 0;
        }

        void cudaMalloc(void **ptr, size_t size)
        {
            *ptr = new char[size];
            allocated_addrs[reinterpret_cast<intptr_t>(*ptr)] = size;
        }

        void cudaFree(void *ptr)
        {
            printCppError(!isAllocated(ptr),
                          "Wrong address for cudaFree",
                          __FILE__, __LINE__);
            delete[] reinterpret_cast<char *>(ptr);
            allocated_addrs.erase(reinterpret_cast<intptr_t>(ptr));
        }

        // byte-level read
        std::vector<u8> read(void *ptr, u64 size)
        {
            // HACK: no check for ptr
            //  (e.g., int a[10][10] in kernel function, and this is not logged)
            std::vector<u8> ret(size);
            std::copy((char *)ptr, (char *)ptr + size, (char *)ret.data());
            return ret;
        }

        // byte-level write
        void write(void *ptr, const std::vector<u8> &data)
        {
            std::copy((char *)data.data(), (char *)data.data() + data.size(), (char *)ptr);
        }
    };

}
