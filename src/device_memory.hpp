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
            auto ptr_addr = reinterpret_cast<intptr_t>(ptr);
            for (auto &addr : allocated_addrs)
            {
                if (addr.first <= ptr_addr &&
                    ptr_addr < addr.first + addr.second)
                {
                    printCppError(ptr_addr - addr.first + size > addr.second,
                                  "Wrong size for read", __FILE__, __LINE__);
                    std::vector<u8> ret(size);
                    std::copy((char *)ptr, (char *)ptr + size, (char *)ret.data());
                    return ret;
                }
            }
            printCppError(true, "Wrong address for read", __FILE__, __LINE__);
        }

        // byte-level write
        void write(void *ptr, const std::vector<u8> &data)
        {
            auto ptr_addr = reinterpret_cast<intptr_t>(ptr);
            for (auto &addr : allocated_addrs)
            {
                if (addr.first <= ptr_addr &&
                    ptr_addr < addr.first + addr.second)
                {
                    printCppError(ptr_addr - addr.first + data.size() > addr.second,
                                  "Wrong size for write", __FILE__, __LINE__);
                    std::copy((char *)data.data(), (char *)data.data() + data.size(), (char *)ptr);
                    return;
                }
            }
            printCppError(true, "Wrong address for write", __FILE__, __LINE__);
        }
    };

}
