#include "mem_cache.h"

void freeMemCache()
{
    for (auto &kv : mem_cache)
        for (auto &ptr : kv.second)
            checkCudaErrors(cudaFree(ptr));
}

cudaError_t cudaCacheMalloc(void **ptr, size_t size)
{
    if (mem_cache.count(size) > 0 && !mem_cache[size].empty())
    {
        *ptr = mem_cache[size].front();
        mem_cache[size].pop_front();
        return cudaSuccess;
    }
    else
    {
        auto err = cudaMalloc(ptr, size);
        ptr_size[*ptr] = size;
        return err;
    }
}

cudaError_t cudaCacheFree(void *ptr)
{
    size_t size = ptr_size[ptr];
    mem_cache[size].push_back(ptr);
    return cudaSuccess;
}