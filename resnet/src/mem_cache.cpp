#include "mem_cache.h"

void freeMemCache()
{
    for (auto &kv : mem_cache)
    {
        for (auto &ptr : kv.second)
            checkCudaErrors(cudaFree(ptr));
        kv.second.clear();
    }
    for (auto &kv : stream_ptr)
        checkCppErrorsMsg(!kv.second.empty(), "stream_ptr not empty");
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

cudaError_t cudaCacheMallocAsync(void **ptr, size_t size, cudaStream_t stream)
{
    if (mem_cache.count(size) > 0 && !mem_cache[size].empty())
    {
        *ptr = mem_cache[size].front();
        mem_cache[size].pop_front();
        return cudaSuccess;
    }
    else
    {
        // HACK: this is sync
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

cudaError_t cudaCacheFreeAsync(void *ptr, cudaStream_t stream)
{
    stream_ptr[stream].push_back(ptr);
    return cudaSuccess;
}

void cudaCacheCommit(cudaStream_t stream)
{
    for (auto &ptr : stream_ptr[stream])
        mem_cache[ptr_size[ptr]].push_back(ptr);
    stream_ptr[stream].clear();
}