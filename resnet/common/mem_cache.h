#pragma once

#include "common.h"

static std::unordered_map<uint64_t, std::deque<void *>> mem_cache;
static std::unordered_map<cudaStream_t, std::deque<void *>> stream_ptr;
static std::unordered_map<void *, uint64_t> ptr_size;

void freeMemCache();

cudaError_t cudaCacheMalloc(void **ptr, size_t size);

cudaError_t cudaCacheMallocAsync(void **ptr, size_t size, cudaStream_t stream);

cudaError_t cudaCacheFree(void *ptr);

cudaError_t cudaCacheFreeAsync(void *ptr, cudaStream_t stream);

void cudaCacheCommit(cudaStream_t stream);