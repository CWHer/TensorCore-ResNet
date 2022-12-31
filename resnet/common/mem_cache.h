#pragma once

#include "common.h"

static std::unordered_map<uint64_t, std::deque<void *>> mem_cache;

void freeMemCache();

cudaError_t cudaCacheMalloc(void **ptr, size_t size);

cudaError_t cudaCacheFree(void *ptr);
