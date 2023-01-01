#include <gtest/gtest.h>
#include "mem_cache.h"

TEST(mem_cache, sync)
{
    float *a, *b;
    cudaCacheMalloc((void **)&a, 1024);
    cudaCacheFree(a);
    cudaCacheMalloc((void **)&b, 1024);
    EXPECT_EQ(a, b);
    cudaCacheFree(b);

    freeMemCache();
}

TEST(mem_cache, async)
{
    void *a, *b, *c;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaCacheMallocAsync(&a, 1024, stream);
    cudaCacheFreeAsync(a, stream);
    cudaCacheMalloc(&b, 1024);
    EXPECT_NE(a, b);

    cudaStreamSynchronize(stream);
    cudaCacheCommit(stream);
    cudaCacheMallocAsync(&c, 1024, stream);
    EXPECT_EQ(a, c);

    cudaCacheFree(b);
    cudaCacheFreeAsync(c, stream);

    cudaStreamSynchronize(stream);
    cudaCacheCommit(stream);
    cudaStreamDestroy(stream);

    freeMemCache();
}