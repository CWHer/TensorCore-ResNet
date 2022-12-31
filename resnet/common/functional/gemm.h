#pragma once

#include "common.h"

enum MatrixMajor
{
    ColMajor,
    RowMajor
};

f16 *float2half(f32 *f32_array, size_t size);

f32 *hafl2float(f16 *f16_array, size_t size);

void gemm(f16 *a, f16 *b, f32 *c,
          size_t m, size_t n, size_t k,
          MatrixMajor major = MatrixMajor::RowMajor);

// batched B
void gemmBatched(f16 *a, f16 *b, f32 *c,
                 size_t m, size_t n, size_t k,
                 size_t batch_size, MatrixMajor major);
