#pragma once

#include "common.h"

f16 *float2half(f32 *f32_array, size_t size);

f32 *hafl2float(f16 *f16_array, size_t size);

// A: m x k, B: k x n, C: m x n
void gemm(f16 *a, f16 *b, f32 *c,
          size_t m, size_t n, size_t k);

// batched B
void gemmBatched(f16 *a, f16 *b, f32 *c,
                 size_t m, size_t n, size_t k,
                 size_t batch_size);
