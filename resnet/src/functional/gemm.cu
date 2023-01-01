/** @file gemm.cu
 */
#include <mma.h>

#include "mem_cache.h"
#include "functional/gemm.h"

using namespace nvcuda;

static __global__ void float2halfKernel(const f32 *input, f16 *output, size_t size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += blockDim.x * gridDim.x)
        output[i] = __float2half(input[i]);
}

static __global__ void half2floatKernel(f16 *input, f32 *output, size_t size)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += blockDim.x * gridDim.x)
        output[i] = __half2float(input[i]);
}

f16 *float2half(f32 *f32_array, size_t size)
{
    f16 *f16_array;
    checkCudaErrors(cudaCacheMalloc((void **)&f16_array, size * sizeof(f16)));
    static const int N_THREADS = 128;
    static const int PER_THREAD = 4;
    dim3 grid_dim((size - 1) / N_THREADS / PER_THREAD + 1);
    float2halfKernel<<<grid_dim, N_THREADS>>>(f32_array, f16_array, size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return f16_array;
}

f32 *half2float(f16 *f16_array, size_t size)
{
    f32 *f32_array;
    checkCudaErrors(cudaCacheMalloc((void **)&f32_array, size * sizeof(f32)));
    static const int N_THREADS = 128;
    static const int PER_THREAD = 4;
    dim3 grid_dim((size - 1) / N_THREADS / PER_THREAD + 1);
    half2floatKernel<<<grid_dim, N_THREADS>>>(f16_array, f32_array, size);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    return f32_array;
}

static const int WARP_SIZE = 32;
// For Volta architecture, only f16 of 16x16x16 is supported.
static const int VOLTA_FACTOR = 16;

template <int block_size>
static __global__ void gemmKernel(f16 *a, f16 *b, f32 *result,
                                  size_t m, size_t n, size_t k,
                                  size_t padded_m, size_t padded_n)
{

    static constexpr int BLOCK_THREADS = WARP_SIZE * (block_size * block_size);
    static constexpr int TILE_SIZE = block_size * VOLTA_FACTOR;
    static constexpr int N_ELEMENTS = TILE_SIZE * TILE_SIZE / BLOCK_THREADS;

    __shared__ f16 shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ f16 shared_b[TILE_SIZE][TILE_SIZE];

    auto tid = threadIdx.y * blockDim.x + threadIdx.x;
    auto block_row = tid % TILE_SIZE;
    auto block_col = tid / TILE_SIZE;
    static constexpr int col_stride = 2 * block_size;

    // base row and col of the tile
    auto row_base = blockIdx.x * TILE_SIZE;
    auto col_base = blockIdx.y * TILE_SIZE;

    wmma::fragment<wmma::matrix_a, VOLTA_FACTOR, VOLTA_FACTOR, VOLTA_FACTOR, f16, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, VOLTA_FACTOR, VOLTA_FACTOR, VOLTA_FACTOR, f16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, VOLTA_FACTOR, VOLTA_FACTOR, VOLTA_FACTOR, f32> result_frag;
    wmma::fill_fragment(result_frag, 0.0f);

    for (int t = 0; t < k; t += TILE_SIZE)
    {
        // Parallel loading
        for (int i = 0, row = block_row, col = block_col;
             i < N_ELEMENTS; ++i, col += col_stride)
        {
            // auto idx = (tid + i);
            // FIXME: this indexing method is extremely slow
            // auto row = idx % TILE_SIZE;
            // auto col = idx / TILE_SIZE;
            shared_a[col][row] = row_base + row < m && t + col < k
                                     ? a[(t + col) * m + row_base + row]
                                     : f16(0);
            shared_b[col][row] = t + row < k && col_base + col < n
                                     ? b[(col_base + col) * k + t + row]
                                     : f16(0);
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i += VOLTA_FACTOR)
        {
            auto a_offset = i * TILE_SIZE + VOLTA_FACTOR * (threadIdx.x / WARP_SIZE);
            auto b_offset = VOLTA_FACTOR * threadIdx.y * TILE_SIZE + i;

            wmma::load_matrix_sync(a_frag, (half *)shared_a + a_offset, TILE_SIZE);
            wmma::load_matrix_sync(b_frag, (half *)shared_b + b_offset, TILE_SIZE);

            wmma::mma_sync(result_frag, a_frag, b_frag, result_frag);
        }
    }

    auto row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * VOLTA_FACTOR;
    auto col = (blockIdx.y * blockDim.y + threadIdx.y) * VOLTA_FACTOR;
    if (row < padded_m && col < padded_n)
        wmma::store_matrix_sync(result + row + col * padded_m,
                                result_frag, padded_m, wmma::mem_col_major);
}

template <int block_size>
static void gemmCaller(f16 *a, f16 *b, f32 *result,
                       size_t m, size_t n, size_t k,
                       size_t padded_m, size_t padded_n,
                       cudaStream_t &stream)
{
    constexpr int TILE_SIZE = block_size * VOLTA_FACTOR;

    dim3 grid((m - 1) / TILE_SIZE + 1, (n - 1) / TILE_SIZE + 1);
    dim3 block(block_size * WARP_SIZE, block_size);

    gemmKernel<block_size><<<grid, block, 0, stream>>>(a, b, result, m, n, k, padded_m, padded_n);
    checkCudaErrors(cudaPeekAtLastError());
}

template <typename T>
static T *paddingColMajor(const T *source,
                          size_t row, size_t col,
                          size_t pad_row, size_t pad_col,
                          cudaStream_t &stream)
{
    if (col == pad_col && row == pad_row)
        return (T *)source;

    T *padded;
    checkCudaErrors(cudaCacheMallocAsync((void **)&padded, sizeof(T) * pad_col * pad_row, stream));

    return padded;
}

template <typename T>
static void unpadColMajor(T *source, T *padded,
                          size_t row, size_t col,
                          size_t pad_row, size_t pad_col,
                          cudaStream_t &stream)
{
    if (source == padded)
        return;

    checkCudaErrors(cudaMemcpy2DAsync(source, sizeof(T) * col,
                                      padded, sizeof(T) * pad_col,
                                      sizeof(T) * col, row,
                                      cudaMemcpyDeviceToDevice,
                                      stream));
}

static void gemmStream(f16 *a, f16 *b, f32 *c,
                       size_t m, size_t n, size_t k,
                       cudaStream_t &stream)
{
    f32 *padded_c;

    // If M and N are not by 16, we need to pad them.
    auto padded_m = (m + (VOLTA_FACTOR - 1)) / VOLTA_FACTOR * VOLTA_FACTOR;
    auto padded_n = (n + (VOLTA_FACTOR - 1)) / VOLTA_FACTOR * VOLTA_FACTOR;
    padded_c = paddingColMajor<f32>(c, n, m, padded_n, padded_m, stream);

    // TODO: this template parameter is adjustable
    gemmCaller<2>(a, b, padded_c, m, n, k, padded_m, padded_n, stream);

    if (padded_c != c)
    {
        unpadColMajor<f32>(c, padded_c, n, m, padded_n, padded_m, stream);
        checkCudaErrors(cudaCacheFreeAsync(padded_c, stream));
    }
}

void gemm(f16 *a, f16 *b, f32 *c,
          size_t m, size_t n, size_t k)
{
    cudaStream_t stream = nullptr;
    gemmStream(b, a, c, n, m, k, stream);
    cudaCacheCommit(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

// batched B
void gemmBatched(f16 *a, f16 *b, f32 *c,
                 size_t m, size_t n, size_t k,
                 size_t batch_size)
{
    constexpr int STREAM_COUNT = 8;
    cudaStream_t streams[STREAM_COUNT];
    for (auto &stream : streams)
        checkCudaErrors(cudaStreamCreate(&stream));

    for (int i = 0; i < batch_size; i++)
    {
        auto b_ptr = b + i * k * n;
        auto c_ptr = c + i * m * n;
        gemmStream(b_ptr, a, c_ptr, n, m, k, streams[i % STREAM_COUNT]);
    }

    for (auto &stream : streams)
    {
        checkCudaErrors(cudaStreamSynchronize(stream));
        cudaCacheCommit(stream);
        checkCudaErrors(cudaStreamDestroy(stream));
    }
}
