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

template <int block_col_warps, int block_row_warps>
static __global__ void gemmKernel(f16 *a, f16 *b, f32 *result,
                                  size_t m, size_t n, size_t k)
{

    const int BLOCK_THREADS = WARP_SIZE * (block_row_warps * block_col_warps);

    constexpr int TILE_M = block_row_warps * VOLTA_FACTOR;
    constexpr int TILE_N = block_col_warps * VOLTA_FACTOR;
    constexpr int TILE_K = TILE_M;

    __shared__ f16 shared_a[TILE_K][TILE_M];
    __shared__ f16 shared_b[TILE_N][TILE_K];

    auto tid = threadIdx.y * blockDim.x + threadIdx.x;

    auto a_row = blockIdx.x * TILE_M;
    auto b_col = blockIdx.y * TILE_N;

    wmma::fragment<wmma::matrix_a, VOLTA_FACTOR, VOLTA_FACTOR, VOLTA_FACTOR, f16, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, VOLTA_FACTOR, VOLTA_FACTOR, VOLTA_FACTOR, f16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, VOLTA_FACTOR, VOLTA_FACTOR, VOLTA_FACTOR, f32> result_frag;
    wmma::fill_fragment(result_frag, 0.0f);

    for (int t = 0; t < k; t += TILE_K)
    {
        // Parallel loading
        for (int i = 0; i < TILE_M * TILE_K; i += BLOCK_THREADS)
        {
            auto idx = (tid + i);

            auto as_row = idx % TILE_M;
            auto as_col = idx / TILE_M;
            auto bs_row = idx % TILE_K;
            auto bs_col = idx / TILE_K;

            shared_a[as_col][as_row] = a_row + as_row < m && t + as_col < k
                                           ? a[(t + as_col) * m + a_row + as_row]
                                           : f16(0);
            shared_b[bs_col][bs_row] = t + bs_row < k && b_col + bs_col < n
                                           ? b[(b_col + bs_col) * k + t + bs_row]
                                           : f16(0);
        }
        __syncthreads();

        for (int i = 0; i < TILE_K; i += VOLTA_FACTOR)
        {
            auto as_offset = i * TILE_M + VOLTA_FACTOR * (threadIdx.x / WARP_SIZE);
            auto bs_offset = VOLTA_FACTOR * threadIdx.y * TILE_K + i;

            wmma::load_matrix_sync(a_frag, (half *)shared_a + as_offset, TILE_M);
            wmma::load_matrix_sync(b_frag, (half *)shared_b + bs_offset, TILE_K);

            wmma::mma_sync(result_frag, a_frag, b_frag, result_frag);
        }
    }

    auto c_row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * VOLTA_FACTOR;
    auto c_col = (blockIdx.y * blockDim.y + threadIdx.y) * VOLTA_FACTOR;
    auto c_offset = c_row + c_col * m;

    if (c_row < m && c_col < n)
        wmma::store_matrix_sync(result + c_offset, result_frag, m, wmma::mem_col_major);
}

template <int block_col_warps, int block_row_warps>
static void gemmCaller(f16 *a, f16 *b, f32 *result,
                       size_t m, size_t n, size_t k,
                       cudaStream_t &stream)
{
    constexpr int TILE_M = block_row_warps * VOLTA_FACTOR;
    constexpr int TILE_N = block_col_warps * VOLTA_FACTOR;

    dim3 grid((m + (TILE_M - 1)) / TILE_M, (n + (TILE_N - 1)) / TILE_N);
    dim3 block(block_row_warps * WARP_SIZE, block_col_warps);

    gemmKernel<block_col_warps, block_row_warps><<<grid, block, 0, stream>>>(a, b, result, m, n, k);
    checkCudaErrors(cudaPeekAtLastError());
}

template <typename T, bool require_copy>
static T *paddingColMajor(const T *source,
                          size_t row, size_t col,
                          size_t pad_row, size_t pad_col,
                          cudaStream_t &stream)
{
    if (col == pad_col && row == pad_row)
        return (T *)source;

    T *padded;
    checkCudaErrors(cudaCacheMallocAsync((void **)&padded, sizeof(T) * pad_col * pad_row, stream));

    if (require_copy)
    {
        checkCudaErrors(cudaMemsetAsync((void *)padded, 0, sizeof(T) * pad_col * pad_row, stream));
        checkCudaErrors(cudaMemcpy2DAsync(padded, sizeof(T) * pad_col,
                                          source, sizeof(T) * col, sizeof(T) * col,
                                          row, cudaMemcpyDeviceToDevice, stream));
    }

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

    if (col == pad_col && row == pad_row)
        checkCudaErrors(cudaMemcpyAsync(source, padded, sizeof(T) * col * row,
                                        cudaMemcpyDeviceToDevice, stream));
    else
        checkCudaErrors(cudaMemcpy2DAsync(source, sizeof(T) * col,
                                          padded, sizeof(T) * pad_col,
                                          sizeof(T) * col, row,
                                          cudaMemcpyDeviceToDevice,
                                          stream));
}

static void gemmDeviceMemory(f16 *a, f16 *b, f32 *result,
                             size_t m, size_t n, size_t k,
                             cudaStream_t &stream)
{
    f16 *padded_a, *padded_b;
    f32 *padded_c;

    // If M and N are not by 16, we need to pad them.
    auto padded_m = (m + (VOLTA_FACTOR - 1)) / VOLTA_FACTOR * VOLTA_FACTOR;
    auto padded_n = (n + (VOLTA_FACTOR - 1)) / VOLTA_FACTOR * VOLTA_FACTOR;

    // Copy A and B by padding to device
    // B needs padding, with a leading dimension of N
    padded_a = paddingColMajor<f16, true>(a, k, m, k, padded_m, stream);
    padded_b = paddingColMajor<f16, true>(b, n, k, padded_n, k, stream);
    padded_c = paddingColMajor<f32, false>(result, n, m, padded_n, padded_m, stream);

    // TODO: this template parameter is adjustable
    gemmCaller<2, 2>(padded_a, padded_b, padded_c, padded_m, padded_n, k, stream);

    if (padded_a != a)
        checkCudaErrors(cudaCacheFreeAsync(padded_a, stream));

    if (padded_b != b)
        checkCudaErrors(cudaCacheFreeAsync(padded_b, stream));

    if (padded_c != result)
    {
        unpadColMajor<f32>(result, padded_c, n, m, padded_n, padded_m, stream);
        checkCudaErrors(cudaCacheFreeAsync(padded_c, stream));
    }
}

static void gemmStream(f16 *a, f16 *b, f32 *c,
                       size_t m, size_t n, size_t k,
                       MatrixMajor major,
                       cudaStream_t &stream)
{
    switch (major)
    {
    case MatrixMajor::ColMajor:
        gemmDeviceMemory(a, b, c, m, n, k, stream);
        break;
    case MatrixMajor::RowMajor:
        gemmDeviceMemory(b, a, c, n, m, k, stream);
        break;
    default:
        throw std::runtime_error("Unsupported matrix major");
    }
}

void gemm(f16 *a, f16 *b, f32 *c,
          size_t m, size_t n, size_t k,
          MatrixMajor major)
{
    cudaStream_t stream = nullptr;
    gemmStream(a, b, c, m, n, k, major, stream);
    cudaCacheCommit(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

// batched B
void gemmBatched(f16 *a, f16 *b, f32 *c,
                 size_t m, size_t n, size_t k,
                 size_t batch_size, MatrixMajor major)
{
    checkCppErrorsMsg(major != MatrixMajor::RowMajor,
                      "Batches does not support col-major");

    constexpr int STREAM_COUNT = 8;
    cudaStream_t streams[STREAM_COUNT];
    for (auto &stream : streams)
        checkCudaErrors(cudaStreamCreate(&stream));

    for (int i = 0; i < batch_size; i++)
    {
        auto b_ptr = b + i * k * n;
        auto c_ptr = c + i * m * n;
        gemmStream(a, b_ptr, c_ptr, m, n, k, major, streams[i % STREAM_COUNT]);
    }

    for (auto &stream : streams)
    {
        checkCudaErrors(cudaStreamSynchronize(stream));
        cudaCacheCommit(stream);
        checkCudaErrors(cudaStreamDestroy(stream));
    }
}
