#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "mma.h"

using namespace nvcuda;
using std::cout;
using std::endl;
using std::vector;

__global__ void wmma_kernel(half *a, half *b, float *c)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

void wmma(half *a, half *b, float *c) {}

int main()
{
    vector<half> a(16 * 16), b(16 * 16);
    vector<float> c(16 * 16);
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            if (j >= i)
            {
                a[i * 16 + j] = __float2half(1);
                b[i + j * 16] = __float2half(1);
            }
            else
            {
                a[i * 16 + j] = __float2half(0);
                b[i + j * 16] = __float2half(0);
            }
        }
    }
    int size = 16 * 16;
    half *d_a, *d_b;
    float *d_c;
    cudaMalloc((void **)(&d_a), sizeof(half) * size);
    cudaMalloc((void **)(&d_b), sizeof(half) * size);
    cudaMalloc((void **)(&d_c), sizeof(float) * size);

    cudaMemcpy(d_a, a.data(), sizeof(half) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(half) * size, cudaMemcpyHostToDevice);

    dim3 blockDim, gridDim;
    blockDim.x = 4 * 32;
    blockDim.y = 4;
    gridDim.x = 1;
    gridDim.y = 1;
    wmma_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c);
    cudaMemcpy(c.data(), d_c, sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            cout << __half2float(a[i * 16 + j]) << ' ';
        }
        cout << endl;
    }
    cout << endl;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            cout << c[i * 16 + j] << ' ';
        }
        cout << endl;
    }
}
