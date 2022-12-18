#include <cuda_runtime.h>
#include <mma.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace nvcuda;

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

template <typename T>
void printCudaError(T result, char const *const msg,
                    const char *const file, int const line)
{
    std::stringstream ss;
    if (result)
    {
        ss << "CUDA [Error] at: " << file << ":" << line
           << " code=" << static_cast<unsigned int>(result) << "("
           << cudaGetErrorName(result) << ")"
           << " \"" << msg << "\"";
        // Use throw to avoid gtest exiting.
        throw std::runtime_error(ss.str());
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(result) printCudaError((result), #result, __FILE__, __LINE__)

int main()
{
    std::vector<half> a(16 * 16), b(16 * 16);
    std::vector<float> c(16 * 16);
    for (int i = 0; i < 16; i++)
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
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            std::cout << __half2float(a[i * 16 + j])
                      << (j == 15 ? '\n' : ' ');
    std::cout << std::endl;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            std::cout << __half2float(b[i + j * 16])
                      << (j == 15 ? '\n' : ' ');
    std::cout << std::endl;

    int n_elements = 16 * 16;
    half *d_a, *d_b;
    float *d_c;
    checkCudaErrors(cudaMalloc((void **)(&d_a), sizeof(half) * n_elements));
    checkCudaErrors(cudaMalloc((void **)(&d_b), sizeof(half) * n_elements));
    checkCudaErrors(cudaMalloc((void **)(&d_c), sizeof(float) * n_elements));

    checkCudaErrors(cudaMemcpy(d_a, a.data(), sizeof(half) * n_elements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b.data(), sizeof(half) * n_elements, cudaMemcpyHostToDevice));

    dim3 block_dim(1, 1, 32);
    dim3 grid_dim(1, 1);
    wmma_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c);
    checkCudaErrors(cudaMemcpy(c.data(), d_c, sizeof(float) * n_elements, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            std::cout << std::setw(3) << c[i * 16 + j] << (j == 15 ? '\n' : ' ');
    std::cout << std::endl;

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));

    return 0;
}
