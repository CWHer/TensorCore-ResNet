#include "functional/ops.h"

__global__ void kernelAdd_(float *result, float *adder, int length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < length; i += blockDim.x * gridDim.x)
        result[i] += adder[i];
}

__global__ void kernelRelu_(float *result, int length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < length; i += blockDim.x * gridDim.x)
        result[i] = result[i] > 0.0f ? result[i] : 0.0f;
}

__global__ void kernelAddRelu_(float *result, float *adder, int length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < length; i += blockDim.x * gridDim.x)
    {
        float value = result[i] + adder[i];
        result[i] = value > 0.0f ? value : 0.0f;
    }
}

void hostAdd_(float *result, float *adder_a, int length)
{
    static const int N_THREADS = 128;
    static const int PER_THREAD = 4;
    kernelAdd_<<<(length - 1) / N_THREADS / PER_THREAD + 1, N_THREADS>>>(result, adder_a, length);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void hostRelu_(float *result, int length)
{
    static const int N_THREADS = 128;
    static const int PER_THREAD = 4;
    kernelRelu_<<<(length - 1) / N_THREADS / PER_THREAD + 1, N_THREADS>>>(result, length);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void hostAddRelu_(float *result, float *adder_a, int length)
{
    static const int N_THREADS = 128;
    static const int PER_THREAD = 4;
    kernelAddRelu_<<<(length - 1) / N_THREADS / PER_THREAD + 1, N_THREADS>>>(result, adder_a, length);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
