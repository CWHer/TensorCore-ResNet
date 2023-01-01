#include "functional/im2col.h"

Tensor makeIm2colResult(int n, int c, int h, int w,
                        int kernel_size, int stride, int padding)
{
    int output_height = (h + 2 * padding - kernel_size) / stride + 1;
    int output_width = (w + 2 * padding - kernel_size) / stride + 1;
    int output_size = output_height * output_width;
    int im2col_size = n * c * kernel_size * kernel_size * output_size;
    checkCppErrorsMsg(im2col_size % 2 != 0, "im2col size should be even");
    auto output = Tensor({im2col_size / 2}, DeviceType::CUDA);
    return output;
}

// borrowed from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/im2col.cuh
__global__ static void im2colKernel(f32 *input, f16 *output,
                                    const int num_kernels, const int height, const int width,
                                    const int kernel_size, const int stride, const int padding,
                                    const int out_height, const int out_width)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_kernels; i += blockDim.x * gridDim.x)
    {
        int w_out = i % out_width;
        int index = i / out_width;
        int h_out = index % out_height;
        int channel_in = index / out_height;
        int channel_out = channel_in * kernel_size * kernel_size;

        int h_in = h_out * stride - padding;
        int w_in = w_out * stride - padding;

        output += (channel_out * out_height + h_out) * out_width + w_out;
        input += (channel_in * height + h_in) * width + w_in;

        for (int i = 0; i < kernel_size; ++i)
            for (int j = 0; j < kernel_size; ++j)
            {
                int h = h_in + i;
                int w = w_in + j;
                *output = (h >= 0 && w >= 0 &&
                           h < height && w < width)
                              ? __float2half(input[i * width + j])
                              : f16(0);
                output += out_height * out_width;
            }
    }
}

static void im2colStream(float *input, f16 *output,
                         int channels, int height, int width,
                         int kernel_size, int stride, int padding,
                         int out_height, int out_width,
                         cudaStream_t stream)
{
    // We are going to launch channels * out_height * out_width kernels, each
    // kernel responsible for copying a single-channel grid.
    int num_kernels = channels * out_height * out_width;
    static const int N_THREADS = 128;
    im2colKernel<<<(num_kernels - 1) / N_THREADS + 1, N_THREADS, 0, stream>>>(
        input, output, num_kernels, height, width,
        kernel_size, stride, padding, out_height, out_width);
    checkCudaErrors(cudaPeekAtLastError());
}

/**
 * @brief im2col, performs as the matlab function and at::Tensor function
 *
 * @param input float of shape (N, C, H, W)
 * @param output Output should be shaped as:
 * (N, C * kernel_size * kernel_size, output_height * output_width)
 *
 * where output_height = (H + 2 * padding - kernel_size) / stride + 1
 * and output_width = (W + 2 * padding - kernel_size) / stride + 1
 *
 * Data are arranged per channel of columns, this results in factor C.
 */
void im2col(float *input, f16 *output,
            int n, int c, int h, int w,
            int kernel_size, int stride, int padding)
{
    int output_height = (h + 2 * padding - kernel_size) / stride + 1;
    int output_width = (w + 2 * padding - kernel_size) / stride + 1;
    int output_size = output_height * output_width;
    auto result_size_per_batch = c * kernel_size * kernel_size * output_size;

    constexpr int N_STREAMS = 8;
    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++)
        checkCudaErrors(cudaStreamCreate(&stream[i]));

    for (int i = 0; i < n; i++)
    {
        im2colStream(input + i * c * h * w,
                     output + i * result_size_per_batch,
                     c, h, w, kernel_size, stride, padding,
                     output_height, output_width, stream[i % N_STREAMS]);
    }

    for (int i = 0; i < N_STREAMS; i++)
    {
        checkCudaErrors(cudaStreamSynchronize(stream[i]));
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }
}
