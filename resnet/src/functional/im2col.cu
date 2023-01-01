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
__global__ static void im2colKernel(float *input, f16 *output,
                                    int n, int c, int h, int w,
                                    int kernel_size, int stride, int padding,
                                    int out_height, int out_width)
{
    int output_size = out_height * out_width;

    int filter_size = kernel_size * kernel_size;
    int input_channel_size = h * w;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < output_size * filter_size * c * n; i += blockDim.x * gridDim.x)
    {
        // FIXME: unify the order with im2col_naive
        // order: output_h, output_w, filter_h, filter_w, c
        int cur_index = i;
        int cur_n = cur_index % n;
        cur_index /= n;
        int cur_c = cur_index % c;
        cur_index /= c;
        int filter_w = cur_index % kernel_size;
        cur_index /= kernel_size;
        int filter_h = cur_index % kernel_size;
        cur_index /= kernel_size;
        int output_w = cur_index % out_width;
        cur_index /= out_width;
        int output_h = cur_index;

        int index_h = output_h * stride + filter_h - padding;
        int index_w = output_w * stride + filter_w - padding;
        int input_index = (cur_n * c + cur_c) * input_channel_size + index_h * w + index_w;
        // clang-format off
        if (input_index > n * c * h * w) continue;
        // clang-format on

        int output_index = (cur_n * c + cur_c) * filter_size + filter_h * kernel_size + filter_w;
        int output_offset = output_h * out_width + output_w;
        output[output_index * output_size + output_offset] = index_h >= 0 && index_h < h &&
                                                                     index_w >= 0 && index_w < w &&
                                                                     cur_c < c && cur_n < n
                                                                 ? __float2half(input[input_index])
                                                                 : f16(0);
    }
}

/**
 * @brief im2col, performs as the matlab function and at::Tensor function
 *
 * @param input float of shape (N, C, H, W)
 * @param output Output should be shaped as:
 * (N, C * filter_height * filter_width, output_height * output_width)
 *
 * where output_height = (H + 2 * padding - filter_height) / stride + 1
 * and output_width = (W + 2 * padding - filter_width) / stride + 1
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
    // Launch CUDA kernel
    constexpr unsigned long MINIBATCH_SIZE = 2;
    constexpr int N_STREAMS = 8;
    unsigned long minibatches = (n + MINIBATCH_SIZE - 1) / MINIBATCH_SIZE;

    cudaStream_t stream[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++)
        checkCudaErrors(cudaStreamCreate(&stream[i]));

    for (unsigned long i = 0; i < minibatches; i++)
    {
        int cur_minibatch_size = std::min(MINIBATCH_SIZE, n - i * MINIBATCH_SIZE);
        int cur_result_size = cur_minibatch_size * result_size_per_batch;

        static const int N_THREADS = 128;
        static const int PER_THREAD = 4;
        dim3 grid_dim((cur_result_size - 1) / N_THREADS / PER_THREAD + 1);
        im2colKernel<<<grid_dim, N_THREADS, 0, stream[i % N_STREAMS]>>>(
            input + i * MINIBATCH_SIZE * c * h * w,
            output + i * MINIBATCH_SIZE * result_size_per_batch,
            cur_minibatch_size, c, h, w,
            kernel_size, stride, padding,
            output_height, output_width);
        checkCudaErrors(cudaPeekAtLastError());
    }

    for (int i = 0; i < N_STREAMS; i++)
    {
        checkCudaErrors(cudaStreamSynchronize(stream[i]));
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }
}
