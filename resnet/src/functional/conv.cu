#include "functional/gemm.h"
#include "functional/im2col.h"
#include "functional/conv.h"

/** @brief Convolutional layer forward propagation.
 *
 * @param input Input float, row major (organizes at last element), of shape (N, C, H, W).
 * @param output Result float, row major (organizes at last element), of shape (N, out_channels, H_out, W_out).
 *  where H_out = floor((H + 2 * padding - kernel_size) / stride) + 1
 *  and W_out = floor((W + 2 * padding - kernel_size) / stride) + 1
 *  @param weight Weight float_16, row major (organizes at last element), of shape (out_channels, C, kernel_size, kernel_size).
 *  @param bias Bias float, row major (organizes at last element), in shape of (out_channels).
 *  @param N Batch size.
 *  @param C Number of input channels.
 *  @param H Height of input.
 *  @param W Width of input.
 *  @param kernel_size Size of kernel (kernel_size x kernel_size).
 *  @param stride Stride of convolution.
 *  @param padding Padding of convolution.
 */
void conv2d(float *input, float *output, f16 *weight,
            int n, int c, int h, int w,
            int out_channels, int out_height, int out_width,
            int kernel_size, int stride, int padding)
{
    int conv_result_size = out_height * out_width;
    int expanded_kernel_size = c * kernel_size * kernel_size;

    static const int MINIBATCH_SIZE = 128;
    auto total_batch_size = (n + MINIBATCH_SIZE - 1) / MINIBATCH_SIZE;
    Tensor im2col_result = makeIm2colResult(MINIBATCH_SIZE, c, h, w,
                                            kernel_size, stride, padding);

    for (int i = 0; i < total_batch_size; ++i)
    {
        auto batch_size = std::min(MINIBATCH_SIZE, n - i * MINIBATCH_SIZE);
        if (batch_size != MINIBATCH_SIZE)
            im2col_result = makeIm2colResult(batch_size, c, h, w,
                                             kernel_size, stride, padding);

        im2col(input + i * MINIBATCH_SIZE * c * h * w,
               reinterpret_cast<f16 *>(im2col_result.data_ptr()),
               batch_size, c, h, w, kernel_size, stride, padding);
        gemmBatched(weight, reinterpret_cast<f16 *>(im2col_result.data_ptr()),
                    output + i * MINIBATCH_SIZE * out_channels * conv_result_size,
                    out_channels, conv_result_size, expanded_kernel_size, batch_size,
                    MatrixMajor::RowMajor);
    }
}