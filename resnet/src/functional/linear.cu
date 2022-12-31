#include "functional/ops.h"
#include "functional/gemm.h"
#include "functional/linear.h"

void linear(f16 *input, float *output,
            f16 *weight, float *bias,
            int batch_size, int input_channel, int output_channel)
{
    gemm(input, weight, output,
         batch_size, output_channel, input_channel,
         MatrixMajor::RowMajor);
    for (int t = 0; t < batch_size; ++t)
        hostAdd_(&output[t * output_channel], bias, output_channel);
}

void makeLinearWeight(float *input, f16 *output, int row, int col)
{
    float *input_data = new float[row * col];
    f16 *output_data = new f16[row * col];
    checkCudaErrors(cudaMemcpy(input_data, input,
                               sizeof(float) * row * col,
                               cudaMemcpyDeviceToHost));
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            output_data[j * row + i] = __float2half(input_data[i * col + j]);
    checkCudaErrors(cudaMemcpy(output, output_data,
                               sizeof(f16) * row * col,
                               cudaMemcpyHostToDevice));
    delete[] input_data;
    delete[] output_data;
}