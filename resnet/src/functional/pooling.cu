#include <cfloat>
#include "common.h"

// TODO: tiling and leverage cache
template <int block_size>
__global__ void deviceMaxPool2dKernel(const float *input_data, int height, int width,
                                      float *output_data, int out_height, int out_width,
                                      int kernel_size, int padding, int stride)
{
    // NOTE: B/2 (block) x 128 (thread) x H x W (within)
    int input_grid_offset = 2 * blockDim.x * height * width;
    int output_grid_offset = 2 * blockDim.x * out_height * out_width;
    int input_thread_offset = height * width;
    int output_thread_offset = out_height * out_width;

    input_data += blockIdx.x * input_grid_offset +
                  threadIdx.x * input_thread_offset;
    output_data += blockIdx.x * output_grid_offset +
                   threadIdx.x * output_thread_offset;

    // HACK: NOTE: Max-pooling uses implicit negative infinity padding,
    //  not zero-padding as indicated in documentation
    // https://github.com/pytorch/pytorch/issues/33384

    int output_offset = 0;
    for (int i = kernel_size - 1 - padding; i < height + padding; i += stride)
        for (int j = kernel_size - 1 - padding; j < width + padding; j += stride)
        {
            float ret = -FLT_MAX;
            for (int x = i - kernel_size + 1; x <= i; x++)
                for (int y = j - kernel_size + 1; y <= j; y++)
                {
                    float value = 0 <= x && x < height &&
                                          0 <= y && y < width
                                      ? input_data[x * width + y]
                                      : -FLT_MAX; // negative infinity padding
                    ret = max(ret, value);
                }
            output_data[output_offset++] = ret;
        }
}

// TODO: tiling and leverage cache
template <int block_size>
__global__ void deviceAvgPool2dKernel(const float *input_data, int height, int width,
                                      float *output_data, int out_height, int out_width,
                                      int kernel_size, int padding, int stride)
{
    // NOTE: B/2 (block) x 128 (thread) x H x W (within)
    int input_grid_offset = 2 * blockDim.x * height * width;
    int output_grid_offset = 2 * blockDim.x * out_height * out_width;
    int input_thread_offset = height * width;
    int output_thread_offset = out_height * out_width;

    input_data += blockIdx.x * input_grid_offset +
                  threadIdx.x * input_thread_offset;
    output_data += blockIdx.x * output_grid_offset +
                   threadIdx.x * output_thread_offset;

    int output_offset = 0;
    float r = 1.0f / (kernel_size * kernel_size);
    for (int i = kernel_size - 1 - padding; i < height + padding; i += stride)
        for (int j = kernel_size - 1 - padding; j < width + padding; j += stride)
        {
            float ret = 0;
            for (int x = i - kernel_size + 1; x <= i; x++)
                for (int y = j - kernel_size + 1; y <= j; y++)
                {
                    float value = 0 <= x && x < height &&
                                          0 <= y && y < width
                                      ? input_data[x * width + y]
                                      : 0; // zero padding
                    ret += value;
                }
            output_data[output_offset++] = ret * r;
        }
}

void maxpool2d(const float *input_data,
               int batch_size,
               int num_channels,
               int height,
               int width,
               float *output_data,
               int out_height,
               int out_width,
               int kernel_size,
               int padding,
               int stride) {
  static const int N_THREADS = 128;
  dim3 block_dim(N_THREADS);
  dim3 grid_dim(batch_size * num_channels / N_THREADS);
  deviceMaxPool2dKernel<32><<<grid_dim, block_dim>>>(
      input_data, height, width, output_data, out_height, out_width,
      kernel_size, padding, stride);
}


void avgpool2d(const float *input_data,
               int batch_size,
               int num_channels,
               int height,
               int width,
               float *output_data,
               int out_height,
               int out_width,
               int kernel_size,
               int padding,
               int stride){
  static const int N_THREADS = 128;
  dim3 block_dim(N_THREADS);
  dim3 grid_dim(batch_size * num_channels / N_THREADS);
  deviceAvgPool2dKernel<32><<<grid_dim, block_dim>>>(
      input_data, height, width, output_data, out_height, out_width,
      kernel_size, padding, stride);
}
