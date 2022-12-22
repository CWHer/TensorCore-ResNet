/**
 * @file test/test_pooling.cpp
 * @brief Test pooling functionality
 */

#include <gtest/gtest.h>
#include "pooling.cu"
#include <float.h>
#include <time.h>

static void max_pool_2d_naive(const int in_batch, const int in_channel, const int total_out_sizes,
                             const float* in_data, const int in_height, const int in_width,
                             float* out_data, const int out_height, const int out_width,
                             const int kernel_size_h, const int kernel_size_w,
                             const int padding_h, const int padding_w, 
                             const int stride_h, const int stride_w)
{
    for (int b = 0; b < in_batch; b++)
    {
        for (int c = 0; c < in_channel; c++)
        {
            for (int h = 0; h < out_height; h++)
            {
                for (int w = 0; w < out_width; w++)
                {
                    const int start_h = max(h * stride_h - padding_h, 0);
                    const int start_w = max(w * stride_w - padding_w, 0);
                    const int end_h = min(start_h + kernel_size_h, in_height + padding_h);
                    const int end_w = min(start_w + kernel_size_w, in_width + padding_w);
                    const float* in_data_start_idx = in_data + (b * in_channel + c) * in_height * in_width;
                    float max_val = -FLT_MAX;
                    for (int hk = start_h; hk < end_h; hk++)
                    {
                        for (int wk = start_w; wk < end_w; wk++)
                        {
                            int cur_idx = hk * in_width + wk;
                            max_val = in_data_start_idx[cur_idx] > max_val ? in_data_start_idx[cur_idx] : max_val;
                        }
                    }
                    //printf("b=%d, c=%d, h=%d, w=%d, max=%f\n", b, c, h, w, max_val);
                    //printf("sh=%d, sw=%d, eh=%d, ew=%d\n", start_h, start_w, end_h, end_w);
                    out_data[(b * in_channel + c) * out_height * out_width + h * out_width + w] = max_val;
                }
            }
        }
    }
}

static void adaptive_avg_pool_2d_naive(const int in_batch, const int in_channel, const int total_out_sizes,
                                        const float* in_data, const int in_height, const int in_width,
                                        float* out_data, const int out_height, const int out_width,
                                        const int kernel_size_h, const int kernel_size_w,
                                        const int padding_h, const int padding_w, 
                                        const int stride_h, const int stride_w)
{
    for (int b = 0; b < in_batch; b++)
    {
        for (int c = 0; c < in_channel; c++)
        {
            for (int h = 0; h < out_height; h++)
            {
                for (int w = 0; w < out_width; w++)
                {
                    const int start_h = max(h * stride_h - padding_h, 0);
                    const int start_w = max(w * stride_w - padding_w, 0);
                    const int end_h = min(start_h + kernel_size_h, in_height + padding_h);
                    const int end_w = min(start_w + kernel_size_w, in_width + padding_w);
                    const float* in_data_start_idx = in_data + (b * in_channel + c) * in_height * in_width;
                    float sum = 0.0;
                    for (int hk = start_h; hk < end_h; hk++)
                    {
                        for (int wk = start_w; wk < end_w; wk++)
                        {
                            sum += in_data_start_idx[hk * in_width + wk];
                        }
                    }
                    out_data[(b * in_channel + c) * out_height * out_width + h * out_width + w] = sum / ((end_h - start_h) * (end_w - start_w));
                }
            }
        }
    }
}

TEST(pooling, max_pool_2d)
{
    // Test if MaxPool2d works properly
    srand(time(NULL));
    const int batch_sizes = 10;
    const int channel_sizes = 10;
    const int height = 100;
    const int width = 100;
    int total_in_sizes = batch_sizes * channel_sizes * height * width;
    float *data_ = new float[total_in_sizes];
    for (int i = 0; i < total_in_sizes; i++)
    {
        data_[i] = (rand() % 100) * 0.1;
    }
    int stride = 2;
    int kernel_size = 3;
    int padding = 0;
    int dilation = 1;

    // cuda implementation
    float *data = new float[total_in_sizes];
    std::copy(data_, data_ + total_in_sizes, data);
    Tensor x({batch_sizes, channel_sizes, height, width}, DeviceType::CPU, data);
    MaxPool2d mp2d(kernel_size, stride, padding, dilation);
    Tensor res = mp2d.forward(x);
    float* cuda_out_data = res.data_ptr();

    // cpu implementation
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int total_out_sizes = batch_sizes * channel_sizes * out_height * out_width;
    float *naive_in_data = new float[total_in_sizes];
    std::copy(data_, data_ + total_in_sizes, naive_in_data);
    float* naive_out_data = new float[total_out_sizes];
    max_pool_2d_naive(batch_sizes, channel_sizes, total_out_sizes,
                      naive_in_data, height, width, 
                      naive_out_data, out_height, out_width,
                      kernel_size, kernel_size,
                      padding, padding,
                      stride, stride);

    const float eps = 1e-3;
    for (int i = 0; i < total_out_sizes; i++)
    {
        //printf("i=%d, cpu=%f, gpu=%f\n", i, naive_out_data[i], cuda_out_data[i]);
        EXPECT_NEAR(naive_out_data[i], cuda_out_data[i], eps);
        //EXPECT_NEAR(0, 0.0, eps);
    }
    delete [] data_;
    delete [] naive_in_data;
    delete [] naive_out_data;
}

TEST(pooling, adaptive_avg_pool_2d)
{
    // Test if AdaptiveAvgPool2d works properly
    srand(time(NULL));
    const int batch_sizes = 10;
    const int channel_sizes = 10;
    const int height = 100;
    const int width = 100;
    int total_in_sizes = batch_sizes * channel_sizes * height * width;
    float *data_ = new float[total_in_sizes];
    for (int i = 0; i < total_in_sizes; i++)
    {
        data_[i] = (rand() % 100) * 0.1;
    }
    int out_height = 2;
    int out_width = 2;
    int total_out_sizes = batch_sizes * channel_sizes * out_height * out_width;

    // cuda implementation
    float *data = new float[total_in_sizes];
    std::copy(data_, data_ + total_in_sizes, data);
    Tensor x({batch_sizes, channel_sizes, height, width}, DeviceType::CPU, data);
    AdaptiveAvgPool2d aap2d(out_height, out_width);
    Tensor res = aap2d.forward(x);
    float* cuda_out_data = res.data_ptr();

    // cpu implementation
    float *naive_in_data = new float[total_in_sizes];
    std::copy(data_, data_ + total_in_sizes, naive_in_data);
    int stride_h = height / out_height;
    int stride_w = width / out_width;
    int kernel_size_h = height - (out_height - 1) * stride_h;
    int kernel_size_w = width - (out_width - 1) * stride_w;
    int padding_h = 0;
    int padding_w = 0;
    float* naive_out_data = new float[total_out_sizes];
    adaptive_avg_pool_2d_naive(batch_sizes, channel_sizes, total_out_sizes,
                               naive_in_data, height, width, naive_out_data,
                               out_height, out_width,
                               kernel_size_h, kernel_size_w,
                               padding_h, padding_w,
                               stride_h, stride_w);

    const float eps = 1e-3;
    for (int i = 0; i < total_out_sizes; i++)
    {
        EXPECT_NEAR(naive_out_data[i], cuda_out_data[i], eps);
    }
    delete [] data_;
    delete [] naive_in_data;
    delete [] naive_out_data;
}