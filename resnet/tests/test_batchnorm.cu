/**
 * @file test/test_batchnorm.cpp
 * @brief Test bn functionality
 */

#include <gtest/gtest.h>
#include <time.h>
#include <numeric>
#include "batchnorm.cu"

void bn_host(float* input_data, int batch_sizes, int channel_sizes, int height, int width, float eps)
{
    float* mean = new float[channel_sizes];
    float* var = new float[channel_sizes];
    int figure_sizes = height * width;
    for (int i = 0; i < channel_sizes; i++)
    {
        // mean
        float sum = 0.0;
        for (int j = 0; j < batch_sizes; j++)
        {
            unsigned int idx = j * channel_sizes * figure_sizes + i * figure_sizes;
            sum += std::accumulate(input_data + idx, input_data + idx + figure_sizes, 0.0);
        }
        float cur_mean = mean[i] = sum / (batch_sizes * figure_sizes);
        //std::cout << "sum = " << sum << std::endl;
        //std::cout << "mean[" << i << "] = " << cur_mean << std::endl;
        
        // var
        float sum_diff2 = 0.0;
        for (int j = 0; j < batch_sizes; j++)
        {
            unsigned int idx = j * channel_sizes * figure_sizes + i * figure_sizes;
            for (int k = 0; k < figure_sizes; k++)
            {
                float diff = input_data[idx + k] - cur_mean;
                sum_diff2 += diff * diff;
            }
        }
        var[i] = sum_diff2 / (batch_sizes * figure_sizes);
        //std::cout << "sum_diff2 = " << sum_diff2 << std::endl;
        //std::cout << "var[" << i << "] = " << var[i] << std::endl;

        // update data
        float denominator = sqrt(var[i] + eps);
        for (int j = 0; j < batch_sizes; j++)
        {
            unsigned int idx = j * channel_sizes * figure_sizes + i * figure_sizes;
            for (int k = 0; k < figure_sizes; k++)
            {
                input_data[idx + k] = (input_data[idx + k] - cur_mean) / denominator;
            }
        }
    }
}

TEST(batchnorm, bn)
{
    srand(time(NULL));
    const float eps = 1e-5;
    // create data
    int batch_sizes = 100;
    int channel_sizes = 3;
    int height = 200;
    int width = 200;
    int total_sizes = batch_sizes * channel_sizes * height * width;
    //float data_[16] = {1,1,1,2,-1,1,0,1,0,-1,2,2,0,-1,3,1};
    float *data_ = new float[total_sizes];
    for (int i = 0; i < total_sizes; i++)
    {
        data_[i] = (rand() % 100) * 0.1;
    }
    
    float *data_cpu = new float[total_sizes];
    std::copy(data_, data_ + total_sizes, data_cpu);
    float *data = new float[total_sizes];
    std::copy(data_, data_ + total_sizes, data);
    Tensor x({batch_sizes, channel_sizes, height, width}, DeviceType::CPU, data);
    
    // execution on cpu
    bn_host(data_cpu, batch_sizes, channel_sizes, height, width, eps);
    // execution on gpu
    BatchNorm2d bn2d(total_sizes);
    bn2d.forward(x);

    // check correctness
    float* res_gpu = x.data_ptr();
    float* res_cpu = data_cpu;
    for (int i = 0; i < total_sizes; i++)
    {
        //std::cout << "cpu = " << res_cpu[i] << ", gpu= " << res_gpu[i] << std::endl;
        EXPECT_NEAR(res_cpu[i], res_gpu[i], 1e-3);
        //EXPECT_NEAR(0, 0.0, 1e-3);
    }
    delete [] data_;
    delete [] data_cpu;
}