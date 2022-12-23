/**
 * @file test/test_batchnorm.cpp
 * @brief Test bn functionality
 */

#include <gtest/gtest.h>
#include "batchnorm.cu"

void GetMean(float* input_data, float* output_data,
             const unsigned int batch_size, const unsigned int num_channels, const unsigned int height, const unsigned int width)
{
    const unsigned int figure_sizes = height * width;
    for (int i = 0; i < num_channels; i++)
    {
        float sum = 0.0;
        for (int j = 0; j < batch_size; j++)
        {
            unsigned int idx = j * num_channels * figure_sizes + i * figure_sizes;
            sum += std::accumulate(input_data + idx, input_data + idx + figure_sizes, 0.0);
        }
        output_data[i] = sum / (batch_size * figure_sizes);
    }
}

void GetVar(float* input_data, float* output_data, float* mean_data,
            const unsigned int batch_size, const unsigned int num_channels, const unsigned int height, const unsigned int width)
{
    const unsigned int figure_sizes = height * width;
    for (int i = 0; i < num_channels; i++)
    {
        float cur_mean = mean_data[i];
        float sum_diff2 = 0.0;
        for (int j = 0; j < batch_size; j++)
        {
            unsigned int idx = j * num_channels * figure_sizes + i * figure_sizes;
            for (int k = 0; k < figure_sizes; k++)
            {
                float diff = input_data[idx + k] - cur_mean;
                sum_diff2 += diff * diff;
            }
        }
        output_data[i] = sum_diff2 / (batch_size * figure_sizes);
    }
}

void BnHost(float *input_data, float *mean_data, float *var_data, float *weight_data, float *bias_data, const float eps,
            const unsigned int batch_size, const unsigned int num_channels, const unsigned int height, const unsigned int width)
{
    int figure_sizes = height * width;
    for (int i = 0; i < num_channels; i++)
    {
        // update data
        float cur_mean = mean_data[i];
        float cur_var = var_data[i];
        float cur_weight = weight_data[i];
        float cur_bias = bias_data[i];
        float denominator = std::sqrt(cur_var + eps);
        for (int j = 0; j < batch_size; j++)
        {
            unsigned int idx = j * num_channels * figure_sizes + i * figure_sizes;
            for (int k = 0; k < figure_sizes; k++)
            {
                input_data[idx + k] = ((input_data[idx + k] - cur_mean) / denominator) * cur_weight + cur_bias;
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
    int channel_sizes = 10;
    int height = 100;
    int width = 100;
    int total_sizes = batch_sizes * channel_sizes * height * width;
    //float data_[16] = {1,1,1,2,-1,1,0,1,0,-1,2,2,0,-1,3,1};
    float *data_ = new float[total_sizes];
    for (int i = 0; i < total_sizes; i++)
    {
        data_[i] = (rand() % 10) * 0.1;
    }

    float *mean_data = new float[channel_sizes];
    float *var_data = new float[channel_sizes];
    GetMean(data_, mean_data, batch_sizes, channel_sizes, height, width);
    GetVar(data_, var_data, mean_data, batch_sizes, channel_sizes, height, width);
    float *weight_data = new float[channel_sizes];
    float *bias_data = new float[channel_sizes];
    for (int i = 0; i < channel_sizes; i++)
    {
        weight_data[i] = 0.9;
        bias_data[i] = 0.1;
    }

    Tensor weight({channel_sizes}, DeviceType::CPU, weight_data);
    Tensor bias({channel_sizes}, DeviceType::CPU, bias_data);
    Tensor running_mean({channel_sizes}, DeviceType::CPU, mean_data);
    Tensor running_var({channel_sizes}, DeviceType::CPU, var_data);
    // execution on cpu
    float *data_cpu = new float[total_sizes];
    std::copy(data_, data_ + total_sizes, data_cpu);
    BnHost(data_cpu, mean_data, var_data, weight_data, bias_data, eps, batch_sizes, channel_sizes, height, width);

    // execution on gpu
    float *data = new float[total_sizes];
    std::copy(data_, data_ + total_sizes, data);
    Tensor x({batch_sizes, channel_sizes, height, width}, DeviceType::CPU, data);
    BatchNorm2d bn2d(weight, bias, running_mean, running_var, eps);
    bn2d.forward(x);

    // check correctness
    float *res_gpu = x.data_ptr();
    float *res_cpu = data_cpu;
    for (int i = 0; i < total_sizes; i++)
    {
        // std::cout << "cpu = " << res_cpu[i] << ", gpu= " << res_gpu[i] << std::endl;
        EXPECT_NEAR(res_cpu[i], res_gpu[i], 1e-3);
        //EXPECT_NEAR(0, 0.0, 1e-3);
    }
    delete[] data_;
    delete[] data_cpu;
}