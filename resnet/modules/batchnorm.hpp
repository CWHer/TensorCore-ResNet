#pragma once

#include "common.h"
#include "module.hpp"

#include "batchnorm.cu"

class BatchNorm2d : public Module
{
private:
    int num_features;
    double eps;
    Tensor weight, bias, running_mean, running_var;

public:
    // FIXME: these parameters are not used,
    //  recommend to add conflict check or remove them
    BatchNorm2d(int num_features, double eps = 1e-5, double momentum = 0.1,
                bool affine = true, bool track_running_stats = true)
        : num_features(num_features), eps(eps)
    {
        addTensor("weight", weight);             // [num_features]
        addTensor("bias", bias);                 // [num_features]
        addTensor("running_mean", running_mean); // [num_features]
        addTensor("running_var", running_var);   // [num_features]
    }

    Tensor forward(Tensor x) override
    {
        checkCppErrorsMsg(x.sizes().size() != 4, "BatchNorm2d only support 4D input");
        checkCppErrorsMsg(x.sizes()[1] != num_features, "BatchNorm2d input channel size mismatch");
        checkCppErrorsMsg(x.getDevice() != DeviceType::CUDA ||
                              weight.getDevice() != DeviceType::CUDA,
                          "BatchNorm2d only support CUDA");

        // NOTE: x is in NCHW format
        // NOTE: computation is done channel-wise
        unsigned int batch_size = x.sizes()[0];
        unsigned int num_channels = x.sizes()[1];
        unsigned int height = x.sizes()[2];
        unsigned int width = x.sizes()[3];

        float *input_data = x.data_ptr();
        float *mean_data = running_mean.data_ptr();
        float *var_data = running_var.data_ptr();
        float *weight_data = weight.data_ptr();
        float *bias_data = bias.data_ptr();

        // TODO: begin
        // FIXME: recommend to write kernel functions in batchnorm.cu
        // HACK: to keep consistent with other function names,
        //   recommend to use CamelCase with first word in lower case
        // gpu execution
        // 1. allocate memory and copy data to device
        // unsigned int channel_data_sizes = in_batch_sizes * in_height * in_width;
        // float *single_channel_data;
        // checkCudaErrors(cudaMalloc(&single_channel_data, channel_data_sizes * sizeof(float)));
        // // 2. do the computation
        // for (int i = 0; i < in_channel; i++)
        // {
        //     // construct the array
        //     get_single_channel_data(input_data, single_channel_data, in_batch_sizes, in_channel, in_height, in_width, i);
        //     // 2.1. calculate mean
        //     mean[i] = cal_mean(single_channel_data, channel_data_sizes);
        //     // std::cout << "mean[" << i << "] = " << mean[i] << std::endl;

        //     // 2.2. calculate population standard deviation and sample standard deviation
        //     var[i] = cal_var(single_channel_data, channel_data_sizes, mean[i], in_batch_sizes, in_height, in_width);
        //     // std::cout << "var[" << i << "] = " << var[i] << std::endl;

        //     // 2.3. update data
        //     dim3 dim_grid(in_batch_sizes);
        //     dim3 dim_block(in_height);
        //     BatchNorm2dKernel_update<<<dim_grid, dim_block>>>(single_channel_data, in_width, mean[i], var[i], eps);
        //     // copy the result to output
        //     restore_single_channel_data(input_data, single_channel_data, in_batch_sizes, in_channel, in_height, in_width, i);
        // }
        // checkCudaErrors(cudaFree(single_channel_data));
        // TODO: end

        // HACK: FIXME: this module is currently inplace
        return x;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":BatchNorm2d" << std::endl;
    }
};