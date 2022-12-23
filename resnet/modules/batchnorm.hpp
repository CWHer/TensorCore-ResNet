#pragma once

#include "common.h"
#include "module.hpp"
#include "batchnorm.cu"

class BatchNorm2d : public Module
{
private:
    float eps;
    int num_features;
    float *mean_data, *var_data, *weight_data, *bias_data;

public:
    BatchNorm2d(Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, float eps = 1e-5) : eps(eps)
    {
        addTensor("weight", weight);             // [num_features]
        addTensor("bias", bias);                 // [num_features]
        addTensor("running_mean", running_mean); // [num_features]
        addTensor("running_var", running_var);   // [num_features]
        weight_data = weight.data_ptr();
        bias_data = bias.data_ptr();
        mean_data = running_mean.data_ptr();
        var_data = running_var.data_ptr();
        num_features = weight.sizes()[0];
    }

    Tensor forward(Tensor x) override
    {
        checkCppErrorsMsg(x.sizes().size() != 4, "BatchNorm2d only support 4D input");
        checkCppErrorsMsg(x.sizes()[1] != num_features, "BatchNorm2d input channel size mismatch");
        checkCppErrorsMsg(x.getDevice() != DeviceType::CPU,
                          "BatchNorm2d only support CUDA");

        // NOTE: x is in NCHW format
        // NOTE: computation is done channel-wise
        unsigned int batch_size = x.sizes()[0];
        unsigned int num_channels = x.sizes()[1];
        unsigned int height = x.sizes()[2];
        unsigned int width = x.sizes()[3];

        float *input_data = x.data_ptr();
        
        BatchNorm2dKernel(input_data, mean_data, var_data, weight_data, bias_data, eps,
                          batch_size, num_channels, height, width);
        
        return x;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":BatchNorm2d" << std::endl;
    }
};