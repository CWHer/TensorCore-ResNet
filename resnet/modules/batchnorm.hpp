#pragma once

#include "common.h"
#include "module.hpp"

#include "batchnorm.cuh"

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
        int batch_size = x.sizes()[0];
        int num_channels = x.sizes()[1];
        int height = x.sizes()[2];
        int width = x.sizes()[3];

        float *input_data = x.data_ptr();
        float *mean_data = running_mean.data_ptr();
        float *var_data = running_var.data_ptr();
        float *weight_data = weight.data_ptr();
        float *bias_data = bias.data_ptr();

        hostBatchNorm2d(input_data, mean_data, var_data, weight_data, bias_data,
                        eps, batch_size, num_channels, height, width);

        // HACK: FIXME: this module is currently inplace
        return x;
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":BatchNorm2d" << std::endl;
    }
};