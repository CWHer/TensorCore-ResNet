/** @file batchnorm.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP

void hostBatchNorm2d(float *input_data, const float *mean_data, const float *var_data,
                     const float *weight_data, const float *bias_data,
                     float eps, int batch_size, int num_channels,
                     int height, int width);

void hostBatchNorm2dRelu(float *input_data, const float *mean_data, const float *var_data,
                         const float *weight_data, const float *bias_data,
                         float eps, int batch_size, int num_channels,
                         int height, int width);
#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP
