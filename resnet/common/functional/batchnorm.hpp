/** @file batchnorm.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP

void hostBatchNorm2d(float *input_data, const float *mean_data, const float *var_data,
                     const float *weight_data, const float *bias_data,
                     float eps, unsigned int batch_size, unsigned int num_channels,
                     unsigned int height, unsigned int width);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP
