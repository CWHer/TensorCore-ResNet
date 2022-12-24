/** @file batchnorm.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP
#define TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP

void hostBatchNorm2d(float *input_data, float *mean_data, float *var_data,
                     float *weight_data, float *bias_data,
                     float eps, unsigned int batch_size, unsigned int num_channels,
                     unsigned int height, unsigned int width);

#endif //TENSORCORE_RESNET_COMMON_FUNCTIONAL_BATCHNORM_HPP
