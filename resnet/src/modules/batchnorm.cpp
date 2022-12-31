/** @file batchnorm.cpp
*/
#include "batchnorm.hpp"
#include "functional/batchnorm.hpp"

Impl::BatchNorm2d::BatchNorm2d(int num_features, float eps, [[gnu::unused]] float momentum, [[gnu::unused]] bool affine, [[gnu::unused]] bool track_running_stats)
    : num_features(num_features), eps(eps), weight({num_features}), bias({num_features}),
      running_mean({num_features}), running_var({num_features})
{
  addTensor("weight", weight);             // [num_features]
  addTensor("bias", bias);                 // [num_features]
  addTensor("running_mean", running_mean); // [num_features]
  addTensor("running_var", running_var);   // [num_features]
}

Impl::BatchNorm2dRelu::BatchNorm2dRelu(int num_features,
                                       float eps,
                                       [[gnu::unused]] float momentum,
                                       [[gnu::unused]] bool affine,
                                       [[gnu::unused]] bool track_running_stats)
    : num_features(num_features), eps(eps), weight({num_features}), bias({num_features}), running_mean({num_features}),
      running_var({num_features}) {
  addTensor("weight", weight);             // [num_features]
  addTensor("bias", bias);                 // [num_features]
  addTensor("running_mean", running_mean); // [num_features]
  addTensor("running_var", running_var);   // [num_features]
}


Impl::Tensor Impl::BatchNorm2d::forward(Impl::Tensor &&x) {
  // If CPU
  auto x_device = x.getDevice();
  // HACK: to support CPU input based batchnorm
  // HUGE OVERHEAD
  if (x_device == DeviceType::CPU) {
    x.to(Impl::DeviceType::CUDA);
    this->to(Impl::DeviceType::CUDA);
  }

  // NOTE: x is in NCHW format
  // NOTE: computation is done channel-wise
  auto batch_size = x.sizes()[0];
  auto num_channels = x.sizes()[1];
  auto height = x.sizes()[2];
  auto width = x.sizes()[3];

  float *input_data = x.data_ptr();
  float *mean_data = running_mean.data_ptr();
  float *var_data = running_var.data_ptr();
  float *weight_data = weight.data_ptr();
  float *bias_data = bias.data_ptr();

  hostBatchNorm2d(input_data, mean_data, var_data, weight_data, bias_data,
                  eps, batch_size, num_channels, height, width);
  // When passed by value, the tensor is copied and can be safely modified
  if (x_device == DeviceType::CPU) {
    x.to(Impl::DeviceType::CPU);
    this->to(Impl::DeviceType::CPU);
  }
  return std::move(x);
}

Impl::Tensor Impl::BatchNorm2dRelu::forward(Impl::Tensor &&x) {
  // If CPU
  auto x_device = x.getDevice();
  // HACK: to support CPU input based batchnorm
  // HUGE OVERHEAD
  if (x_device == DeviceType::CPU) {
    x.to(Impl::DeviceType::CUDA);
    this->to(Impl::DeviceType::CUDA);
  }

  // NOTE: x is in NCHW format
  // NOTE: computation is done channel-wise
  auto batch_size = x.sizes()[0];
  auto num_channels = x.sizes()[1];
  auto height = x.sizes()[2];
  auto width = x.sizes()[3];

  float *input_data = x.data_ptr();
  float *mean_data = running_mean.data_ptr();
  float *var_data = running_var.data_ptr();
  float *weight_data = weight.data_ptr();
  float *bias_data = bias.data_ptr();

  hostBatchNorm2dRelu(input_data, mean_data, var_data, weight_data, bias_data,
                  eps, batch_size, num_channels, height, width);
  // When passed by value, the tensor is copied and can be safely modified
  if (x_device == DeviceType::CPU) {
    x.to(Impl::DeviceType::CPU);
    this->to(Impl::DeviceType::CPU);
  }
  return std::move(x);
}


Impl::Tensor Impl::BatchNorm2d::forward(const Tensor &x) {
  checkCppErrorsMsg(x.sizes().size() != 4, "BatchNorm2d only support 4D input");
  checkCppErrorsMsg(x.sizes()[1] != num_features, "BatchNorm2d input channel size mismatch");

  Tensor result = x;
  return std::move(forward(std::move(result)));
}


Impl::Tensor Impl::BatchNorm2dRelu::forward(const Impl::Tensor &x) {
  checkCppErrorsMsg(x.sizes().size() != 4, "BatchNorm2d only support 4D input");
  checkCppErrorsMsg(x.sizes()[1] != num_features, "BatchNorm2d input channel size mismatch");

  Tensor result = x;
  return std::move(forward(std::move(result)));
}


void Impl::BatchNorm2d::printModule(const std::string &prefix) {
  std::cout << prefix << ":BatchNorm2d" << std::endl;
}

void Impl::BatchNorm2dRelu::printModule(const std::string &prefix) {
  std::cout << prefix << ":BatchNorm2d" << std::endl;
}
