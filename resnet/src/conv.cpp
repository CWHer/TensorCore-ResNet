/** @file conv.cpp
*/
#include "functional/conv2d.hpp"
#include "conv.hpp"
#include "functional/gemm.hpp"

using namespace Impl;

Conv2d::Conv2d(int in_channels,
               int out_channels,
               int kernel_size,
               int stride,
               int padding,
               int dilation,
               int groups,
               bool bias)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding),
      dilation(dilation), groups(groups), weight_f16(nullptr) {
  if (dilation != 1 || groups != 1) {
    throw std::runtime_error("Not implemented");
  }

  weight = Tensor({out_channels, in_channels, kernel_size, kernel_size});
  // TODO: store the weight in float16 to increase the performance
  addTensor("weight", weight);

  this->bias = Tensor({out_channels});
  if (bias) {
    addTensor("bias", this->bias);
  } else {
    // Set bias to zeroes
    for (int i = 0; i < this->bias.totalSize(); i++) {
      this->bias.data_ptr()[i] = 0.0f;
    }
  }
}

Tensor Conv2d::forward(const Tensor &x) {
  if (weight_f16 == nullptr) {
    weight_f16 = fp32_array_to_fp16_array(weight.data_ptr(), weight.totalSize(), weight.getDevice());
  }

  return functional::conv2d(x, weight_f16, bias, out_channels, kernel_size, stride, padding, dilation, groups);
}

void Conv2d::printModule(const std::string &prefix) {
  std::cout << prefix << ":Conv2d" << std::endl;
}
void Conv2d::setWeight(const Tensor &new_weight) {
  if (weight_f16 != nullptr) {
    switch (this->weight.getDevice()) {

    case CPU:delete[] weight_f16;
      break;
    case CUDA:cudaFree(weight_f16);
      break;
    }

    weight_f16 = nullptr;
  }

  this->weight = new_weight;
}
void Conv2d::setBias(const Tensor &new_bias) {
  this->bias = new_bias;
}

Conv2d::~Conv2d() {
  if (weight_f16 != nullptr) {
    switch (this->weight.getDevice()) {
    case CPU:delete[] weight_f16;
      break;
    case CUDA:cudaFree(weight_f16);
      break;
    }
  }
}

void Conv2d::to(Impl::DeviceType device) {
  Module::to(device);
  if (bias.getDevice() != device) {
    bias.to(device);
  }
}

// Perform the convert on-size, this could be inefficient
Tensor functional::conv2d(const Tensor &input,
                          const Tensor &weight,
                          const Tensor &bias,
                          int stride,
                          int padding,
                          int dilation,
                          int groups) {
  auto shape_input = input.sizes();

  if (shape_input.size() != 4) {
    throw std::runtime_error("Input must be 4D");
  }

  auto minibatch = shape_input[0];
  auto in_channels = shape_input[1];
  auto in_height = shape_input[2];
  auto in_width = shape_input[3];

  auto shape_weight = weight.sizes();
  auto out_channels = shape_weight[0];
  auto weight_in_channels = shape_weight[1];

  if (in_channels != weight_in_channels) {
    throw std::runtime_error("in_channels != weight in_channels");
  }

  auto kernel_height = shape_weight[2];
  auto kernel_width = shape_weight[3];

  if (kernel_height != kernel_width) {
    throw std::runtime_error("kernel height != kernel width");
  }

  auto shape_output =
      conv2d_result_shape(minibatch, in_channels, in_height, in_width, out_channels, kernel_height, stride, padding);

  auto weight_16 = fp32_array_to_fp16_array(weight.data_ptr(), weight.totalSize(), weight.getDevice());

  auto result =
      functional::conv2d(input, weight_16, bias, out_channels, kernel_height, stride, padding, dilation, groups);

  switch (weight.getDevice()) {
  case Impl::DeviceType::CPU:delete[] weight_16;
    break;
  case Impl::DeviceType::CUDA:cudaFree(weight_16);
  }

  return result;
}

Tensor functional::conv2d(const Tensor &input,
                          const float_16 *weight,
                          const Tensor &bias,
                          int out_channels,
                          int kernel_size,
                          int stride,
                          int padding,
                          int dilation,
                          int groups) {
  auto shape_input = input.sizes();

  if (shape_input.size() != 4) {
    throw std::runtime_error("Input must be 4D");
  }

  auto minibatch = shape_input[0];
  auto in_channels = shape_input[1];
  auto in_height = shape_input[2];
  auto in_width = shape_input[3];

  auto shape_output =
      conv2d_result_shape(minibatch, in_channels, in_height, in_width, out_channels, kernel_size, stride, padding);

  auto result = Tensor(shape_output, input.getDevice(), nullptr);

  ::conv2d(input.data_ptr(),
           result.data_ptr(),
           weight,
           bias.data_ptr(),
           minibatch,
           in_channels,
           in_height,
           in_width,
           out_channels,
           kernel_size,
           stride,
           padding,
           input.getDevice());

  return result;
}
