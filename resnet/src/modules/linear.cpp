/** @file linear.cpp
*/
#include "linear.hpp"
#include "functional/linear.hpp"
#include "functional/gemm.hpp"
#include "mem_pool.h"

using namespace Impl;

Impl::Linear::Linear(int in_features, int out_features) : weight({out_features, in_features}), bias({out_features}),
                                                          in_features(in_features), out_features(out_features),
                                                          weight_f16(nullptr) {
  addTensor("weight", weight);
  addTensor("bias", bias);
}

Impl::Tensor functional::linear(const Impl::Tensor &input, const Impl::Tensor &weight, const Impl::Tensor &bias) {
  float_16 *weight_f16 = nullptr;
  auto weight_shape = weight.sizes();
#if DEBUG
  if (weight_shape.size() != 2) {
    throw std::invalid_argument("weight must be 2D tensor");
  }
#endif
  switch (input.getDevice()) {
  case Impl::DeviceType::CPU:weight_f16 = new float_16[weight.totalSize()];
    break;
  case Impl::DeviceType::CUDA:Impl::cudaPooledMalloc(&weight_f16, weight.totalSize() * sizeof(float_16));
    break;
  }
  prepare_linear_weight(weight.data_ptr(), weight_f16, weight_shape[0], weight_shape[1], weight.getDevice());

  auto result = functional::linear(input, weight_f16, bias);

  switch (input.getDevice()) {
  case Impl::DeviceType::CPU:delete[] weight_f16;
    break;
  case Impl::DeviceType::CUDA:cudaPooledFree(weight_f16);
    break;
  }
  return result;
}

Impl::Tensor functional::linear(const Tensor &input, const float_16 *weight, const Tensor &bias) {
  auto input_shape = input.sizes();
  auto bias_shape = bias.sizes();

  auto batch_size = input_shape[0];
  auto input_channel = input_shape[1];
  auto output_channel = bias_shape[0];

  auto result = Impl::Tensor({batch_size, output_channel});
  result.to(input.getDevice());

  auto input_fp16 = fp32_array_to_fp16_array(input.data_ptr(), input.totalSize(), input.getDevice());

  ::linear(input_fp16,
           result.data_ptr(),
           weight,
           bias.data_ptr(),
           batch_size,
           input_channel,
           output_channel,
           input.getDevice());

  switch (input.getDevice()) {
  case Impl::DeviceType::CPU:delete[] input_fp16;
    break;
  case Impl::DeviceType::CUDA:cudaPooledFree(input_fp16);
    break;
  }
  return result;
}

Impl::Tensor Impl::Linear::forward(Impl::Tensor &&x) {
  return std::move(forward(x));
}

Impl::Tensor Impl::Linear::forward(const Tensor &x) {
  if (weight_f16 == nullptr) {
    switch (x.getDevice()) {
    case Impl::DeviceType::CPU:weight_f16 = new float_16[weight.totalSize()];
      break;
    case Impl::DeviceType::CUDA:Impl::cudaPooledMalloc(&weight_f16, weight.totalSize() * sizeof(float_16));
      break;
    }
    prepare_linear_weight(weight.data_ptr(), weight_f16, out_features, in_features, x.getDevice());
  }

#if DEBUG
  // x should be a 2D tensor
  auto x_shape = x.sizes();

  if (x_shape.size() != 2) {
    throw std::runtime_error("x should be a 2D tensor");
  }

  auto batches = x_shape[0];
  auto x_features = x_shape[1];

  if (x_features != in_features) {
    throw std::runtime_error("x_features != in_features");
  }
#endif
  timer.start("forward");
  auto result = std::move(functional::linear(x, weight_f16, bias));
  timer.end("forward");
  return std::move(result);
}

void Impl::Linear::printModule(const std::string &prefix) {
  std::cout << prefix << ":Linear" << std::endl;
}
void Linear::printStat(const std::string &prefix) {
  printModule(prefix);
  timer.printStat("forward");
}
Impl::Linear::~Linear() {
  if (weight_f16 != nullptr) {
    switch (weight.getDevice()) {
    case DeviceType::CPU:delete[] weight_f16;
      break;
    case DeviceType::CUDA:cudaPooledFree(weight_f16);
      break;
    }
  }
}
void Impl::Linear::setWeight(const Impl::Tensor &new_weight) {
  if (weight_f16 != nullptr) {
    switch (weight.getDevice()) {
    case DeviceType::CPU:delete[] weight_f16;
      break;
    case DeviceType::CUDA:cudaPooledFree(weight_f16);
      break;
    }
    weight_f16 = nullptr;
  }
  weight = new_weight;
}

void Impl::Linear::setBias(const Impl::Tensor &new_bias) {
  bias = new_bias;
}


