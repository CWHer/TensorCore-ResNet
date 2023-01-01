/** @file tensor_conv2d.cpp
*/

#if WITH_TORCH
#include <torch/torch.h>
#endif

#include <gtest/gtest.h>
#include <random>
#include "conv.hpp"

using namespace std;

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

#if WITH_TORCH
TEST(conv2d, tensors) {
  // 56x56x64 -> 56x56x64
  // 3x3 conv with stride 1
  auto batch = 2;
  auto channel = 64;
  auto input_height = 56;
  auto input_width = 56;
  auto stride = 1;
  auto padding = 1;
  auto filter_channel = 64;
  auto filter_size = 3;

  auto output_layers = 64;

  auto input = Impl::Tensor({batch, channel, input_height, input_width});
  auto weight = Impl::Tensor({output_layers, filter_channel, filter_size, filter_size});
  auto bias = Impl::Tensor({output_layers});

  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Randomize the input
  for (int i = 0; i < input.totalSize(); i++) {
    input.data_ptr()[i] = matrix_dist(generator);
  }
  // Randomize the weight
  for (int i = 0; i < weight.totalSize(); i++) {
    weight.data_ptr()[i] = matrix_dist(generator);
  }

  // no bias
  for (int i = 0; i < bias.totalSize(); i++) {
    bias.data_ptr()[i] = 0;
  }

  auto input_at_tensor = torch::from_blob(input.data_ptr(), {batch, channel, input_height, input_width});
  auto
      weight_at_tensor = torch::from_blob(weight.data_ptr(), {output_layers, filter_channel, filter_size, filter_size});
  auto bias_at_tensor = torch::from_blob(bias.data_ptr(), {output_layers});

  auto output_ref_at_tensor =
      torch::conv2d(input_at_tensor, weight_at_tensor, bias_at_tensor, {stride, stride}, {padding, padding});
  auto output = functional::conv2d(input, weight, bias, stride, padding);

  auto layer = Impl::Conv2d(channel, output_layers, filter_size, stride, padding);
  layer.setWeight(weight);
  layer.setBias(bias);

  auto output_layer = layer.forward(input);

  auto output_ref_shape = output_ref_at_tensor.sizes();
  auto output_shape = output.sizes();
  auto output_layer_shape = output_layer.sizes();

  // Assert the shape
  ASSERT_EQ(output_ref_shape.size(), output_shape.size());
  ASSERT_EQ(output_ref_shape.size(), output_layer_shape.size());

  for (size_t i = 0; i < output_ref_shape.size(); i++) {
    ASSERT_EQ(output_ref_shape[i], output_shape[i]);
    ASSERT_EQ(output_ref_shape[i], output_layer_shape[i]);
  }

  auto output_at_tensor = torch::from_blob(output.data_ptr(), output_ref_shape);
  auto output_layer_at_tensor = torch::from_blob(output_layer.data_ptr(), output_ref_shape);

  // Assert the value
  auto tolerance = 1e-1;

  auto close_value = torch::isclose(output_at_tensor, output_ref_at_tensor, 1e0, 1e-1);
  auto close_count = close_value.sum().item<int64_t>();

  ASSERT_LE(output.totalSize() - close_count, output.totalSize() * tolerance);

  close_value = torch::isclose(output_layer_at_tensor, output_at_tensor, 1e0, 1e-1);
  close_count = close_value.sum().item<int64_t>();

  ASSERT_LE(output.totalSize() - close_count, output.totalSize() * tolerance);
}
#endif

#if WITH_TORCH
TEST(conv2d, tensors_cuda) {
  // 56x56x64 -> 56x56x64
  // 3x3 conv with stride 1
  auto batch = 2;
  auto channel = 64;
  auto input_height = 56;
  auto input_width = 56;
  auto stride = 1;
  auto padding = 1;
  auto filter_channel = 64;
  auto filter_size = 3;

  auto output_layers = 64;

  auto input = Impl::Tensor({batch, channel, input_height, input_width});
  auto weight = Impl::Tensor({output_layers, filter_channel, filter_size, filter_size});
  auto bias = Impl::Tensor({output_layers});

  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Randomize the input
  for (int i = 0; i < input.totalSize(); i++) {
    input.data_ptr()[i] = matrix_dist(generator);
  }
  // Randomize the weight
  for (int i = 0; i < weight.totalSize(); i++) {
    weight.data_ptr()[i] = matrix_dist(generator);
  }

  // Randomize the bias
  for (int i = 0; i < bias.totalSize(); i++) {
    bias.data_ptr()[i] = matrix_dist(generator);
  }

  auto input_at_tensor = torch::from_blob(input.data_ptr(), {batch, channel, input_height, input_width});
  auto
      weight_at_tensor = torch::from_blob(weight.data_ptr(), {output_layers, filter_channel, filter_size, filter_size});
  auto bias_at_tensor = torch::from_blob(bias.data_ptr(), {output_layers});

  auto output_ref_at_tensor =
      torch::conv2d(input_at_tensor, weight_at_tensor, bias_at_tensor, {stride, stride}, {padding, padding});

  input.to(Impl::DeviceType::CUDA);
  weight.to(Impl::DeviceType::CUDA);
  bias.to(Impl::DeviceType::CUDA);

  auto output = functional::conv2d(input, weight, bias, stride, padding);
  output.to(Impl::DeviceType::CPU);

  auto layer = Impl::Conv2d(channel, output_layers, filter_size, stride, padding);
  layer.setWeight(weight);
  layer.setBias(bias);

  auto output_layer = layer.forward(input);
  output_layer.to(Impl::DeviceType::CPU);

  auto output_ref_shape = output_ref_at_tensor.sizes();
  auto output_shape = output.sizes();
  auto output_layer_shape = output_layer.sizes();

  // Assert the shape
  ASSERT_EQ(output_ref_shape.size(), output_shape.size());
  ASSERT_EQ(output_ref_shape.size(), output_layer_shape.size());

  for (size_t i = 0; i < output_ref_shape.size(); i++) {
    ASSERT_EQ(output_ref_shape[i], output_shape[i]);
    ASSERT_EQ(output_ref_shape[i], output_layer_shape[i]);
  }

  auto output_at_tensor = torch::from_blob(output.data_ptr(), output_ref_shape);
  auto output_layer_at_tensor = torch::from_blob(output_layer.data_ptr(), output_ref_shape);

  // Assert the value
  auto tolerance = 1e-1;

  auto close_value = torch::isclose(output_at_tensor, output_ref_at_tensor, 1e0, 1e-1);
  auto close_count = close_value.sum().item<int64_t>();

  ASSERT_LE(output.totalSize() - close_count, output.totalSize() * tolerance);

  close_value = torch::isclose(output_layer_at_tensor, output_at_tensor, 1e0, 1e-1);
  close_count = close_value.sum().item<int64_t>();

  ASSERT_LE(output.totalSize() - close_count, output.totalSize() * tolerance);
}
#endif
