/** @file linear.cpp
*/

#if WITH_TORCH
#include <torch/torch.h>
#endif

#include <gtest/gtest.h>
#include <random>
#include "linear.hpp"
#include "functional/linear.hpp"

using namespace std;

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

#if WITH_TORCH
TEST(linear, functional) {
  auto batch = 16;
  auto input_size = 256;
  auto output_size = 1000;

  auto input = Impl::Tensor({batch, input_size});
  auto weight = Impl::Tensor({output_size, input_size});
  auto bias = Impl::Tensor({output_size});

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

  auto output = functional::linear(input, weight, bias);

  auto input_at_tensor = torch::from_blob(input.data_ptr(), {batch, input_size}, torch::kFloat32);
  auto weight_at_tensor = torch::from_blob(weight.data_ptr(), {output_size, input_size}, torch::kFloat32);
  auto bias_at_tensor = torch::from_blob(bias.data_ptr(), {output_size}, torch::kFloat32);
  auto output_ref_at_tensor = torch::linear(input_at_tensor, weight_at_tensor, bias_at_tensor);

  input.to(Impl::DeviceType::CUDA);
  weight.to(Impl::DeviceType::CUDA);
  bias.to(Impl::DeviceType::CUDA);

  auto output_d = functional::linear(input, weight, bias);
  output_d.to(Impl::DeviceType::CPU);

  auto tolerance = 1.0e-1;

  auto output_at_tensor = torch::from_blob(output.data_ptr(), {batch, output_size});
  auto output_d_at_tensor = torch::from_blob(output_d.data_ptr(), {batch, output_size});

  auto close = torch::isclose(output_ref_at_tensor, output_at_tensor, 1e0, 5e-1);
  auto close_count = close.sum().item<int64_t>();

  EXPECT_LE(output.totalSize() - close_count, output.totalSize() * tolerance)
          << "Reference and output are not close enough";

  close = torch::isclose(output_at_tensor, output_d_at_tensor, 1e0, 5e-1);
  close_count = close.sum().item<int64_t>();

  EXPECT_LE(output.totalSize() - close_count, output.totalSize() * tolerance)
          << "CUDA and CPU results are not close enough";

}
#endif

#if WITH_TORCH
TEST(linear, layer) {
  auto batch = 16;
  auto input_size = 256;
  auto output_size = 1000;

  auto input = Impl::Tensor({batch, input_size});
  auto weight = Impl::Tensor({output_size, input_size});
  auto bias = Impl::Tensor({output_size});

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

  auto layer = Impl::Linear(input_size, output_size);
  layer.setWeight(weight);
  layer.setBias(bias);

  auto output = layer.forward(input);

  auto input_at_tensor = torch::from_blob(input.data_ptr(), {batch, input_size}, torch::kFloat32);
  auto weight_at_tensor = torch::from_blob(weight.data_ptr(), {output_size, input_size}, torch::kFloat32);
  auto bias_at_tensor = torch::from_blob(bias.data_ptr(), {output_size}, torch::kFloat32);
  auto output_ref_at_tensor = torch::linear(input_at_tensor, weight_at_tensor, bias_at_tensor);

  input.to(Impl::DeviceType::CUDA);
  weight.to(Impl::DeviceType::CUDA);
  bias.to(Impl::DeviceType::CUDA);

  auto layer_d = Impl::Linear(input_size, output_size);
  layer_d.setWeight(weight);
  layer_d.setBias(bias);

  auto output_d = layer_d.forward(input);
  output_d.to(Impl::DeviceType::CPU);

  auto tolerance = 1.0e-1;
  auto output_at_tensor = torch::from_blob(output.data_ptr(), {batch, output_size});
  auto output_d_at_tensor = torch::from_blob(output_d.data_ptr(), {batch, output_size});

  auto close = torch::isclose(output_ref_at_tensor, output_at_tensor, 1e0, 5e-1);
  auto close_count = close.sum().item<int64_t>();

  EXPECT_LE(output.totalSize() - close_count, output.totalSize() * tolerance)
          << "Reference and output are not close enough";

  close = torch::isclose(output_at_tensor, output_d_at_tensor, 1e0, 5e-1);
  close_count = close.sum().item<int64_t>();

  EXPECT_LE(output.totalSize() - close_count, output.totalSize() * tolerance)
          << "CUDA and CPU results are not close enough";

}
#endif
