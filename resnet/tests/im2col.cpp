/** @file im2col.cu
*/
#include <gtest/gtest.h>
#include "functional/conv2d.hpp"
#include <random>

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

#if WITH_TORCH
#include <torch/torch.h>
#endif

using namespace std;

#if WITH_TORCH
static void im2col_torch(float *input,
                         float_16 *output,
                         int C,
                         int H,
                         int W,
                         int filter_height,
                         int filter_width,
                         int stride,
                         int padding) {
  // Allocate a tensor of shape (C, H, W)
  auto input_tensor = torch::from_blob(input, {C, H, W}, torch::kFloat32);
  // Dilation should be 1x1, according to
  // https://stackoverflow.com/questions/43474072/default-dilation-value-in-pytorch
  auto result = at::im2col(input_tensor, {filter_height, filter_width}, {1, 1}, {padding, padding}, {stride, stride});

  // Copy the result to output
  auto result_data = result.data_ptr<float>();
  // Should do this to get float_32 to float_16.
  for (int i = 0; i < result.numel(); i++) {
    output[i] = result_data[i];
  }
}
#endif

#if WITH_TORCH
TEST(conv2d, im2col_naive_impl_correctness_1) {
  int input_C = 2;
  int input_H = 4;
  int input_W = 4;

  int filter_height = 2;
  int filter_width = 2;
  int stride = 1;
  int padding = 0;

  // Reference result
  auto ref_result = create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  auto result = create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  auto input = std::make_unique<float[]>(input_C * input_H * input_W);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Fill input with increasing values.
  for (int i = 0; i < input_C * input_H * input_W; i++) {
    input[i] = matrix_dist(generator);
  }

  im2col_naive(input.get(), result.get(), input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  im2col_torch(input.get(), ref_result.get(), input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  // Verify result.
  auto result_size = im2col_result_size(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  for (int i = 0; i < result_size; i++) {
    EXPECT_NEAR(result[i], ref_result[i], 1e-3);
  }
}

TEST(conv2d, im2col_naive_impl_correctness_2) {
  int input_C = 3;
  int input_H = 32;
  int input_W = 32;

  int filter_height = 3;
  int filter_width = 3;
  int stride = 2;
  int padding = 4;

  // Reference result
  auto ref_result = create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  auto result = create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  auto input = std::make_unique<float[]>(input_C * input_H * input_W);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Fill input with increasing values.
  for (int i = 0; i < input_C * input_H * input_W; i++) {
    input[i] = matrix_dist(generator);
  }

  im2col_naive(input.get(), result.get(), input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  im2col_torch(input.get(), ref_result.get(), input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  // Verify result.
  auto result_size = im2col_result_size(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  for (int i = 0; i < result_size; i++) {
    EXPECT_NEAR(result[i], ref_result[i], 1e-3);
  }
}
#endif

TEST(conv2d, im2col_cuda_impl) {
  int input_C = 3;
  int input_H = 32;
  int input_W = 32;

  int filter_height = 3;
  int filter_width = 3;
  int stride = 2;
  int padding = 4;

  auto result = create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  auto ref_result = create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  auto input = std::make_unique<float[]>(input_C * input_H * input_W);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Fill input with increasing values.
  for (int i = 0; i < input_C * input_H * input_W; i++) {
    input[i] = matrix_dist(generator);
  }

  im2col_cuda(input.get(), result.get(), input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  im2col_naive(input.get(), ref_result.get(), input_C, input_H, input_W, filter_height, filter_width, stride, padding);

// Verify result.
  auto result_size = im2col_result_size(input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  for (int i = 0; i < result_size; i++) {
    EXPECT_NEAR(result[i], ref_result[i], 1e-3);
  }
}
