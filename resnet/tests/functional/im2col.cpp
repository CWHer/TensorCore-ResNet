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
                         int N,
                         int C,
                         int H,
                         int W,
                         int filter_height,
                         int filter_width,
                         int stride,
                         int padding) {
  // Allocate a tensor of shape (N, C, H, W)
  auto input_tensor = torch::from_blob(input, {N, C, H, W}, torch::kFloat32);
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

/**
 * @brief im2col, performs as the matlab function.
 *
 * @param input float of shape (N, C, H, W)
 * @param output Output should be shaped as:
 * (N * C * filter_height * filter_width, output_height * output_width)
 *
 * where output_height = (H + 2 * padding - filter_height) / stride + 1
 * and output_width = (W + 2 * padding - filter_width) / stride + 1
 *
 * Data are arranged per channel of columns, this results in factor C.
 */
void im2col_naive(float *input,
                  float_16 *output,
                  int N,
                  int C,
                  int H,
                  int W,
                  int filter_height,
                  int filter_width,
                  int stride,
                  int padding) {
  int output_height = (H + 2 * padding - filter_height) / stride + 1;
  int output_width = (W + 2 * padding - filter_width) / stride + 1;
  int output_size = output_height * output_width;

  int filter_size = filter_height * filter_width;
  int output_ldm = output_size;
  int input_channel_size = H * W;

  for (int batch_n = 0; batch_n < N; ++batch_n) {
    for (int filter_h = 0; filter_h < filter_height; ++filter_h) {
      for (int filter_w = 0; filter_w < filter_width; ++filter_w) {
        for (int output_h = 0; output_h < output_height; ++output_h) {
          for (int output_w = 0; output_w < output_width; ++output_w) {
            for (int c = 0; c < C; ++c) {
              int index_h = output_h * stride + filter_h - padding;
              int index_w = output_w * stride + filter_w - padding;
              int input_index = (c + batch_n * C) * input_channel_size + index_h * W + index_w;
              int output_index = (c + batch_n * C) * filter_size + filter_h * filter_width + filter_w;
              int output_offset = output_h * output_width + output_w;
              if (index_h >= 0 && index_h < H && index_w >= 0 && index_w < W) {
                output[output_index * output_ldm + output_offset] = input[input_index];
              } else {
                output[output_index * output_ldm + output_offset] = __half(0.0f);
              }
            }
          }
        }
      }
    }
  }
}

#if WITH_TORCH
TEST(conv2d, im2col_naive_impl_correctness_1) {
  constexpr int input_N = 2;
  constexpr int input_C = 2;
  constexpr int input_H = 4;
  constexpr int input_W = 4;

  constexpr int filter_height = 2;
  constexpr int filter_width = 2;
  constexpr int stride = 1;
  constexpr int padding = 0;

  // Reference result
  auto ref_result =
      create_im2col_result_store_host(input_N,
                                      input_C,
                                      input_H,
                                      input_W,
                                      filter_height,
                                      filter_width,
                                      stride,
                                      padding);
  auto result =
      create_im2col_result_store_host(input_N,
                                      input_C,
                                      input_H,
                                      input_W,
                                      filter_height,
                                      filter_width,
                                      stride,
                                      padding);

  auto input = std::make_unique<float[]>(input_N * input_C * input_H * input_W);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Fill input with increasing values.
  for (int i = 0; i < input_N * input_C * input_H * input_W; i++) {
    input[i] = matrix_dist(generator);
  }

  im2col_naive(input.get(),
               result.get(),
               input_N,
               input_C,
               input_H,
               input_W,
               filter_height,
               filter_width,
               stride,
               padding);
  im2col_torch(input.get(),
               ref_result.get(),
               input_N,
               input_C,
               input_H,
               input_W,
               filter_height,
               filter_width,
               stride,
               padding);

  // Verify result.
  auto result_size =
      im2col_result_size(input_N, input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  for (size_t i = 0; i < result_size; i++) {
    EXPECT_NEAR(result[i], ref_result[i], 1e-3);
  }
}

TEST(conv2d, im2col_naive_impl_correctness_2) {
  int input_N = 16;
  int input_C = 3;
  int input_H = 32;
  int input_W = 32;

  int filter_height = 3;
  int filter_width = 3;
  int stride = 2;
  int padding = 4;

  // Reference result
  auto ref_result =
      create_im2col_result_store_host(input_N,
                                      input_C,
                                      input_H,
                                      input_W,
                                      filter_height,
                                      filter_width,
                                      stride,
                                      padding);
  auto result =
      create_im2col_result_store_host(input_N,
                                      input_C,
                                      input_H,
                                      input_W,
                                      filter_height,
                                      filter_width,
                                      stride,
                                      padding);

  auto input = std::make_unique<float[]>(input_N * input_C * input_H * input_W);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Fill input with increasing values.
  for (int i = 0; i < input_N * input_C * input_H * input_W; i++) {
    input[i] = matrix_dist(generator);
  }

  im2col_naive(input.get(),
               result.get(),
               input_N,
               input_C,
               input_H,
               input_W,
               filter_height,
               filter_width,
               stride,
               padding);
  im2col_torch(input.get(),
               ref_result.get(),
               input_N,
               input_C,
               input_H,
               input_W,
               filter_height,
               filter_width,
               stride,
               padding);

  // Verify result.
  auto result_size =
      im2col_result_size(input_N, input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  for (size_t i = 0; i < result_size; i++) {
    EXPECT_NEAR(result[i], ref_result[i], 1e-3);
  }
}
#endif

TEST(conv2d, im2col_cuda_impl) {
  int input_N = 16;
  int input_C = 3;
  int input_H = 32;
  int input_W = 32;

  int filter_height = 3;
  int filter_width = 3;
  int stride = 2;
  int padding = 4;
// Reference result
  auto ref_result =
      create_im2col_result_store_host(input_N,
                                      input_C,
                                      input_H,
                                      input_W,
                                      filter_height,
                                      filter_width,
                                      stride,
                                      padding);
  auto result =
      create_im2col_result_store_host(input_N,
                                      input_C,
                                      input_H,
                                      input_W,
                                      filter_height,
                                      filter_width,
                                      stride,
                                      padding);

  auto input = std::make_unique<float[]>(input_N * input_C * input_H * input_W);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);
  // Fill input with increasing values.
  for (int i = 0; i < input_N * input_C * input_H * input_W; i++) {
    input[i] = matrix_dist(generator);
  }

  im2col_naive(input.get(),
               result.get(),
               input_N,
               input_C,
               input_H,
               input_W,
               filter_height,
               filter_width,
               stride,
               padding);
  im2col(input.get(),
         ref_result.get(),
         input_N,
         input_C,
         input_H,
         input_W,
         filter_height,
         filter_width,
         stride,
         padding,
         Impl::DeviceType::CPU);

  // Verify result.
  auto result_size =
      im2col_result_size(input_N, input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  for (size_t i = 0; i < result_size; i++) {
    EXPECT_NEAR(result[i], ref_result[i], 1e-3);
  }
}
