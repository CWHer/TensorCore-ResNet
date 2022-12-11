/** @file im2col.cu
*/
#include <gtest/gtest.h>
#include "functional/conv2d.hpp"

TEST(conv2d, im2col_naive_impl) {
  int input_C = 2;
  int input_H = 4;
  int input_W = 4;

  int filter_height = 2;
  int filter_width = 2;
  int stride = 1;
  int padding = 0;

  // Reference result
  float ref_result[] = {
      1, 2, 5, 6, 17, 18, 21, 22, 2, 3, 6, 7, 18, 19, 22, 23, 3, 4, 7, 8, 19, 20, 23, 24, 5, 6, 9, 10, 21, 22, 25, 26,
      6, 7, 10, 11, 22, 23, 26, 27, 7, 8, 11, 12, 23, 24, 27, 28, 9, 10, 13, 14, 25, 26, 29, 30, 10, 11, 14, 15, 26, 27,
      30, 31, 11, 12, 15, 16, 27, 28, 31, 32
  };

  auto im2col_result =
      create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  auto input = std::make_unique<float[]>(input_C * input_H * input_W);
  // Fill input with increasing values.
  for (int i = 0; i < input_C * input_H * input_W; i++) {
    input[i] = static_cast<float>(i + 1);
  }

  im2col_naive(input.get(),
               im2col_result.get(),
               input_C,
               input_H,
               input_W,
               filter_height,
               filter_width,
               stride,
               padding);

  // Verify result.
  auto result_size = im2col_result_size(input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  for (int i = 0; i < result_size; i++) {
    EXPECT_NEAR(im2col_result[i], ref_result[i], 1e-3);
  }

}

TEST(conv2d, im2col_cuda_impl) {
  int input_C = 2;
  int input_H = 4;
  int input_W = 4;

  int filter_height = 2;
  int filter_width = 2;
  int stride = 1;
  int padding = 0;

  auto im2col_result =
      create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  auto ref_result = create_im2col_result_store(input_C, input_H, input_W, filter_height, filter_width, stride, padding);

  auto input = std::make_unique<float[]>(input_C * input_H * input_W);
  // Fill input with increasing values.
  for (int i = 0; i < input_C * input_H * input_W; i++) {
    input[i] = static_cast<float>(i + 1);
  }

  im2col_cuda(input.get(),
              im2col_result.get(),
              input_C,
              input_H,
              input_W,
              filter_height,
              filter_width,
              stride,
              padding);

  im2col_naive(input.get(), ref_result.get(), input_C, input_H, input_W, filter_height, filter_width, stride, padding);

// Verify result.
  auto result_size = im2col_result_size(input_C, input_H, input_W, filter_height, filter_width, stride, padding);
  for (int i = 0; i < result_size; i++) {
    EXPECT_NEAR(im2col_result[i], ref_result[i], 1e-3);
  }

}

TEST(conv2d, conv2d_impl) {
  int input_C = 2;
  int input_H = 4;
  int input_W = 4;

  int filter_height = 2;
  int filter_width = 2;
  int stride = 1;
  int padding = 0;



}
