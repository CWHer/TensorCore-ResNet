/** @file conv2d.cu
*/

#include <gtest/gtest.h>
#include <sstream>
#include <random>
#include "functional/conv2d.hpp"
#include "common.h"

#if WITH_TORCH
#include <torch/torch.h>
#endif

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

using namespace std;

/**
 * @brief Using Python to run a given script and return the output.
 *
 * @param script Python script string, should be same as just reading a Python file (i.e. lines separated by LF)
 *
 * @return Result of a python script
 */
static string python_command_invoke(const string &script) {
  int in_fd[2];
  int out_fd[2];
  pid_t pid;

  if (pipe(in_fd) == -1) {
    perror("pipe");
    exit(EXIT_FAILURE);
  }

  if (pipe(out_fd) == -1) {
    perror("pipe");
    exit(EXIT_FAILURE);
  }

  pid = fork();
  if (pid == -1) {
    perror("fork");
    exit(EXIT_FAILURE);
  }

  if (pid == 0) {
    // Child process.
    // redirect stdin and stdout to the pipe.
    if (dup2(in_fd[0], STDIN_FILENO) == -1) {
      perror("dup2");
      exit(EXIT_FAILURE);
    }

    if (dup2(out_fd[1], STDOUT_FILENO) == -1) {
      perror("dup2");
      exit(EXIT_FAILURE);
    }
    // close unused pipe ends.
    if (close(in_fd[1])) {
      perror("close");
      exit(EXIT_FAILURE);
    }
    if (close(out_fd[0])) {
      perror("close");
      exit(EXIT_FAILURE);
    }

    // swap to python
    execlp("python", "python", NULL);
    // Error if failed (exec never returns)
    perror("execlp");
    exit(EXIT_FAILURE);
  }

  // close unused end in parent end
  if (close(in_fd[0])) {
    perror("close");
    exit(EXIT_FAILURE);
  }
  if (close(out_fd[1])) {
    perror("close");
    exit(EXIT_FAILURE);
  }

  // write the script to the child process
  ssize_t written = write(in_fd[1], script.c_str(), script.size());
  if (written != script.size() || written == -1) {
    perror("write");
    exit(EXIT_FAILURE);
  }
  if (close(in_fd[1])) {
    perror("close");
    exit(EXIT_FAILURE);
  }

  // read the result from the child process
  stringstream ss;
  char buf[1024];
  ssize_t n;
  while (true) {
    n = read(out_fd[0], buf, sizeof(buf));
    if (n < 0) {
      perror("read");
      exit(EXIT_FAILURE);
    }
    if (n == 0)
      break;

    ss.write(buf, n);
  }
  if (close(out_fd[0])) {
    perror("close");
    exit(EXIT_FAILURE);
  }

  // wait for the child process to exit
  int status;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status)) {
    if (WEXITSTATUS(status) != 0) {
      throw std::runtime_error("child exited with status " + to_string(WEXITSTATUS(status)));
    }
  }

  return ss.str();

}

#if WITH_TORCH
/** @brief Convolutional layer forward propagation performed with libtorch.
 *
 * @param input Input float, row major (organizes at last element), of shape (C, H, W).
 * @param output Result float, row major (organizes at last element), of shape (out_channels, H_out, W_out).
 *  where H_out = floor((H + 2 * padding - kernel_size) / stride) + 1
 *  and W_out = floor((W + 2 * padding - kernel_size) / stride) + 1
 *  @param weight Weight float_16, row major (organizes at last element), of shape (out_channels, C, kernel_size, kernel_size).
 *  @param bias Bias float, row major (organizes at last element), in shape of (out_channels).
 *  @param C Number of input channels.
 *  @param H Height of input.
 *  @param W Width of input.
 *  @param kernel_size Size of kernel (kernel_size x kernel_size).
 *  @param stride Stride of convolution.
 *  @param padding Padding of convolution.
 */
static void conv2d_torch(float *input,
                         float *output,
                         float_16 *weight,
                         float *bias,
                         int N,
                         int C,
                         int H,
                         int W,
                         int out_channels,
                         int kernel_size,
                         int stride,
                         int padding) {
  auto input_tensor = torch::from_blob(input, {N, C, H, W}).to(torch::kFloat32);
  torch::nn::Conv2dOptions options(C, out_channels, kernel_size);
  options = options.stride(stride);
  options = options.padding(padding);
  auto conv_ref_l = torch::nn::Conv2d(options);
  // as torch cannot handle the float_16  correctly
  // sic. Input type (CPUFloatType) and weight type (CPUHalfType) should be the same
  conv_ref_l->weight =
      torch::from_blob(weight, {out_channels, C, kernel_size, kernel_size}, torch::kFloat16).to(torch::kFloat32);
  conv_ref_l->bias = torch::from_blob(bias, {out_channels});
  auto output_tensor = conv_ref_l->forward(input_tensor);
  // Assert the output size is correct.
  auto shape = output_tensor.sizes();
  auto ref_shape = conv2d_result_shape(N, C, H, W, out_channels, kernel_size, stride, padding);
  for (int i = 0; i < shape.size(); i++) {
    assert(shape[i] == ref_shape[i]);
  }
  memcpy(output, output_tensor.data_ptr(), output_tensor.numel() * sizeof(float));
}
#endif

/**
 * @copydoc conv2d_torch

 * @brief Convolutional layer forward propagation performed by invoke Python commands directly
 */
static void conv2d_torch_python_script(float *input,
                                       float *output,
                                       float_16 *weight,
                                       float *bias,
                                       int C,
                                       int H,
                                       int W,
                                       int out_channels,
                                       int kernel_size,
                                       int stride,
                                       int padding) {
  std::stringstream ss;

  ss << "import torch" << std::endl;
  ss << "import numpy as np" << std::endl;
  ss << "import torch.nn as nn" << std::endl;
  ss << "import torch.nn.functional as F" << std::endl;

  // Input tensor
  ss << "input = torch.from_numpy(np.array(" << std::endl;
  ss << "[";
  for (int i = 0; i < C * H * W; i++) {
    ss << input[i] << ", ";
  }
  ss << "]).reshape(1, " << C << ", " << H << ", " << W << "))" << std::endl;

  // Weight tensor
  ss << "weight = torch.from_numpy(np.array(" << std::endl;
  ss << "[";
  for (int i = 0; i < out_channels * C * kernel_size * kernel_size; i++) {
    ss << __half2float(weight[i]) << ", ";
  }
  ss << "]).reshape(" << out_channels << ", " << C << ", " << kernel_size << ", " << kernel_size << "))" << std::endl;

  // Bias tensor
  ss << "bias = torch.from_numpy(np.array(" << std::endl;
  ss << "[";
  for (int i = 0; i < out_channels; i++) {
    ss << bias[i] << ", ";
  }
  ss << "]).reshape(" << out_channels << "))" << std::endl;

  // nn.conv2d
  ss << "result = F.conv2d(input, weight, bias, stride=" << stride << ", padding=" << padding << ")" << std::endl;

  // flatten result
  ss << "result = result.flatten()" << std::endl;
  ss << "result = [str(float(a)) for a in result]" << std::endl;

  // Print result
  ss << "print(\" \".join(result))" << std::endl;

  string result = python_command_invoke(ss.str());

  // result are separated by space
  std::stringstream ss_result(result);
  float result_value;

  for (int i = 0; i < conv2d_output_sizes(1, C, H, W, out_channels, kernel_size, stride, padding); i++) {
    ss_result >> result_value;
    output[i] = result_value;
  }

}

#if WITH_TORCH
TEST(conv2d, basic_conv2d) {
  auto batch = 1;
  auto channel = 3;
  auto input_height = 32;
  auto input_width = 32;
  auto stride = 1;
  auto padding = 0;
  auto filter_channel = 3;
  auto filter_size = 3;

  auto input = std::make_unique<float[]>(batch * channel * input_height * input_width);
  auto weight = std::make_unique<float_16[]>(filter_channel * channel * filter_size * filter_size);
  auto bias = std::make_unique<float[]>(filter_channel);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  // Fill them with random values
  for (int i = 0; i < batch * channel * input_height * input_width; i++) {
    input[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel * channel * filter_size * filter_size; i++) {
    weight[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel; i++) {
    bias[i] = matrix_dist(generator);
  }

  auto output_size =
      conv2d_output_sizes(batch, channel, input_height, input_width, filter_channel, filter_size, stride, padding);

  auto output_ref = std::make_unique<float[]>(output_size);
  auto output = std::make_unique<float[]>(output_size);

  conv2d_torch(input.get(),
               output_ref.get(),
               weight.get(),
               bias.get(),
               batch,
               channel,
               input_height,
               input_width,
               filter_channel,
               filter_size,
               stride,
               padding);

  conv2d(input.get(),
         output.get(),
         weight.get(),
         bias.get(),
         batch,
         channel,
         input_height,
         input_width,
         filter_channel,
         filter_size,
         stride,
         padding,
         Impl::DeviceType::CPU);

  // Compare the results
  double max_diff_ratio = 0;
  double avg_diff_ratio = 0;
  for (int i = 0; i < output_size; i++) {
    double diff_ratio = output_ref[i] != 0 ? abs(output_ref[i] - output[i]) / abs(output_ref[i]) : 0;
    max_diff_ratio = max(max_diff_ratio, diff_ratio);
    avg_diff_ratio += diff_ratio / output_size;
  }

  std::cout << "Max difference ratio due to precision loss: " << max_diff_ratio << std::endl;

  std::cout << "Avg difference ratio due to precision loss: " << avg_diff_ratio << std::endl;
  EXPECT_LT(avg_diff_ratio, 1e-2);
}

TEST(conv2d, basic_conv2d_conv1) {
  // 224x224x3 -> 112x112x64
  // 7x7 conv with stride 2
  auto batch = 16;
  auto channel = 3;
  auto input_height = 224;
  auto input_width = 224;
  auto stride = 2;
  auto padding = 3;
  auto filter_channel = 64;
  auto filter_size = 7;

  auto output_layers = 64;
  auto output_height = 112;
  auto output_width = 112;

  // Check if shapes are correct
  auto shape =
      conv2d_result_shape(batch, channel, input_height, input_width, filter_channel, filter_size, stride, padding);

  auto input = std::make_unique<float[]>(batch * channel * input_height * input_width);
  auto weight = std::make_unique<float_16[]>(filter_channel * channel * filter_size * filter_size);
  auto bias = std::make_unique<float[]>(filter_channel);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  // Fill them with random values
  for (int i = 0; i < batch * channel * input_height * input_width; i++) {
    input[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel * channel * filter_size * filter_size; i++) {
    weight[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel; i++) {
    bias[i] = matrix_dist(generator);
  }

  auto output_size =
      conv2d_output_sizes(batch, channel, input_height, input_width, filter_channel, filter_size, stride, padding);

  auto output_ref = std::make_unique<float[]>(output_size);
  auto output = std::make_unique<float[]>(output_size);

  conv2d_torch(input.get(),
               output_ref.get(),
               weight.get(),
               bias.get(),
               batch,
               channel,
               input_height,
               input_width,
               filter_channel,
               filter_size,
               stride,
               padding);

  conv2d(input.get(),
         output.get(),
         weight.get(),
         bias.get(),
         batch,
         channel,
         input_height,
         input_width,
         filter_channel,
         filter_size,
         stride,
         padding,
         Impl::DeviceType::CPU);

  // Compare the results
  double max_diff_ratio = 0;
  double avg_diff_ratio = 0;
  for (int i = 0; i < output_size; i++) {
    double diff_ratio = output_ref[i] != 0 ? abs(output_ref[i] - output[i]) / abs(output_ref[i]) : 0;
    max_diff_ratio = max(max_diff_ratio, diff_ratio);
    avg_diff_ratio += diff_ratio / output_size;
  }

  std::cout << "Max difference ratio due to precision loss: " << max_diff_ratio << std::endl;

  std::cout << "Avg difference ratio due to precision loss: " << avg_diff_ratio << std::endl;
  EXPECT_LT(avg_diff_ratio, 1e-1);
}

TEST(conv2d, basic_conv2d_conv2x) {
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
  auto output_height = 56;
  auto output_width = 56;

  // Check if shapes are correct
  auto shape =
      conv2d_result_shape(batch, channel, input_height, input_width, filter_channel, filter_size, stride, padding);
  ASSERT_EQ(shape[0], batch);
  ASSERT_EQ(shape[1], output_layers);
  ASSERT_EQ(shape[2], output_height);
  ASSERT_EQ(shape[3], output_width);

  auto input = std::make_unique<float[]>(batch * channel * input_height * input_width);
  auto weight = std::make_unique<float_16[]>(filter_channel * channel * filter_size * filter_size);
  auto bias = std::make_unique<float[]>(filter_channel);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  // Fill them with random values
  for (int i = 0; i < batch * channel * input_height * input_width; i++) {
    input[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel * channel * filter_size * filter_size; i++) {
    weight[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel; i++) {
    bias[i] = matrix_dist(generator);
  }

  auto output_size =
      conv2d_output_sizes(batch, channel, input_height, input_width, filter_channel, filter_size, stride, padding);

  auto output_ref = std::make_unique<float[]>(output_size);
  auto output = std::make_unique<float[]>(output_size);

  conv2d_torch(input.get(),
               output_ref.get(),
               weight.get(),
               bias.get(),
               batch,
               channel,
               input_height,
               input_width,
               filter_channel,
               filter_size,
               stride,
               padding);

  conv2d(input.get(),
         output.get(),
         weight.get(),
         bias.get(),
         batch,
         channel,
         input_height,
         input_width,
         filter_channel,
         filter_size,
         stride,
         padding,
         Impl::DeviceType::CPU);

  // Compare the results
  double max_diff_ratio = 0;
  double avg_diff_ratio = 0;
  for (int i = 0; i < output_size; i++) {
    double diff_ratio = output_ref[i] != 0 ? abs(output_ref[i] - output[i]) / abs(output_ref[i]) : 0;
    max_diff_ratio = max(max_diff_ratio, diff_ratio);
    avg_diff_ratio += diff_ratio / output_size;
  }

  std::cout << "Max difference ratio due to precision loss: " << max_diff_ratio << std::endl;

  std::cout << "Avg difference ratio due to precision loss: " << avg_diff_ratio << std::endl;
  EXPECT_LT(avg_diff_ratio, 1e-2);
}

TEST(conv2d, basic_conv2d_conv2x_cuda) {
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
  auto output_height = 56;
  auto output_width = 56;

  // Check if shapes are correct
  auto shape =
      conv2d_result_shape(batch, channel, input_height, input_width, filter_channel, filter_size, stride, padding);
  ASSERT_EQ(shape[0], batch);
  ASSERT_EQ(shape[1], output_layers);
  ASSERT_EQ(shape[2], output_height);
  ASSERT_EQ(shape[3], output_width);

  auto input = std::make_unique<float[]>(batch * channel * input_height * input_width);
  auto weight = std::make_unique<float_16[]>(filter_channel * channel * filter_size * filter_size);
  auto bias = std::make_unique<float[]>(filter_channel);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  // Fill them with random values
  for (int i = 0; i < batch * channel * input_height * input_width; i++) {
    input[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel * channel * filter_size * filter_size; i++) {
    weight[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_channel; i++) {
    bias[i] = matrix_dist(generator);
  }

  auto output_size =
      conv2d_output_sizes(batch, channel, input_height, input_width, filter_channel, filter_size, stride, padding);

  auto output_ref = std::make_unique<float[]>(output_size);
  auto output = std::make_unique<float[]>(output_size);

  conv2d_torch(input.get(),
               output_ref.get(),
               weight.get(),
               bias.get(),
               batch,
               channel,
               input_height,
               input_width,
               filter_channel,
               filter_size,
               stride,
               padding);

  float_16 *weight_d;
  float *bias_d;
  float *input_d;
  float *output_d;

  cudaMalloc(&weight_d, filter_channel * channel * filter_size * filter_size * sizeof(float_16));
  cudaMalloc(&bias_d, filter_channel * sizeof(float));
  cudaMalloc(&input_d, batch * channel * input_height * input_width * sizeof(float));
  cudaMalloc(&output_d, output_size * sizeof(float));

  cudaMemcpy(weight_d,
             weight.get(),
             filter_channel * channel * filter_size * filter_size * sizeof(float_16),
             cudaMemcpyHostToDevice);
  cudaMemcpy(bias_d, bias.get(), filter_channel * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(input_d,
             input.get(),
             batch * channel * input_height * input_width * sizeof(float),
             cudaMemcpyHostToDevice);

  conv2d(input_d,
         output_d,
         weight_d,
         bias_d,
         batch,
         channel,
         input_height,
         input_width,
         filter_channel,
         filter_size,
         stride,
         padding,
         Impl::DeviceType::CUDA);

  cudaMemcpy(output.get(), output_d, output_size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(output_d);
  cudaFree(input_d);
  cudaFree(bias_d);
  cudaFree(weight_d);

  // Compare the results
  double max_diff_ratio = 0;
  double avg_diff_ratio = 0;
  for (int i = 0; i < output_size; i++) {
    double diff_ratio = output_ref[i] != 0 ? abs(output_ref[i] - output[i]) / abs(output_ref[i]) : 0;
    max_diff_ratio = max(max_diff_ratio, diff_ratio);
    avg_diff_ratio += diff_ratio / output_size;
  }

  std::cout << "Max difference ratio due to precision loss: " << max_diff_ratio << std::endl;

  std::cout << "Avg difference ratio due to precision loss: " << avg_diff_ratio << std::endl;
  EXPECT_LT(avg_diff_ratio, 1e-2);
}

#endif
