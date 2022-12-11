/** @file conv2d.cu
*/

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <c++/12.2.0/fstream>
#include <sstream>
#include <a.out.h>
#include <random>
#include "functional/conv2d.hpp"
#include "common.h"

#ifndef RANDOM_SEED
#define RANDOM_SEED std::random_device{}()
#endif

using namespace std;

static string python_command_invoke(const string &script) {
  // Run the Python script, and get the result stored in a string.
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
    dup2(in_fd[0], STDIN_FILENO);
    dup2(out_fd[1], STDOUT_FILENO);
    // close unused pipe ends.
    close(in_fd[1]);
    close(out_fd[0]);

    // swap to python
    execlp("python", "python", NULL);
    // Error if failed
    perror("execlp");
    exit(EXIT_FAILURE);
  }

  // close unused end in parent end
  close(in_fd[0]);
  close(out_fd[1]);

  // write the script to the child process
  write(in_fd[1], script.c_str(), script.size());
  close(in_fd[1]);

  // read the result from the child process
  stringstream ss;
  char buf[1024];
  int n;
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
  close(out_fd[0]);

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

static int conv2d_output_size(int C, int H, int W, int out_channels, int kernel_size, int stride, int padding) {
  int out_height = (H + 2 * padding - kernel_size) / stride + 1;
  int out_width = (W + 2 * padding - kernel_size) / stride + 1;
  return out_channels * out_height * out_width;
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
                         int C,
                         int H,
                         int W,
                         int out_channels,
                         int kernel_size,
                         int stride,
                         int padding) {
  auto input_tensor = torch::from_blob(input, {1, C, H, W}).to(torch::kFloat32);
  torch::nn::Conv2dOptions options(C, out_channels, kernel_size);
  options = options.stride(stride);
  options = options.padding(padding);
  auto conv_ref_l = torch::nn::Conv2d(options);
  conv_ref_l->weight = torch::zeros({out_channels, C, kernel_size, kernel_size});

  // as torch cannot handle the float_16 blob correctly, we need to fill it manually
  for (int i = 0; i < out_channels; i++) {
    for (int j = 0; j < C; j++) {
      for (int k = 0; k < kernel_size; k++) {
        for (int l = 0; l < kernel_size; l++) {
          conv_ref_l->weight[i][j][k][l] =
              __half2float(weight[i * C * kernel_size * kernel_size + j * kernel_size * kernel_size + k * kernel_size
                  + l]);
        }
      }
    }
  }

  conv_ref_l->weight.to(torch::kFloat16);

  conv_ref_l->bias = torch::from_blob(bias, {out_channels});
  auto output_tensor = conv_ref_l->forward(input_tensor);
  memcpy(output, output_tensor.data_ptr(), output_tensor.numel() * sizeof(float));
}
#endif

static void conv2d_torch_pythonscript(float *input,
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

  for (int i = 0; i < conv2d_output_size(C, H, W, out_channels, kernel_size, stride, padding); i++) {
    ss_result >> result_value;
    output[i] = result_value;
  }

}

#if WITH_TORCH
TEST(conv2d, libtorch_correctness_conv1) {
  auto channel = 3;
  auto input_height = 224;
  auto input_width = 224;
  auto stride = 2;
  auto padding = 3;
  auto filter_chanel = 64;
  auto filter_size = 3;

  auto input = std::make_unique<float[]>(channel * input_height * input_width);
  auto weight = std::make_unique<float_16[]>(filter_chanel * channel * filter_size * filter_size);
  auto bias = std::make_unique<float[]>(filter_chanel);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  // Fill them with random values
  for (int i = 0; i < channel * input_height * input_width; i++) {
    input[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_chanel * channel * filter_size * filter_size; i++) {
    weight[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_chanel; i++) {
    bias[i] = matrix_dist(generator);
  }

  auto size = conv2d_output_size(channel, input_height, input_width, filter_chanel, filter_size, stride, padding);

  auto output_ref = std::make_unique<float[]>(size);

  auto output = std::make_unique<float[]>(size);

  conv2d_torch(input.get(),
               output.get(),
               weight.get(),
               bias.get(),
               channel,
               input_height,
               input_width,
               filter_chanel,
               filter_size,
               stride,
               padding);

  conv2d_torch_pythonscript(input.get(),
                            output_ref.get(),
                            weight.get(),
                            bias.get(),
                            channel,
                            input_height,
                            input_width,
                            filter_chanel,
                            filter_size,
                            stride,
                            padding);

  // Compare the results
  for (int i = 0;
       i < conv2d_output_size(channel, input_height, input_width, filter_chanel, filter_size, stride, padding); i++) {
    ASSERT_NEAR(output_ref[i], output[i], 1e0);
  }

}

TEST(conv2d, libtorch_correctness_3_64) {
  auto channel = 3;
  auto input_height = 64;
  auto input_width = 64;
  auto stride = 1;
  auto padding = 0;
  auto filter_chanel = 3;
  auto filter_size = 3;

  auto input = std::make_unique<float[]>(channel * input_height * input_width);
  auto weight = std::make_unique<float_16[]>(filter_chanel * channel * filter_size * filter_size);
  auto bias = std::make_unique<float[]>(filter_chanel);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  // Fill them with random values
  for (int i = 0; i < channel * input_height * input_width; i++) {
    input[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_chanel * channel * filter_size * filter_size; i++) {
    weight[i] = matrix_dist(generator);
  }
  for (int i = 0; i < filter_chanel; i++) {
    bias[i] = matrix_dist(generator);
  }

  auto output_ref = std::make_unique<float[]>(conv2d_output_size(channel,
                                                                 input_height,
                                                                 input_width,
                                                                 filter_chanel,
                                                                 filter_size,
                                                                 stride,
                                                                 padding));

  auto output = std::make_unique<float[]>(conv2d_output_size(channel,
                                                             input_height,
                                                             input_width,
                                                             filter_chanel,
                                                             filter_size,
                                                             stride,
                                                             padding));

  conv2d_torch(input.get(),
               output.get(),
               weight.get(),
               bias.get(),
               channel,
               input_height,
               input_width,
               filter_chanel,
               filter_size,
               stride,
               padding);

  conv2d_torch_pythonscript(input.get(),
                            output_ref.get(),
                            weight.get(),
                            bias.get(),
                            channel,
                            input_height,
                            input_width,
                            filter_chanel,
                            filter_size,
                            stride,
                            padding);

  // Compare the results
  for (int i = 0;
       i < conv2d_output_size(channel, input_height, input_width, filter_chanel, filter_size, stride, padding); i++) {
    ASSERT_NEAR(output_ref[i], output[i], 1e0);
  }

}
#endif

#if WITH_TORCH
TEST(conv2d, basic_conv2d) {
  auto channel = 3;
  auto input_height = 32;
  auto input_width = 32;
  auto stride = 1;
  auto padding = 0;
  auto filter_channel = 3;
  auto filter_size = 3;

  auto input = std::make_unique<float[]>(channel * input_height * input_width);
  auto weight = std::make_unique<float_16[]>(filter_channel * channel * filter_size * filter_size);
  auto bias = std::make_unique<float[]>(filter_channel);

  // Randomly initialize
  default_random_engine generator(RANDOM_SEED);
  uniform_real_distribution<float> matrix_dist(-1.0e2, 1.0e2);

  // Fill them with random values
  for (int i = 0; i < channel * input_height * input_width; i++) {
    //input[i] = matrix_dist(generator);
    input[i] = 1;
  }
//  for (int i = 0; i < filter_channel * channel * filter_size * filter_size; i++) {
//    //weight[i] = matrix_dist(generator);
//    weight[i] = i;
//  }
  for (int i = 0; i < filter_channel; i += 1) {
    for (int j = 0; j < channel * filter_size * filter_size; j++) {
      weight[i * channel * filter_size * filter_size + j] = i == 0 ? float(i + 1) : 0.0f;
    }
  }

  for (int i = 0; i < filter_channel; i++) {
    //bias[i] = matrix_dist(generator);
    bias[i] = 0;
  }

  auto output_size =
      conv2d_output_size(channel, input_height, input_width, filter_channel, filter_size, stride, padding);

  auto output_ref = std::make_unique<float[]>(output_size);
  auto output = std::make_unique<float[]>(output_size);

  conv2d_torch(input.get(),
               output_ref.get(),
               weight.get(),
               bias.get(),
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
         channel,
         input_height,
         input_width,
         filter_channel,
         filter_size,
         stride,
         padding);

  // Compare the results
  double max_diff_ratio = 0;
  double avg_diff_ratio = 0;
  for (int i = 0; i < output_size; i++) {
    double diff_ratio = output_ref[i] != 0 ? abs(output_ref[i] - output[i]) / abs(output_ref[i]) : 0;
    if (diff_ratio > 10) {
      std::cout << "output_ref[" << i << "] = " << output_ref[i] << " while output[" << i << "] = " << output[i]
                << std::endl;
    }
    max_diff_ratio = max(max_diff_ratio, diff_ratio);
    avg_diff_ratio += diff_ratio / output_size;
  }

  std::cout << "Max difference ratio due to precision loss: " << max_diff_ratio << std::endl;

  std::cout << "Avg difference ratio due to precision loss: " << avg_diff_ratio << std::endl;
  EXPECT_LT(avg_diff_ratio, 1e-2);
  // Print a tensor
  auto output_height = (input_height - filter_size + 2 * padding) / stride + 1;
  auto output_width = (input_width - filter_size + 2 * padding) / stride + 1;
  at::Tensor tensor = torch::from_blob(output.get(), {filter_channel, output_height, output_width});
  std::cout << tensor << std::endl;
  tensor = torch::from_blob(output_ref.get(), {filter_channel, output_height, output_width});
  std::cout << tensor << std::endl;

}

#endif
