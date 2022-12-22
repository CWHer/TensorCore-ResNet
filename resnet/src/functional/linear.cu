/** @file linear.cu
*/

// Linear layer is performed by trivial GEMM

#include "functional/gemm.hpp"

/**
 * @brief In-place add
 * @param Dest Destination
 * @param adder Adder
 * @param length Length of both Dest and adder
 */
void add_(float *Dest, const float *adder, int length) {
  for (int i = 0; i < length; ++i) {
    Dest[i] += adder[i];
  }
}

/**
 * @brief Linear layer implementation.
 *
 * @param input Input tensor of shape (input_channel)
 * @param output Output tensor of shape (output_channel)
 * @param weight Weight tensor of shape (input_channel, output_channel)
 * @param bias Bias tensor of shape (output_channel)
 * @param input_channel
 * @param output_channel
 */
void linear(const float_16 *input,
            float *output,
            const float_16 *weight,
            const float *bias,
            int input_channel,
            int output_channel,
            Impl::DeviceType device_type) {
  switch (device_type) {
  case Impl::DeviceType::CPU : {
    gemm(input, weight, output, 1, input_channel, output_channel, GEMM::Major::row_major, Impl::DeviceType::CPU);
  }
    break;
  case Impl::DeviceType::CUDA : {
    gemm(input, weight, output, 1, input_channel, output_channel, GEMM::Major::row_major, Impl::DeviceType::CUDA);
  }
    break;
  }
  add_(output, bias, output_channel);
}
