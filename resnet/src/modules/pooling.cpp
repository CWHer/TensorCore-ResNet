/** @file pooling.cpp
*/
#include "pooling.hpp"
#include "functional/pool.hpp"

Impl::MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size(kernel_size), stride(stride), padding(padding) {
}

Impl::Tensor Impl::MaxPool2d::forward(const Impl::Tensor &x) {
#if DEBUG
  checkCppErrorsMsg(x.sizes().size() != 4, "MaxPool2d only support 4D tensor");
  // FIXME: (or not) this is a shape (2N, 64, H, W) oriented implementation,
  //  we DO NOT guarantee the correctness of inputs with other shapes
  checkCppErrorsMsg(x.sizes()[0] % 2 != 0 || x.sizes()[1] != 64,
                    "MaxPool2d is shape oriented, only support 2N x 64 x H x W");
#endif

  // NOTE: x is in NCHW format
  const auto batch_size = x.sizes()[0];
  const auto num_channels = x.sizes()[1];
  const auto height = x.sizes()[2];
  const auto width = x.sizes()[3];

  const auto output_height = (height + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
  const auto output_width = (width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;


  // FIXME: can we find a way to pre-allocate and reuse the memory,
  //  instead of allocating them every time (though this should be the way)
  Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);
  if (likely(x.getDevice() == DeviceType::CUDA)) {
    timer.start("forward");
    maxpool2d(x.data_ptr(),
              batch_size,
              num_channels,
              height,
              width,
              output.data_ptr(),
              output_height,
              output_width,
              kernel_size,
              padding,
              stride);
    timer.end("forward");
  } else {
    // If not cuda, move to cuda
    Tensor x_cuda = x;
    x_cuda.to(DeviceType::CUDA);
    timer.start("forward");
    maxpool2d(x_cuda.data_ptr(),
              batch_size,
              num_channels,
              height,
              width,
              output.data_ptr(),
              output_height,
              output_width,
              kernel_size,
              padding,
              stride);
    timer.end("forward");
    output.to(DeviceType::CPU);
  }

  return output;
}
Impl::Tensor Impl::MaxPool2d::forward(Impl::Tensor &&x) {
  return forward(x);
}

void Impl::MaxPool2d::printModule(const std::string &prefix) {
  std::cout << prefix << ":MaxPool2d" << std::endl;
}
void Impl::MaxPool2d::printStat(const std::string &prefix) {
  printModule(prefix);
  timer.printStat("forward");
}

Impl::AvgPool2d::AvgPool2d(int kernel_size, int stride, int padding)
    : kernel_size(kernel_size), stride(stride), padding(padding) {
}

Impl::Tensor Impl::AvgPool2d::forward(const Impl::Tensor &x) {
#if DEBUG
  checkCppErrorsMsg(x.sizes().size() != 4, "AvgPool2d only support 4D tensor");
  // FIXME: (or not) this is a shape (N, 128K, H, W) oriented implementation,
  //  we DO NOT guarantee the correctness of inputs with other shapes
  checkCppErrorsMsg(x.sizes()[1] % 128 != 0, "AvgPool2d is shape oriented, only support N x 128K x H x W");
#endif

  // NOTE: x is in NCHW format
  const auto batch_size = x.sizes()[0];
  const auto num_channels = x.sizes()[1];
  const auto height = x.sizes()[2];
  const auto width = x.sizes()[3];

  const auto output_height = (height + 2 * padding - (kernel_size - 1) - 1) / stride + 1;
  const auto output_width = (width + 2 * padding - (kernel_size - 1) - 1) / stride + 1;


  // FIXME: can we find a way to pre-allocate and reuse the memory,
  //  instead of allocating them every time (though this should be the way)
  Tensor output({batch_size, num_channels, output_height, output_width}, DeviceType::CUDA);

  if (likely(x.getDevice() == Impl::DeviceType::CUDA)) {
    timer.start("forward");
    avgpool2d(x.data_ptr(),
              batch_size,
              num_channels,
              height,
              width,
              output.data_ptr(),
              output_height,
              output_width,
              kernel_size,
              padding,
              stride);
    timer.end("forward");
  } else {
    // If not cuda, move to cuda
    Tensor x_cuda = x;
    x_cuda.to(DeviceType::CUDA);
    timer.start("forward");
    avgpool2d(x_cuda.data_ptr(),
              batch_size,
              num_channels,
              height,
              width,
              output.data_ptr(),
              output_height,
              output_width,
              kernel_size,
              padding,
              stride);
    timer.end("forward");
    output.to(DeviceType::CPU);
  }

  return output;
}
Impl::Tensor Impl::AvgPool2d::forward(Impl::Tensor &&x) {
  return forward(x);
}
void Impl::AvgPool2d::printModule(const std::string &prefix) {
  std::cout << prefix << ":AvgPool2d" << std::endl;
}

void Impl::AvgPool2d::printStat(const std::string &prefix) {
  printModule(prefix);
  timer.printStat("forward");
}

