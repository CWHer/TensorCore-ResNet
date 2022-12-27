#include "resnet.hpp"
#include "dataset.hpp"

using namespace Impl;

int main(int argc, char *argv[]) {
  // 1. load resnet18 model
  // 2. load image dataset
  // 3. inference

  // TODO: argument parsing
  auto dataset = Impl::ImageDataset("imagenet_tensor", Impl::DeviceType::CUDA);
  ResNet18 resnet18("weight/resnet18");
  resnet18.to(Impl::DeviceType::CUDA);
  resnet18.printModule("resnet18");
  std::cout << std::endl;

  int num_correct = 0;
  int num_total = 0;
  for (int i = 0; i < dataset.size(); i++) {
    auto data = dataset.next();
    auto result = std::move(resnet18.forward(std::move(data.first.first)));

    result.to(Impl::DeviceType::CPU);
    auto label = std::move(data.first.second);
    label.to(Impl::DeviceType::CPU);

    Tensor original_label = TensorOps::argmax(label, 1);
    Tensor predicted_result = TensorOps::argmax(result, 1);
    num_correct += (int) TensorOps::sum_equal(original_label, predicted_result);
    num_total += predicted_result.sizes()[0];
  }

  std::cout << "Accuracy Compared to PyTorch ResNet18 Implementation: " << num_correct << "/" << num_total << std::endl;

  return 0;
}
