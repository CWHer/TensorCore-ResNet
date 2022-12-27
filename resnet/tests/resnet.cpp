/** @file resnet.cpp
*/
#include <gtest/gtest.h>
#include "resnet.hpp"
#include "dataset.hpp"

using namespace Impl;
using namespace std;

#if WITH_DATA
#ifdef DATASET_ROOT
#ifdef RESNET18_ROOT
TEST(resnet, inference_all_dataset) {
  auto dataset = Impl::ImageDataset(DATASET_ROOT, Impl::DeviceType::CUDA);
  string model_path = RESNET18_ROOT;
  model_path += "/resnet18";
  ResNet18 resnet18(model_path);
  resnet18.to(Impl::DeviceType::CUDA);

  for (int i = 0; i < 6; i++) {
    auto data = dataset.next();
    auto result = std::move(resnet18.forward(std::move(data.first.first)));

    result.to(Impl::DeviceType::CPU);
    auto label = std::move(data.first.second);
    label.to(Impl::DeviceType::CPU);

    auto shape = result.sizes();
    auto label_shape = label.sizes();

    if (shape.size() != label_shape.size())
      throw std::runtime_error("result and label size mismatch");

    for (int j = 0; j < shape.size(); j++)
      if (shape[j] != label_shape[j])
        throw std::runtime_error("result and label size mismatch");

    double total_diff = 0;
    for (int j = 0; j < result.totalSize(); j++) {
      double diff = std::fabs((result.data_ptr()[j] - label.data_ptr()[j]) / (label.data_ptr()[j] + 1e-6));
      total_diff += diff / result.totalSize();
    }

    ASSERT_LE(total_diff, 1e-1);
  }
}


#endif
#endif
#endif
