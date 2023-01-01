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

  double current_diff = 0;
  double total_diff = 0;

  const int batches = 32;
  for (int i = 0; i < batches; i++) {
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

    current_diff = 0;
    float label_max = -1e10, label_min = 1e10;
    for (int j = 0; j < result.totalSize(); j++) {
      float label_data = label.data_ptr()[j];
      if (label_data > label_max)
        label_max = label_data;
      if (label_data < label_min)
        label_min = label_data;
    }
    float label_range = label_max - label_min;

    for (int j = 0; j < result.totalSize(); j++) {
      double diff = std::fabs((result.data_ptr()[j] - label.data_ptr()[j]) / label_range);
      current_diff += diff / result.totalSize();
    }
    total_diff += current_diff / batches;
  }
  ASSERT_LE(total_diff, 1e-1);
  cout << "Avg difference ratio due to precision loss: " << total_diff << endl;
}

#endif
#endif
#endif
