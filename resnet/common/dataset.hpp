#pragma once

#include "common.h"
#include "tensor.hpp"
#include "ops.hpp"

namespace Impl {

class ImageDataset {
  std::vector<Tensor> batched_tensors;
  std::vector<Tensor> batched_labels, predicted_logits;
  Impl::DeviceType device;

public:
  ImageDataset(Impl::DeviceType device = Impl::DeviceType::CUDA) : device(device) {}

  void load(const std::string &path, int num_tensors);

    Tensor operator[](int index);

    void logPredicted(Tensor logits);

    int size() const;

    void printStats();
};

}
