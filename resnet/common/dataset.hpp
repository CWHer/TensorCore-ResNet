#pragma once


#include "common.h"
#include "tensor.hpp"
#include "ops.hpp"

namespace Impl {

class ImageDataset {
private:
  std::vector<Tensor> batched_tensors;
  std::vector<Tensor> batched_labels, predicted_logits;
  Impl::DeviceType device;
  std::string data_path;
  unsigned int cur_idx;
  unsigned int total_size;

public:
  explicit ImageDataset(std::string data_path,
                        Impl::DeviceType device = Impl::DeviceType::CUDA,
                        unsigned int size = -1);

  std::pair<Tensor, Tensor> load(const std::string &path, unsigned int index);
  std::pair<std::pair<Tensor, Tensor>, bool> next();
  unsigned int size() const;

};

}
