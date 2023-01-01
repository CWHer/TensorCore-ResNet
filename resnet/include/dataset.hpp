#pragma once

#include "common.hpp"
#include "tensor.hpp"
#include "ops.hpp"

namespace Impl {

class ImageDataset {
private:
  std::deque<Tensor> batched_tensors;
  std::deque<Tensor> batched_labels;
  Impl::DeviceType device;
  std::string data_path;
  unsigned int cur_idx;
  unsigned int cur_loaded_idx;
  unsigned int total_size;

public:
  explicit ImageDataset(std::string data_path,
                        Impl::DeviceType device = Impl::DeviceType::CUDA,
                        int size = -1);

  std::pair<Tensor, Tensor> load_single(const std::string &path, unsigned int index);
  void load_next_batch();
  std::pair<std::pair<Tensor, Tensor>, bool> next();
  unsigned int size() const;

};

}
