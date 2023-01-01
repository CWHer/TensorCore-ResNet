/** @file dataset.cpp
*/
#include <utility>
#include <dirent.h>

#include "dataset.hpp"

using namespace Impl;
using namespace std;

constexpr unsigned int MAX_BATCH_SIZE = 256;

ImageDataset::ImageDataset(std::string data_path, Impl::DeviceType device, int size)
    : device(device), data_path(std::move(data_path)), cur_idx(0), cur_loaded_idx(0) {
  if (size == -1) {
    total_size = 0;
    // Traverse the directory to get the number of files
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(this->data_path.c_str())) != nullptr) {
      while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_type != DT_REG) {
          continue;
        }
        string file_name = ent->d_name;
        // If the filename starts with images, it is an image
        if (file_name.find("images") == 0) {
          total_size++;
        }
      }
      closedir(dir);
    } else {
      throw std::runtime_error("Cannot open directory " + this->data_path);
    }
  } else {
    total_size = size;
  }
  load_next_batch();
}

pair<Tensor, Tensor> ImageDataset::load_single(const std::string &path, unsigned int index) {
  // NOTE: HACK: use preprocess.py to generate the binary files
  // Load batched tensors
  string index_str = std::to_string(index);
  index_str = std::string(4 - index_str.size(), '0') + index_str;
  Impl::Tensor image, label;
  image.load(path + "/images_" + index_str + ".bin");
  label.load(path + "/labels_" + index_str + ".bin");
  image.to(device);
  label.to(device);

  return std::move(std::make_pair(std::move(image), std::move(label)));
}

void ImageDataset::load_next_batch() {
  if (cur_loaded_idx >= total_size) {
    return;
  }
  unsigned int batch_size = std::min(MAX_BATCH_SIZE, total_size - cur_loaded_idx);
  for (unsigned int i = 0; i < batch_size; i++) {
    auto il = load_single(data_path, cur_loaded_idx + i);
    auto image = std::move(il.first);
    auto label = std::move(il.second);
    batched_tensors.emplace_back(std::move(image));
    batched_labels.emplace_back(std::move(label));
  }
  cur_loaded_idx += batch_size;
}

pair<pair<Tensor, Tensor>, bool> Impl::ImageDataset::next() {
  if (cur_idx >= cur_loaded_idx) {
    load_next_batch();
  }
  if (cur_idx >= total_size)
    return std::make_pair(std::make_pair(Tensor(), Tensor()), false);
  auto image = std::move(batched_tensors.front());
  auto label = std::move(batched_labels.front());
  batched_tensors.pop_front();
  batched_labels.pop_front();
  cur_idx++;
  return std::make_pair(std::make_pair(std::move(image), std::move(label)), true);
}

unsigned int ImageDataset::size() const { return total_size; }



