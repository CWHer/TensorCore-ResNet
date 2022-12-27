/** @file dataset.cpp
*/
#include <utility>
#include <dirent.h>

#include "dataset.hpp"

using namespace Impl;
using namespace std;

ImageDataset::ImageDataset(std::string data_path, Impl::DeviceType device, unsigned int size)
    : device(device), cur_idx(0), data_path(std::move(data_path)) {
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

}

pair<Tensor, Tensor> ImageDataset::load(const std::string &path, int index) {
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

pair<pair<Tensor, Tensor>, bool> Impl::ImageDataset::next() {
  if (cur_idx >= total_size)
    return std::make_pair(std::make_pair(Tensor(), Tensor()), false);
  auto data = load(data_path, cur_idx);
  cur_idx++;
  return std::make_pair(std::move(data), true);
}

unsigned int ImageDataset::size() const { return total_size; }


