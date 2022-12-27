/** @file dataset.cpp
*/
#include "dataset.hpp"

void Impl::ImageDataset::load(const std::string &path, int num_tensors) {
  // NOTE: HACK: use preprocess.py to generate the binary files
  // Load batched tensors
  data_path = path;
  for (int i = 0; i < num_tensors; i++) {
    batched_tensors.emplace_back();
    batched_labels.emplace_back();
  }
}
void Impl::ImageDataset::next() {
  int num_tensors = batched_tensors.size();
  if (cur_idx + 1 < num_tensors) {
    std::string index = std::to_string(cur_idx);
    index = std::string(4 - index.size(), '0') + index;
    batched_tensors.back().load(data_path + "/images_" + index + ".bin");
    batched_labels.back().load(data_path + "/labels_" + index + ".bin");
    cur_idx++;
  } else {
    std::cerr << "dataset loader error: index out of bounds" << std::endl;
  }
}
Impl::Tensor Impl::ImageDataset::operator[](int index) {
  return batched_tensors[index];
}
void Impl::ImageDataset::logPredicted(Impl::Tensor logits) {
  predicted_logits.push_back(logits);
}
int Impl::ImageDataset::size() const { return batched_tensors.size(); }
void Impl::ImageDataset::printStats() {
  int num_correct = 0;
  int total_num = 0;
  for (int i = 0; i < this->size(); i++)
  {
    Tensor predicted_label = TensorOps::argmax(predicted_logits[i], 1);
    num_correct += TensorOps::sum_equal(
        batched_labels[i], predicted_label);
    total_num += batched_labels[i].sizes()[0];
  }
  std::cout << "Accuracy: " << num_correct * 100.0 / total_num << "%" << std::endl;
}
