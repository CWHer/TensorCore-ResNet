/** @file dataset.cpp
*/
#include "dataset.hpp"

void Impl::ImageDataset::load(const std::string &path, int num_tensors) {
  // NOTE: HACK: use preprocess.py to generate the binary files
  // Load batched tensors
  for (int i = 0; i < num_tensors; i++) {
    batched_tensors.emplace_back();
    batched_labels.emplace_back();
    std::string index = std::to_string(i);
    index = std::string(4 - index.size(), '0') + index;
    batched_tensors.back().load(path + "/images_" + index + ".bin");
    batched_labels.back().load(path + "/labels_" + index + ".bin");
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
