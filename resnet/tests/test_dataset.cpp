#include "gtest/gtest.h"
#include "dataset.hpp"

using namespace std;
using namespace Impl;

struct validation_data {
  std::vector<int> sizes;
  std::vector<float_32> flatten_data;
};

static struct validation_data validation_data_load(const std::string &path,
                                                   const string &category,
                                                   const int id,
                                                   int read_amount = -1) {
  string index = std::to_string(id);
  index = std::string(4 - index.size(), '0') + index;
  string file_path = path + "/validation_" + index + "_" + category + ".txt";
  struct validation_data data{};
  ifstream f(file_path);
  int n_dim;
  f >> n_dim;
  data.sizes = std::vector<int>(n_dim);
  int total_size = 1;

  for (int i = 0; i < n_dim; i++) {
    f >> data.sizes[i];
    total_size *= data.sizes[i];
  }

  if (read_amount == -1 || read_amount > total_size) {
    read_amount = 1;
    for (int i = 0; i < n_dim; i++)
      read_amount *= data.sizes[i];
  }
  data.flatten_data = std::vector<float_32>(read_amount);
  for (int i = 0; i < read_amount; i++)
    f >> data.flatten_data[i];

  return data;
}

#if WITH_DATA
#ifdef DATASET_ROOT
TEST(dataset, load) {
  constexpr int max_inspect_data_length = 64;
  // only stored 3 validation data
  constexpr int max_inspect_dataset_size = 3;

  ImageDataset dataset(Impl::DeviceType::CUDA);
  dataset.load(DATASET_ROOT, max_inspect_dataset_size);

  for (int dataset_idx = 0; dataset_idx < max_inspect_dataset_size; dataset_idx++) {
    auto validation_image = validation_data_load(DATASET_ROOT, "image", dataset_idx, max_inspect_data_length);
    Tensor x = dataset[dataset_idx];

    ASSERT_EQ(x.sizes().size(), validation_image.sizes.size());
    for (int i = 0; i < x.sizes().size(); i++)
      ASSERT_EQ(x.sizes()[i], validation_image.sizes[i]);

    for (int i = 0; i < max_inspect_data_length; i++)
      EXPECT_EQ(x.data_ptr()[i], validation_image.flatten_data[i]);
  }

}
#endif
#endif
