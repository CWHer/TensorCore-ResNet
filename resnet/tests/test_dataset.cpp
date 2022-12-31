#include "gtest/gtest.h"
#include "dataset.hpp"

struct ValData
{
    std::vector<int> sizes;
    std::vector<float> flatten_data;
};

ValData loadValData(const std::string &path,
                    const std::string &category,
                    const int id,
                    int read_amount = -1)
{
    std::string index = std::to_string(id);
    index = std::string(4 - index.size(), '0') + index;
    std::string file_path = path + "/validation_" + index + "_" + category + ".txt";

    ValData data;
    std::ifstream fin(file_path);

    int n_dim;
    fin >> n_dim;
    data.sizes = std::vector<int>(n_dim);
    int total_size = 1;

    for (int i = 0; i < n_dim; i++)
    {
        fin >> data.sizes[i];
        total_size *= data.sizes[i];
    }

    if (read_amount == -1 || read_amount > total_size)
    {
        read_amount = 1;
        for (int i = 0; i < n_dim; i++)
            read_amount *= data.sizes[i];
    }
    data.flatten_data = std::vector<float>(read_amount);
    for (int i = 0; i < read_amount; i++)
        fin >> data.flatten_data[i];

    return data;
}

#ifdef DATASET_ROOT
#ifdef TEST_DATA_ROOT
TEST(dataset, load)
{
    constexpr int MAX_INSPECT_DATA_LENGTH = 64;
    // only stored 3 validation data
    constexpr int MAX_INSPECT_DATASET_SIZE = 3;

    ImageDataset dataset(DeviceType::CUDA);
    dataset.load(DATASET_ROOT, MAX_INSPECT_DATASET_SIZE);

    for (int i = 0; i < MAX_INSPECT_DATASET_SIZE; i++)
    {
        auto data = dataset.next();
        auto x = data.first, y = data.second;
        EXPECT_EQ(x.getDevice(), DeviceType::CUDA);
        EXPECT_EQ(y.getDevice(), DeviceType::CUDA);
        x.to(DeviceType::CPU);
        y.to(DeviceType::CPU);

        auto val_image = loadValData(TEST_DATA_ROOT, "image", i, MAX_INSPECT_DATA_LENGTH);

        ASSERT_EQ(x.sizes().size(), val_image.sizes.size());
        for (int i = 0; i < x.sizes().size(); i++)
            ASSERT_EQ(x.sizes()[i], val_image.sizes[i]);
        for (int i = 0; i < MAX_INSPECT_DATA_LENGTH; i++)
            EXPECT_EQ(x.data_ptr()[i], val_image.flatten_data[i]);

        auto val_label = loadValData(TEST_DATA_ROOT, "label", i, MAX_INSPECT_DATA_LENGTH);

        ASSERT_EQ(y.sizes().size(), val_label.sizes.size());
        for (int i = 0; i < y.sizes().size(); i++)
            ASSERT_EQ(y.sizes()[i], val_label.sizes[i]);
        for (int i = 0; i < MAX_INSPECT_DATA_LENGTH; i++)
            EXPECT_EQ(y.data_ptr()[i], val_label.flatten_data[i]);
    }
}
#endif
#endif
