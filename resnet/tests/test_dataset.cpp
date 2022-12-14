#include "gtest/gtest.h"
#include "dataset.hpp"

TEST(dataset, load)
{
    ImageDataset dataset(Impl::DeviceType::CUDA);
    dataset.load("./dataset_tensor", 1);
    EXPECT_EQ(dataset.size(), 1);

    Tensor x = dataset[0];
    int batch_size = x.sizes()[0];
    EXPECT_EQ(x.sizes()[1], 3);
    EXPECT_EQ(x.sizes()[2], 224);
    EXPECT_EQ(x.sizes()[3], 224);

    EXPECT_GE(x.index({batch_size - 1, 0, 0, 0}), 0);
    EXPECT_LE(x.index({batch_size - 1, 0, 0, 0}), 1);

    dataset.logPredicted(Tensor({batch_size, 1000}));
    dataset.printStats();
}
