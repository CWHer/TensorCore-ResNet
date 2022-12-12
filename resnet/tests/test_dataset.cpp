#include "gtest/gtest.h"
#include "dataset.hpp"

TEST(dataset, load)
{
    ImageDataset dataset(DeviceType::CUDA);
    dataset.load("./dataset");
    EXPECT_EQ(dataset.size().first, 1000);
    dataset.partition(128);
    EXPECT_EQ(dataset.size().second, 8);

    Tensor x = dataset[0];
    EXPECT_EQ(x.sizes()[0], 128);
    EXPECT_EQ(x.sizes()[1], 3);
    EXPECT_EQ(x.sizes()[2], 224);
    EXPECT_EQ(x.sizes()[3], 224);
}
