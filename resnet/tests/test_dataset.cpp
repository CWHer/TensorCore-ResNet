#include "gtest/gtest.h"
#include "dataset.hpp"

#if WITH_DATASET
#ifdef DATASET_ROOT
TEST(dataset, load)
{
    ImageDataset dataset(Impl::DeviceType::CUDA);
    dataset.load(DATASET_ROOT, 1);
    EXPECT_EQ(dataset.size(), 1);

    Tensor x = dataset[0];

    std::cout << x;
    int batch_size = x.sizes()[0];
    EXPECT_EQ(x.sizes()[1], 3);
    EXPECT_EQ(x.sizes()[2], 224);
    EXPECT_EQ(x.sizes()[3], 224);

    float t = x.index({batch_size - 1, 0, 0, 0});
    EXPECT_GE(t, 0);
    EXPECT_LE(t, 1);

    dataset.logPredicted(Tensor({batch_size, 1000}));
    dataset.printStats();
}
#endif
#endif
