#include <gtest/gtest.h>
#include "functional/im2col.h"

#ifdef TEST_DATA_ROOT
TEST(conv2d, im2col_cuda)
{
    std::string file_dir = TEST_DATA_ROOT;

    int input_n = 23, input_c = 7, input_h = 117, input_w = 191;
    int kernel_size = 3, stride = 2, padding = 1;

    Tensor result_tensor = makeIm2colResult(input_n, input_c, input_h, input_w,
                                            kernel_size, stride, padding);

    Tensor x;
    x.load(file_dir + "/test_im2col_x.bin");
    x.to(DeviceType::CUDA);

    im2col(x.data_ptr(),
           reinterpret_cast<f16 *>(result_tensor.data_ptr()),
           input_n, input_c, input_h, input_w,
           kernel_size, stride, padding);

    // Verify result.
    result_tensor.to(DeviceType::CPU);
    auto result_size = im2colResultSize(input_n, input_c, input_h, input_w,
                                        kernel_size, stride, padding);

    Tensor y;
    y.load(file_dir + "/test_im2col_y.bin");
    EXPECT_EQ(result_size, y.totalSize());

    f16 *naive_res = reinterpret_cast<f16 *>(result_tensor.data_ptr());
    f32 *std_res = y.data_ptr();
    for (int i = 0; i < y.totalSize(); i++)
        EXPECT_NEAR(naive_res[i], std_res[i], 1e-3);
}
#endif
