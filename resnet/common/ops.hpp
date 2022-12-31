#pragma once

#include "common.h"
#include "tensor.hpp"
#include "functional/ops.h"

class TensorOps
{
public:
    static void relu_(Tensor &x)
    {
        checkCppErrorsMsg(x.storage->device != DeviceType::CUDA,
                          "This operation is only supported on CUDA");

        hostRelu_(x.data_ptr(), x.totalSize());
    }

    static void add_(Tensor &x, Tensor &y)
    {
        checkCppErrorsMsg(x.sizes() != y.sizes(),
                          "Tensor sizes are not equal");
        checkCppErrorsMsg(x.storage->device != DeviceType::CUDA ||
                              y.storage->device != DeviceType::CUDA,
                          "This operation is only supported on CUDA");

        hostAdd_(x.data_ptr(), y.data_ptr(), x.totalSize());
    }

    static Tensor argmax(const Tensor &x, int dim)
    {
        checkCppErrorsMsg(x.storage->device != DeviceType::CPU,
                          "This operation is not supported on CUDA");

        auto shape = x.sizes();
        shape.erase(shape.begin() + dim);
        Tensor result(shape, DeviceType::CPU);

        // NOTE: outer_size | dim | inner_size
        int64_t dim_size = x.storage->shape[dim];
        int64_t inner_size = x.storage->strides[dim];
        int64_t outer_size = x.storage->total_size / inner_size / dim_size;
        int64_t outer_stride = inner_size * dim_size;
        for (int64_t i = 0; i < outer_size; i++)
        {
            for (int64_t j = 0; j < inner_size; j++)
            {
                int64_t max_idx = 0;
                float max_value = *(x.storage->data + i * outer_stride + j);
                for (int64_t k = 1; k < dim_size; k++)
                {
                    float value = *(x.storage->data + i * outer_stride + k * inner_size + j);
                    if (value > max_value)
                        max_value = value, max_idx = k;
                }
                *(result.storage->data + i * inner_size + j) = static_cast<float>(max_idx);
            }
        }
        return result;
    }

    static float sum_equal(const Tensor &lhs, const Tensor &rhs)
    {
        checkCppErrorsMsg(lhs.sizes() != rhs.sizes(),
                          "Tensor sizes are not equal");
        checkCppErrorsMsg(lhs.storage->device != DeviceType::CPU ||
                              rhs.storage->device != DeviceType::CPU,
                          "This operation is not supported on CUDA");

        static const float eps = 1e-3;
        float num_elements = 0;
        for (int i = 0; i < lhs.storage->total_size; i++)
            num_elements += std::fabs(*(lhs.storage->data + i) -
                                      *(rhs.storage->data + i)) < eps;
        return num_elements;
    }
};