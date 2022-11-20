#pragma once

#include "common.h"
#include "ops.hpp"

class Tensor
{
private:
    int64_t sizes;
    int32_t n_dim;
    std::vector<int32_t> shape;
    std::vector<int32_t> strides;
    float *data;
    // TODO: data_type conversion

public:
    // TODO
    Tensor() {}

    Tensor(float *data, const std::vector<int> &shape)
    {
        this->data = data;
        this->shape = shape;
        n_dim = shape.size();
        strides = std::vector<int>(n_dim);
        strides.back() = 1;
        for (int i = n_dim - 1; i > 0; i--)
            strides[i - 1] = strides[i] * shape[i];
        sizes = strides[0] * shape[0];
    }

    ~Tensor() {}

    void load(const std::string &path)
    {
        // TODO
    }

    void save(const std::string &path)
    {
        // TODO
    }

    Tensor clone()
    {
        // TODO
    }

    float *data_ptr() { return data; }

    float index(const std::vector<int> &indices)
    {
        int offset = 0;
        for (int i = 0; i < indices.size(); i++)
            offset += indices[i] * strides[i];
        return data[offset];
    }

    std::vector<int> sizes() { return shape; }

    void view(const std::vector<int> &shape)
    {
        // HACK: DO NOT support -1
        this->shape = shape;
        n_dim = shape.size();
        strides.resize(n_dim);
        strides.back() = 1;
        for (int i = n_dim - 1; i > 0; i--)
            strides[i - 1] = strides[i] * shape[i];
        checkCppErrorsMsg(strides[0] * shape[0] != sizes, "Invalid shape");
    }

    void to(DeviceType device)
    {
        // TODO
    }

public:
    Tensor operator+=(const Tensor &other)
    {
        return add(*this, other);
    }

    std::ostream &operator<<(std::ostream &out)
    {
        std::cout << "Tensor(";
        for (int i = 0; i < n_dim; i++)
            std::cout << shape[i] << (i == n_dim - 1 ? "" : ", ");
        std::cout << ")" << std::endl;
    }
};