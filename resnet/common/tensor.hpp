#pragma once

#include "common.h"

class Tensor
{
private:
    int64_t total_size;
    int32_t n_dim;
    std::vector<int32_t> shape;
    std::vector<int32_t> strides;
    float *data;
    // TODO: data_type conversion

public:
    // TODO
    Tensor()
    {
        // TODO
    }

    Tensor(const std::vector<int> &shape)
    {
        this->shape = shape;
        n_dim = shape.size();
        strides = std::vector<int>(n_dim);
        strides.back() = 1;
        for (int i = n_dim - 1; i > 0; i--)
            strides[i - 1] = strides[i] * shape[i];
        total_size = strides[0] * shape[0];
        this->data = new float[total_size];
    }

    // HACK: FIXME:
    // ~Tensor() { delete[] data; }

    void load(const std::string &path)
    {
        // TODO
    }

    // HACK: DO NOT support save
    // void save(const std::string &path) {}

    Tensor clone()
    {
        // TODO
    }

    float *data_ptr() { return data; }

    int64_t totalSize() { return total_size; }

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
        checkCppErrorsMsg(strides[0] * shape[0] != total_size, "Invalid shape");
    }

    void to(Impl::DeviceType device)
    {
        // TODO
    }

public:
    friend std::ostream &operator<<(std::ostream &out, const Tensor &x)
    {
        std::cout << "Tensor(";
        for (int i = 0; i < x.n_dim; i++)
            std::cout << x.shape[i] << (i == x.n_dim - 1 ? "" : ", ");
        std::cout << ")" << std::endl;
    }
};
