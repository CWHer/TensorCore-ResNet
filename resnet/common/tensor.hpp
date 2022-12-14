#pragma once

#include "common.h"

class Tensor {
  // NOTE: This is the logical layer of Tensor
  friend class TensorOps;

public:
// NOTE: HACK: TensorStorage was created to make it
//  safe and behave correctly when copying Tensor

// Expose it to public
  class TensorStorage {
    // NOTE: This is the data layer of Tensor
    friend class Tensor;
    friend class TensorOps;

  private:
    int64_t total_size;
    int32_t n_dim;
    std::vector<int32_t> shape;
    std::vector<int32_t> strides;
    Impl::DeviceType device;
    float *data;

  public:
    TensorStorage();
    TensorStorage(const std::vector<int> &shape,
                  Impl::DeviceType device = Impl::DeviceType::CPU,
                  float *data = nullptr);
    ~TensorStorage();

    void load(const std::string &file_path);
    std::shared_ptr<TensorStorage> clone();
    float index(const std::vector<int> &indices);
    void view(const std::vector<int> &shape);
    void to(Impl::DeviceType device);
    friend std::ostream &operator<<(std::ostream &out, const std::shared_ptr<Tensor::TensorStorage> &x);
  };
private:
  std::shared_ptr<TensorStorage> storage;

public:
  Tensor();
  Tensor(std::shared_ptr<TensorStorage> storage);
  Tensor(const std::vector<int> &shape, Impl::DeviceType device = Impl::DeviceType::CPU, float *data = nullptr);
  void load(const std::string &file_path);
  // HACK: DO NOT support save
  // void save(const std::string &path) {}

  bool empty();
  Tensor clone();
  float *data_ptr();
  int64_t totalSize();
  float index(const std::vector<int> &indices);
  std::vector<int> sizes() const;
  void view(const std::vector<int> &shape);
  void to(Impl::DeviceType device);
  Impl::DeviceType getDevice();

public:
  friend std::ostream &operator<<(std::ostream &out, const Tensor &x);
};
