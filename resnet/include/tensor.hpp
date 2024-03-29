#pragma once

#include "common.hpp"

namespace Impl {

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
    int64_t total_size{};
    int32_t n_dim{};
    std::vector<int32_t> shape;
    std::vector<int32_t> strides;
    Impl::DeviceType device;
    float *data;

  public:
    TensorStorage();
    explicit TensorStorage(const std::vector<int> &shape,
                           Impl::DeviceType device = Impl::DeviceType::CPU,
                           float *data = nullptr);
    TensorStorage(const TensorStorage &other);
    TensorStorage(TensorStorage &&other) noexcept;
    ~TensorStorage();

    void load(const std::string &file_path);
    std::shared_ptr<TensorStorage> clone();
    float index(const std::vector<int> &indices);
    void reshape(const std::vector<int> &shape);
    void to(Impl::DeviceType device);
    friend std::ostream &operator<<(std::ostream &out, const std::shared_ptr<Tensor::TensorStorage> &x);

    TensorStorage &operator=(const TensorStorage &other);
  };
private:
  std::shared_ptr<TensorStorage> storage;

public:
  Tensor();
  explicit Tensor(std::shared_ptr<TensorStorage> storage);
  explicit Tensor(const std::vector<int> &shape,
                  Impl::DeviceType device = Impl::DeviceType::CPU,
                  float *data = nullptr);
  Tensor(const Tensor &other);
  Tensor(Tensor &&other) noexcept;
  void load(const std::string &file_path);
  // HACK: DO NOT support save
  // void save(const std::string &path) {}

  bool empty();
  Tensor clone();
  float *data_ptr();
  const float *data_ptr() const;
  int64_t totalSize() const;
  float index(const std::vector<int> &indices);
  std::vector<int> sizes() const;
  void reshape(const std::vector<int> &shape);
  void to(Impl::DeviceType device);
  Impl::DeviceType getDevice() const;

  Tensor &operator=(const Tensor &other);
  Tensor &operator=(Tensor &&other) noexcept;

public:
  friend std::ostream &operator<<(std::ostream &out, const Tensor &x);
};
}


