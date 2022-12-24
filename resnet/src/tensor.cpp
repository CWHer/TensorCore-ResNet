/** @file tensor.cpp
 *
 * Moved from tensor.hpp to avoid long compilation.
 */

#include "tensor.hpp"
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace Impl;

Tensor::TensorStorage::TensorStorage() : data(nullptr), device(Impl::DeviceType::UNKNOWN) {}

Tensor::TensorStorage::TensorStorage(const std::vector<int> &shape, Impl::DeviceType device, float *data)
    : shape(shape), n_dim(shape.size()), device(device) {
  strides = std::vector<int>(n_dim);
  strides.back() = 1;
  for (int i = n_dim - 1; i > 0; i--)
    strides[i - 1] = strides[i] * shape[i];
  total_size = strides[0] * shape[0];

  if (data == nullptr) {
    switch (device) {
    case Impl::DeviceType::CPU:data = new float[total_size];
      break;
    case Impl::DeviceType::CUDA:checkCudaErrors(cudaMalloc(&data, total_size * sizeof(float)));
      break;
    default:checkCppErrorsMsg(true, "Unknown device type");
    }
  }

  this->data = data;
}

Tensor::TensorStorage::TensorStorage(const Tensor::TensorStorage &other)
    : shape(other.shape), n_dim(other.n_dim), strides(other.strides), total_size(other.total_size),
      device(other.device) {
  switch (device) {
  case Impl::DeviceType::CPU:data = new float[total_size];
    std::copy(other.data, other.data + total_size, data);
    break;
  case Impl::DeviceType::CUDA:checkCudaErrors(cudaMalloc(&data, total_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(data, other.data, total_size * sizeof(float), cudaMemcpyDeviceToDevice));
    break;
  default:checkCppErrorsMsg(true, "Unknown device type");
  }
}

Tensor::TensorStorage::TensorStorage(Tensor::TensorStorage &&other) noexcept
    : shape(std::move(other.shape)), n_dim(other.n_dim), strides(std::move(other.strides)),
      total_size(other.total_size), data(other.data), device(other.device) {
  other.data = nullptr;
}

Tensor::TensorStorage::~TensorStorage() {
  if (data != nullptr) {
    switch (device) {
    case Impl::DeviceType::CPU:delete[] data;
      break;
    case Impl::DeviceType::CUDA:checkCudaErrors(cudaFree(data));
      break;
    default:checkCppErrorsMsg(true, "Unknown device type");
    }
  }
}

void Tensor::TensorStorage::load(const std::string &file_path) {
  // NOTE: content,
  //   1KB header (int_32 shape)
  //   indefinite data (float_32)
  static const int DATA_OFFSET = 1024;

  struct stat file_stat;
  stat(file_path.c_str(), &file_stat);
  auto file_size = file_stat.st_size;

  int fd = open(file_path.c_str(), O_RDONLY, 0);
  if (fd == -1) {
    checkCppErrorsMsg(true, ("Cannot open file: " + file_path).c_str());
    return;
  }

  void *mmap_ptr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mmap_ptr == MAP_FAILED) {
    checkCppErrorsMsg(true, ("Cannot mmap file: " + file_path).c_str());
    return;
  }

  int32_t *header_ptr = (int32_t *) mmap_ptr;
  float *data_ptr = (float *) ((char *) mmap_ptr + DATA_OFFSET);

  auto maximum_shape_length = DATA_OFFSET / sizeof(int32_t);
  shape = std::vector<int32_t>(header_ptr, header_ptr + maximum_shape_length);
  for (int i = 0; i < maximum_shape_length; i++) {
    if (shape[i] == 0) {
      shape.resize(i);
      break;
    }
  }

  n_dim = shape.size();
  strides = std::vector<int>(n_dim);
  strides.back() = 1;
  for (int i = n_dim - 1; i > 0; i--)
    strides[i - 1] = strides[i] * shape[i];
  total_size = strides[0] * shape[0];
  data = new float[total_size];
  device = Impl::DeviceType::CPU;

  memcpy(data, data_ptr, total_size * sizeof(float));
  munmap(mmap_ptr, file_size);
  close(fd);
}

std::shared_ptr<Tensor::TensorStorage> Tensor::TensorStorage::clone() {
  auto cloned_tensor = std::make_shared<TensorStorage>(shape, device);
  switch (device) {
  case Impl::DeviceType::CPU:std::copy(data, data + total_size, cloned_tensor->data);
    break;
  case Impl::DeviceType::CUDA:checkCudaErrors(cudaMemcpy(cloned_tensor->data,
                                                         data,
                                                         total_size * sizeof(float),
                                                         cudaMemcpyDeviceToDevice));
    break;
  default:checkCppErrorsMsg(true, "Unknown device type");
  }
  return cloned_tensor;
}

float Tensor::TensorStorage::index(const std::vector<int> &indices) {
  int offset = 0;
  for (int i = 0; i < indices.size(); i++)
    offset += indices[i] * strides[i];

  switch (device) {
  case Impl::DeviceType::CPU:return data[offset];
  case Impl::DeviceType::CUDA: {
    // HACK: FIXME: this is extremely inefficient
    auto h_data = std::make_shared<float>();
    checkCudaErrors(cudaMemcpy(h_data.get(), data + offset, sizeof(float), cudaMemcpyDeviceToHost));
    return *h_data;
  }
  default:checkCppErrorsMsg(true, "Unknown device type");
    return 0.0f;
  }
}
[[deprecated("Use reshape")]]
void Tensor::TensorStorage::view(const std::vector<int> &shape) {
  // HACK: DO NOT support -1
  this->shape = shape;
  n_dim = shape.size();
  strides.resize(n_dim);
  strides.back() = 1;
  for (int i = n_dim - 1; i > 0; i--)
    strides[i - 1] = strides[i] * shape[i];
  checkCppErrorsMsg(strides[0] * shape[0] != total_size, "Invalid shape");
}

void Tensor::TensorStorage::to(Impl::DeviceType device) {
  if (this->device == device)
    return;

  float *data;
  this->device = device;
  switch (device) {
  case Impl::DeviceType::CPU:data = new float[total_size];
    checkCudaErrors(cudaMemcpy(data, this->data, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(this->data));
    break;

  case Impl::DeviceType::CUDA:checkCudaErrors(cudaMalloc(&data, total_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(data, this->data, total_size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] this->data;
    break;

  default:checkCppErrorsMsg(true, "Unknown device type");
  }
  this->data = data;
}

Tensor::TensorStorage &Tensor::TensorStorage::operator=(const Tensor::TensorStorage &other) {
  if (this == &other)
    return *this;
  this->~TensorStorage();
  new(this) Tensor::TensorStorage(other);
  return *this;
}
void Tensor::TensorStorage::reshape(const std::vector<int> &shape) {
  // HACK: DO NOT support -1
  this->shape = shape;
  n_dim = shape.size();
  strides.resize(n_dim);
  strides.back() = 1;
  for (int i = n_dim - 1; i > 0; i--)
    strides[i - 1] = strides[i] * shape[i];
  checkCppErrorsMsg(strides[0] * shape[0] != total_size, "Invalid shape");
}

namespace Impl {
std::ostream &operator<<(std::ostream &out, const std::shared_ptr<Tensor::TensorStorage> &x) {
  std::cout << "Tensor(";
  for (int i = 0; i < x->n_dim; i++)
    std::cout << x->shape[i] << ", ";
  static auto deviceType = [](Impl::DeviceType device) {
    switch (device) {
    case Impl::DeviceType::CPU:return "CPU";
    case Impl::DeviceType::CUDA:return "CUDA";
    default:return "Unknown";
    }
  };
  std::cout << deviceType(x->device) << ")" << std::endl;
  return out;
}
}
Tensor::Tensor() : storage(nullptr) {}
Tensor::Tensor(const std::vector<int> &shape, Impl::DeviceType device, float *data) {
  storage = std::make_shared<TensorStorage>(shape, device, data);
}

void Tensor::load(const std::string &file_path) {
  storage = std::make_shared<TensorStorage>();
  storage->load(file_path);
}

bool Tensor::empty() { return storage == nullptr; }
Tensor Tensor::clone() { return Tensor(storage->clone()); }
float *Tensor::data_ptr() { return storage->data; }
const float *Tensor::data_ptr() const {
  return storage->data;
}
int64_t Tensor::totalSize() const { return storage->total_size; }
float Tensor::index(const std::vector<int> &indices) {
  return storage->index(indices);
}
std::vector<int> Tensor::sizes() const { return storage->shape; }
[[deprecated("Use reshape")]]
void Tensor::view(const std::vector<int> &shape) {
  storage->view(shape);
}
void Tensor::to(Impl::DeviceType device) {
  storage->to(device);
}
Impl::DeviceType Tensor::getDevice() const { return storage->device; }
namespace Impl {
std::ostream &operator<<(std::ostream &out, const Tensor &x) {
  return out << x.storage;
}
}
Tensor::Tensor(std::shared_ptr<TensorStorage> storage) : storage(storage) {}
Tensor::Tensor(const Tensor &other) {
  Tensor::TensorStorage other_storage = *other.storage;
  storage = std::make_shared<TensorStorage>(std::move(other_storage));
}
Tensor &Tensor::operator=(const Tensor &other) {
  if (this == &other)
    return *this;
  this->~Tensor();
  new(this) Tensor(other);
  return *this;
}
void Tensor::reshape(const std::vector<int> &shape) {
  storage->reshape(shape);
}
