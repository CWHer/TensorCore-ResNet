/** @file ops.cpp
*/
#include "ops.hpp"

Impl::Tensor Impl::TensorOps::argmax(const Impl::Tensor &x, int dim) {
#if DEBUG
  // Hack: for a embedded function, this will not happen
  checkCppErrorsMsg(x.storage->device != Impl::DeviceType::CPU, "This operation is not supported on CUDA");
#endif

  auto shape = x.sizes();
  shape.erase(shape.begin() + dim);
  Tensor result(shape, Impl::DeviceType::CPU);

  // NOTE: outer_size | dim | inner_size
  int64_t dim_size = x.storage->shape[dim];
  int64_t inner_size = x.storage->strides[dim];
  int64_t outer_size = x.storage->total_size / inner_size / dim_size;
  int64_t outer_stride = inner_size * dim_size;
  for (int64_t i = 0; i < outer_size; i++) {
    for (int64_t j = 0; j < inner_size; j++) {
      int64_t max_idx = 0;
      float max_value = *(x.storage->data + i * outer_stride + j);
      for (int64_t k = 1; k < dim_size; k++) {
        float value = *(x.storage->data + i * outer_stride + k * inner_size + j);
        if (value > max_value)
          max_value = value, max_idx = k;
      }
      *(result.storage->data + i * inner_size + j) = static_cast<float>(max_idx);
    }
  }
  return result;
}

int Impl::TensorOps::sum_equal(const Impl::Tensor &lhs, const Impl::Tensor &rhs) {
#if DEBUG
  // Hack: for a embedded function, this will not happen
  checkCppErrorsMsg(lhs.sizes() != rhs.sizes(), "Tensor sizes are not equal");
  checkCppErrorsMsg(lhs.storage->device != Impl::DeviceType::CPU || rhs.storage->device != Impl::DeviceType::CPU,
                    "This operation is not supported on CUDA");
#endif

  static const float eps = 1e-3;
  int num_elements = 0;
  for (int i = 0; i < lhs.storage->total_size; i++)
    num_elements += std::fabs(*(lhs.storage->data + i) - *(rhs.storage->data + i)) < eps;
  return num_elements;
}

void Impl::TensorOps::add_(Impl::Tensor &x, const Impl::Tensor &y) {
  if(unlikely(x.storage == y.storage)) {
    throw std::runtime_error("Inplace operation on the same operator is not supported");
  }
#if DEBUG
  // Hack: for a embedded function, this will not happen

  // Check the shape of x and y
  auto x_shape = x.sizes();
  auto y_shape = y.sizes();
  if (x_shape.size() != y_shape.size()) {
    throw std::runtime_error("x_shape != y_shape");
  }

  for (int i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] != y_shape[i]) {
      throw std::runtime_error("x_shape != y_shape");
    }
  }

  if (x.getDevice() != y.getDevice()) {
    throw std::runtime_error("x.getDevice() != y.getDevice()");
  }
#endif

  ::add_(x.data_ptr(), y.data_ptr(), x.totalSize(), x.getDevice());
}

void Impl::TensorOps::relu_(Impl::Tensor &x) {
  ::relu_(x.data_ptr(), x.totalSize(), x.getDevice());
}

void Impl::TensorOps::add_relu_(Impl::Tensor &x, const Impl::Tensor &y) {
  if(unlikely(x.storage == y.storage)) {
    throw std::runtime_error("Inplace operation on the same operator is not supported");
  }
  ::add_relu_(x.data_ptr(), y.data_ptr(), x.totalSize(), x.getDevice());
}
