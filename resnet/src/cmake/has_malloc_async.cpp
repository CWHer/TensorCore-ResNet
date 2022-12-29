/** @file has_malloc_async.cpp
 *
 * Test if the device has cudaMallocAsync
*/

#include <cuda_runtime.h>

int main() {
  int *ptr;
  auto result = cudaMallocAsync(&ptr, 1024, nullptr);
  if (result == cudaSuccess) {
    cudaFree(ptr);
    return 0;
  } else {
    return 1;
  }
}
