#pragma once

// common includes
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <memory>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
//Volta structure only allows FP16 to be multiplied.
#include <cuda_fp16.h>

// Type aliases

typedef __half float_16;
typedef float float_32;

// utility functions
namespace Impl {
enum DeviceType {
  CPU, CUDA
};
}

template<typename T> void printCudaError(T result, char const *const msg, const char *const file, int const line) {
  std::stringstream ss;
  if (result) {
    ss << "CUDA [Error] at: " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << "("
       << cudaGetErrorName(result) << ")" << " \"" << msg << "\"";
    // Use throw to avoid gtest exiting.
    throw std::runtime_error(ss.str());
  }
}

template<typename T> void printCppError(T result, char const *const msg, const char *const file, int const line) {
  std::stringstream ss;
  if (result) {
    ss << "CPP [Error] at: " << file << ":" << line << " \"" << msg << "\"";
    // Use throw to avoid gtest exiting.
    throw std::runtime_error(ss.str());
  }
}

template<typename T> void printWarning(T result, char const *const msg, const char *const file, int const line) {
  if (result) {
    std::cerr << "[Warning] at: " << file << ":" << line << " \"" << msg << "\" = " << result << std::endl;
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(result) printCudaError((result), #result, __FILE__, __LINE__)

#define checkCppErrors(result) printCppError((result), #result, __FILE__, __LINE__)
#define checkCppErrorsMsg(result, msg) printCppError((result), msg, __FILE__, __LINE__)

#define checkWarning(result) printWarning((result), #result, __FILE__, __LINE__)
#define checkWarningMsg(result, msg) printWarning((result), msg, __FILE__, __LINE__)

class [[maybe_unused]] SimpleTimer {
private:
  bool is_logging;
  decltype(std::chrono::system_clock::now()) start_time;
  std::chrono::duration<double, std::milli> duration{};

public:
  SimpleTimer() : is_logging(false) {}

  [[maybe_unused]] void start() {
    is_logging = true;
    start_time = std::chrono::system_clock::now();
  }

  // return (ms)
  [[maybe_unused]] double end() {
    checkWarning(!is_logging);
    is_logging = false;
    duration = std::chrono::system_clock::now() - start_time;
    return duration.count();
  }
};

[[maybe_unused]] int randomInt(int min, int max);

[[maybe_unused]] float randomFloat(float min, float max);

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
