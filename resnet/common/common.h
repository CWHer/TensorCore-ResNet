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
#include <unordered_map>

// utility functions
enum DeviceType
{
    CPU,
    CUDA,
    UNKNOWN
};

template <typename T>
void printCudaError(T result, char const *const msg,
                    const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA [Error] at: " << file << ":" << line << " code="
                  << static_cast<unsigned int>(result) << "(" << cudaGetErrorName(result) << ")"
                  << " \"" << msg << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void printCppError(T result, char const *const msg,
                   const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CPP [Error] at: " << file << ":" << line
                  << " \"" << msg << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
void printWarning(T result, char const *const msg,
                  const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "[Warning] at: " << file << ":" << line
                  << " \"" << msg << "\" = " << result << std::endl;
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(result) printCudaError((result), #result, __FILE__, __LINE__)

#define checkCppErrors(result) printCppError((result), #result, __FILE__, __LINE__)
#define checkCppErrorsMsg(result, msg) printCppError((result), msg, __FILE__, __LINE__)

#define checkWarning(result) printWarning((result), #result, __FILE__, __LINE__)
#define checkWarningMsg(result, msg) printWarning((result), msg, __FILE__, __LINE__)

class SimpleTimer
{
private:
    struct TimingItem
    {
        std::chrono::system_clock::time_point last_time;
        std::chrono::duration<double, std::milli> total_duration;
        uint64_t count;

        TimingItem()
            : last_time(std::chrono::system_clock::now()),
              total_duration(0.0), count(0) {}
    };

    std::unordered_map<std::string, TimingItem> timing_items;

public:
    SimpleTimer() {}

    void start(const std::string &name)
    {
        auto &timing_item = timing_items[name];
        timing_item.last_time = std::chrono::system_clock::now();
    }

    void end(const std::string &name)
    {
        auto &timing_item = timing_items[name];
        auto duration = std::chrono::system_clock::now() - timing_item.last_time;
        timing_item.total_duration += duration;
        timing_item.count++;
    }

    void printStat(const std::string &name)
    {
        auto &timing_item = timing_items[name];
        std::cout << "Timing: " << name << " = "
                  << timing_item.total_duration.count() / timing_item.count
                  << " ms" << std::endl;
    }
};

int randomInt(int min, int max);

float randomFloat(float min, float max);