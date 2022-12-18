#pragma once

// common includes
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
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <bitset>
#include <array>

namespace Sim
{

    // rust type aliases
    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using i8 = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;
    using f32 = float;
    using f64 = double;

    // simulator utils
    enum CUDAMemcpyType
    {
        MemcpyHostToDevice,
        MemcpyDeviceToHost
    };

    // clang-format off
    class FatalError : public std::exception {};
    // clang-format on

    template <typename T>
    void printCppError(T result, char const *const msg,
                       const char *const file, int const line)
    {
        if (result)
        {
            std::cerr << "[Error] at: " << file << ":" << line
                      << " \"" << msg << "\"" << std::endl;
            throw FatalError();
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

    class SimpleTimer
    {
    private:
        struct TimingItem
        {
            std::chrono::system_clock::time_point last_time;
            std::chrono::duration<f64, std::milli> total_duration;
            u64 count;

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

}