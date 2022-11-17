#pragma once

// common includes
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>
#include <algorithm>

// utility functions
enum DeviceType
{
    CPU,
    CUDA
};

enum ErrorType
{
    CUDA_ERROR,
    CPP_ERROR
};

template <typename T>
void printError(T result, char const *const msg,
                ErrorType type, const char *const file, int const line)
{
    if (result)
    {
        switch (type)
        {
        case CUDA_ERROR:
            std::cerr << "CUDA [Error] at: " << file << ":" << line << " code="
                      << static_cast<unsigned int>(result) << "(" << cudaGetErrorName(result) << ")"
                      << " \"" << msg << "\"" << std::endl;
            break;
        case CPP_ERROR:
            std::cerr << "CPP [Error] at: " << file << ":" << line
                      << " \"" << msg << "\"" << std::endl;
            break;
        }
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
        break;
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(result) printError((result), #result, ErrorType::CUDA_ERROR, __FILE__, __LINE__)

#define checkCppErrors(result) printError((result), #result, ErrorType::CPP_ERROR, __FILE__, __LINE__)
#define checkCppErrorsMsg(result, msg) printError((result), msg, ErrorType::CPP_ERROR, __FILE__, __LINE__)

#define checkWarning(result) printWarning((result), #result, __FILE__, __LINE__)
#define checkWarningMsg(result, msg) printWarning((result), msg, __FILE__, __LINE__)

class SimpleTimer
{
private:
    bool is_logging;
    decltype(std::chrono::system_clock::now()) start_time;
    std::chrono::duration<double, std::milli> duration;

public:
    SimpleTimer() : is_logging(false) {}

    void start()
    {
        is_logging = true;
        start_time = std::chrono::system_clock::now();
    }

    // return (ms)
    double end()
    {
        checkWarning(!is_logging);
        is_logging = false;
        duration = std::chrono::system_clock::now() - start_time;
        return duration.count();
    }
};
