#pragma once

#include "common.h"
#include "tensor.hpp"

class ImageDataset
{

public:
    // TODO

    void load(const std::string &path)
    {
    }
};

class DataPrefetcher
{
public:
    Tensor next()
    {
        // TODO
    }
    // CUDA stream, AsyncCopy
};