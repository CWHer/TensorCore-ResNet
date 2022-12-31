#pragma once

#include "common.h"
#include "tensor.hpp"
#include "ops.hpp"

class ImageDataset
{
    std::deque<Tensor> inputs, std_logits;
    DeviceType device;

public:
    ImageDataset(DeviceType device = DeviceType::CUDA) : device(device) {}

    void load(const std::string &path, int num_batches)
    {
        // NOTE: HACK: use preprocess.py to generate the binary files
        for (int i = 0; i < num_batches; i++)
        {
            std::string index = std::to_string(i);
            index = std::string(4 - index.size(), '0') + index;

            Tensor input, std_logit;
            input.load(path + "/images_" + index + ".bin");
            std_logit.load(path + "/labels_" + index + ".bin");
            inputs.push_back(input);
            std_logits.push_back(std_logit);
        }
    }

    std::pair<Tensor, Tensor> next()
    {
        checkCppErrorsMsg(inputs.empty(), "No more batches");
        Tensor input = inputs.front();
        Tensor std_logit = std_logits.front();
        inputs.pop_front(), std_logits.pop_front();
        input.to(device), std_logit.to(device);
        return std::make_pair(input, std_logit);
    }

    int size() const { return inputs.size(); }
};