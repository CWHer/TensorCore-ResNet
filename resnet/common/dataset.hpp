#pragma once

#include "common.h"
#include "tensor.hpp"
#include "ops.hpp"

class ImageDataset
{
    std::vector<Tensor> batched_tensors;
    std::vector<Tensor> batched_labels, predicted_logits;
    DeviceType device;

public:
    ImageDataset(DeviceType device =
                     DeviceType::CUDA) : device(device) {}

    void load(const std::string &path, int num_tensors)
    {
        // NOTE: HACK: use preprocess.py to generate the binary files
        // Load batched tensors
        for (int i = 0; i < num_tensors; i++)
        {
            batched_tensors.emplace_back();
            batched_labels.emplace_back();
            std::string index = std::to_string(i);
            index = std::string(4 - index.size(), '0') + index;
            batched_tensors.back().load(path + "/images_" + index + ".bin");
            batched_labels.back().load(path + "/labels_" + index + ".bin");
        }
    }

    Tensor operator[](int index)
    {
        return batched_tensors[index];
    }

    void logPredicted(Tensor logits)
    {
        predicted_logits.push_back(logits);
    }

    int size() const { return batched_tensors.size(); }

    void printStats()
    {
        int num_correct = 0;
        int total_num = 0;
        for (int i = 0; i < this->size(); i++)
        {
            Tensor predicted_label = TensorOps::argmax(predicted_logits[i], 1);
            num_correct += TensorOps::sum_equal(
                batched_labels[i], predicted_label);
            total_num += batched_labels[i].sizes()[0];
        }
        std::cout << "Accuracy: " << num_correct * 100.0 / total_num << "%" << std::endl;
    }
};