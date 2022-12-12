#pragma once

#include "common.h"
#include "tensor.hpp"
#include "ops.hpp"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

// HACK: FIXME: ImageNet-oriented dataset with hard code transforms
class ImageDataset
{
    std::vector<std::string> file_paths;
    std::vector<torch::Tensor> image_tensors;
    std::vector<Tensor> batched_tensors;
    DeviceType device;

public:
    ImageDataset(DeviceType device = DeviceType::CUDA)
        : device(device) {}

    void load(const std::string &path)
    {
        // Load metadata (metadata.txt)
        std::ifstream metadata(path + "/metadata.txt");
        checkCppErrorsMsg(!metadata.is_open(),
                          ("Cannot open " + path + "/metadata.txt").c_str());
        std::string file_name;
        while (std::getline(metadata, file_name))
            file_paths.push_back(path + "/" + file_name);
        metadata.close();

        // Load images
        image_tensors.reserve(file_paths.size());
        for (const auto &file_path : file_paths)
        {
            cv::Mat image = cv::imread(file_path);
            checkCppErrorsMsg(image.empty(),
                              ("Cannot open " + file_path).c_str());

            // FIXME: hard code transforms
            cv::resize(image, image, cv::Size(256, 256));
            auto cropped_image = image(cv::Rect(16, 16, 224, 224));
            std::vector<cv::Mat> channels(3);
            cv::split(cropped_image, channels);

            const float data_mean[3] = {0.485, 0.456, 0.406};
            const float data_std[3] = {0.229, 0.224, 0.225};

            auto R = torch::from_blob(channels[2].ptr(), {224, 224}, torch::kI8);
            auto G = torch::from_blob(channels[1].ptr(), {224, 224}, torch::kI8);
            auto B = torch::from_blob(channels[0].ptr(), {224, 224}, torch::kI8);

            R = R.to(torch::kF32).div(255.0).sub(data_mean[0]).div(data_std[0]);
            G = G.to(torch::kF32).div(255.0).sub(data_mean[1]).div(data_std[1]);
            B = B.to(torch::kF32).div(255.0).sub(data_mean[2]).div(data_std[2]);

            image_tensors.emplace_back(torch::cat({R, G, B}, 0).reshape({3, 224, 224}));
        }
    }

    void partition(int batch_size)
    {
        int n_batch = (image_tensors.size() - 1) / batch_size + 1;
        int total_size = image_tensors.size();
        for (int i = 0; i < n_batch; i++)
        {
            int cur_batch_size = std::min(batch_size, total_size);
            total_size -= cur_batch_size;

            auto tensor_list = torch::TensorList(image_tensors.begin() + i * batch_size,
                                                 image_tensors.begin() + i * batch_size + cur_batch_size);
            torch::Tensor tensor = torch::stack(tensor_list, 0).contiguous();
            Tensor batched_tensor({cur_batch_size, 3, 224, 224});
            batched_tensor.fromData(reinterpret_cast<float *>(tensor.data_ptr()));
            batched_tensor.to(device);
            batched_tensors.push_back(batched_tensor);
        }
    }

    Tensor &operator[](int index)
    {
        return batched_tensors[index];
    }

    std::pair<int, int> size() const
    {
        return std::make_pair(image_tensors.size(),
                              batched_tensors.size());
    }
};