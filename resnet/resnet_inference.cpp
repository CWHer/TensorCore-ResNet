#include "common.h"
#include "dataset.hpp"
#include "tensor.hpp"
#include "ops.hpp"

#include "conv.hpp"
#include "batchnorm.hpp"
#include "linear.hpp"
#include "pooling.hpp"

using namespace Impl;

class BasicBlock : public ModuleList
{
private:
    std::shared_ptr<Conv2d> conv1;
    std::shared_ptr<BatchNorm2d> bn1;
    std::shared_ptr<Conv2d> conv2;
    std::shared_ptr<BatchNorm2d> bn2;
    std::shared_ptr<ModuleList> downsample;

public:
    static const int expansion = 1;

public:
    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
               ModuleList downsample = ModuleList())
        : conv1(std::make_shared<Conv2d>(inplanes, planes, 3, stride, 1, 1)),
          bn1(std::make_shared<BatchNorm2d>(planes)),
          conv2(std::make_shared<Conv2d>(planes, planes, 3, 1, 1, 1)),
          bn2(std::make_shared<BatchNorm2d>(planes)),
          downsample(std::make_shared<ModuleList>(downsample))
    {
        addModule("conv1", conv1);
        addModule("bn1", bn1);
        addModule("conv2", conv2);
        addModule("bn2", bn2);
        if (!downsample.empty())
            addModule("downsample", this->downsample);
    }

  Tensor forward(const Tensor &x) override {
    Tensor identity = x, result;

    result = conv1->forward(x);
    result = bn1->forward(result);
    TensorOps::relu_(result);

    result = conv2->forward(result);
    result = bn2->forward(result);

    if (!downsample->empty())
      identity = downsample->forward(identity);

    TensorOps::add_(result, identity);
    TensorOps::relu_(result);

    return result;
  }
};

class ResNet18 : public ModuleList
{
private:
    int64_t inplanes = 64;
    std::shared_ptr<Conv2d> conv1;
    std::shared_ptr<BatchNorm2d> bn1;
    std::shared_ptr<MaxPool2d> maxpool;
    std::shared_ptr<ModuleList> layer1;
    std::shared_ptr<ModuleList> layer2;
    std::shared_ptr<ModuleList> layer3;
    std::shared_ptr<ModuleList> layer4;
    std::shared_ptr<AdaptiveAvgPool2d> avgpool;
    std::shared_ptr<Linear> fc;

public:
    ResNet18(const std::string &model_path,
             std::vector<int> layers = {2, 2, 2, 2},
             int64_t num_classes = 1000)
        : conv1(std::make_shared<Conv2d>(3, inplanes, 7, 2, 3)),
          bn1(std::make_shared<BatchNorm2d>(inplanes)),
          maxpool(std::make_shared<MaxPool2d>(3, 2, 1)),
          layer1(std::make_shared<ModuleList>(makeLayer(64, layers[0]))),
          layer2(std::make_shared<ModuleList>(makeLayer(128, layers[1], 2))),
          layer3(std::make_shared<ModuleList>(makeLayer(256, layers[2], 2))),
          layer4(std::make_shared<ModuleList>(makeLayer(512, layers[3], 2))),
          avgpool(std::make_shared<AdaptiveAvgPool2d>(1)),
          fc(std::make_shared<Linear>(512 * BasicBlock::expansion, num_classes))
    {
        addModule("conv1", conv1);
        addModule("bn1", bn1);
        addModule("maxpool", maxpool);
        addModule("layer1", layer1);
        addModule("layer2", layer2);
        addModule("layer3", layer3);
        addModule("layer4", layer4);
        addModule("avgpool", avgpool);
        addModule("fc", fc);

        // TODO: load weights from model_path
    }

  Tensor forward(const Tensor &x) override {
    Tensor result;
    result = conv1->forward(x);
    result = bn1->forward(result);
    TensorOps::relu_(result);
    result = maxpool->forward(result);

    result = layer1->forward(result);
    result = layer2->forward(result);
    result = layer3->forward(result);
    result = layer4->forward(result);

    result = avgpool->forward(result);
    auto x_shape = result.sizes();
    result.view({x_shape[0], static_cast<int>(result.totalSize() / x_shape[0])});
    result = fc->forward(x);

    return result;
  }

private:
    ModuleList makeLayer(int64_t planes, int64_t blocks, int64_t stride = 1)
    {
        ModuleList downsample;
        if (stride != 1 || inplanes != planes * BasicBlock::expansion)
        {
            downsample.addModule("0", std::make_shared<Conv2d>(
                                          inplanes, planes * BasicBlock::expansion, 1, stride));
            downsample.addModule("1", std::make_shared<BatchNorm2d>(
                                          planes * BasicBlock::expansion));
        }

        ModuleList layers;
        layers.addModule("0", std::make_shared<BasicBlock>(
                                  inplanes, planes, stride, downsample));
        inplanes = planes * BasicBlock::expansion;
        for (int64_t i = 1; i < blocks; i++)
            layers.addModule(std::to_string(i), std::make_shared<BasicBlock>(inplanes, planes));
        return layers;
    }
};

int main()
{
    // 1. load resnet18 model
    // 2. load image dataset
    // 3. inference

    Tensor x({50, 3, 224, 224});
    ResNet18 resnet18("path");
    resnet18.printModule("resnet18");
    std::cout << std::endl;

    x = resnet18.forward(x);
    std::cout << x << std::endl;

    return 0;
}
