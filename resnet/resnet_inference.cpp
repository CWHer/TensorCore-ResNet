#include "common.h"
#include "data.hpp"
#include "tensor.hpp"
#include "ops.hpp"

#include "conv.hpp"
#include "batchnorm.hpp"
#include "linear.hpp"
#include "pooling.hpp"

class BasicBlock : public ModuleList
{
private:
    Conv2d conv1;
    BatchNorm2d bn1;
    Conv2d conv2;
    BatchNorm2d bn2;
    ModuleList downsample;

public:
    static const int expansion = 1;

public:
    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
               ModuleList downsample = ModuleList())
        : conv1(inplanes, planes, 3, stride), bn1(planes),
          conv2(planes, planes, 3), bn2(planes), downsample(downsample)
    {
        addModule("conv1", conv1);
        addModule("bn1", bn1);
        addModule("conv2", conv2);
        addModule("bn2", bn2);
        if (!downsample->empty())
            addModule("downsample", downsample);
    }

    Tensor forward(Tensor x) override
    {
        Tensor identity(x.clone());

        x = conv1.forward(x);
        x = bn1.forward(x);
        x = relu(x);

        x = conv2.forward(x);
        x = bn2.forward(x);

        if (!downsample.empty())
            identity = downsample.forward(identity);

        x += identity;
        x = relu(x);

        return x;
    }
};

class ResNet18 : public ModuleList
{
private:
    int64_t inplanes = 64;
    Conv2d conv1;
    BatchNorm2d bn1;
    MaxPool2d maxpool;
    std::vector<Module> layer1;
    std::vector<Module> layer2;
    std::vector<Module> layer3;
    std::vector<Module> layer4;
    AdaptiveAvgPool2d avgpool;
    Linear fc;

public:
    ResNet18(const std::string &model_path,
             std::vector<int> layers = {2, 2, 2, 2},
             int64_t num_classes = 1000)
        : conv1(3, inplanes, 7, 2, 3), bn1(inplanes), maxpool(3, 2, 1),
          layer1(makeLayer(64, layers[0])),
          layer2(makeLayer(128, layers[1], 2)),
          layer3(makeLayer(256, layers[2], 2)),
          layer4(makeLayer(512, layers[3], 2)),
          avgpool(1), fc(512 * BasicBlock::expansion, num_classes)
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

    Tensor forward(Tensor x) override
    {
        x = conv1.forward(x);
        x = bn1.forward(x);
        x = relu(x);
        x = maxpool.forward(x);

        x = layer1.forward(x);
        x = layer2.forward(x);
        x = layer3.forward(x);
        x = layer4.forward(x);

        x = avgpool.forward(x);
        x.view({x.sizes()[0], -1});
        x = fc.forward(x);

        return x;
    }

private:
    ModuleList makeLayer(int64_t planes, int64_t blocks, int64_t stride = 1)
    {
        ModuleList downsample;
        if (stride != 1 || inplanes != planes * BasicBlock::expansion)
        {
            downsample.addModule("0", Conv2d(inplanes, planes * BasicBlock::expansion, 1, stride));
            downsample.addModule("1", BatchNorm2d(planes * BasicBlock::expansion));
        }

        ModuleList layers;
        layers.addModule("0", BasicBlock(inplanes, planes, stride, downsample));
        inplanes = planes * BasicBlock::expansion;
        for (int64_t i = 1; i < blocks; i++)
            layers.addModule(std::to_string(i), BasicBlock(inplanes, planes));
        return layers;
    }
};

int main()
{
    // 1. load resnet18 model
    // 2. load image dataset
    // 3. inference

    return 0;
}