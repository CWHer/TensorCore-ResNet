#include "common.h"
#include "dataset.hpp"
#include "tensor.hpp"
#include "ops.hpp"

#include "conv.hpp"
#include "batchnorm.hpp"
#include "linear.hpp"
#include "pooling.hpp"

class BasicBlockImpl : public ModuleListImpl
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
    BasicBlockImpl(int64_t inplanes, int64_t planes, int64_t stride = 1,
                   ModuleList downsample = ModuleList())
        : conv1(inplanes, planes, 3, stride), bn1(planes),
          conv2(planes, planes, 3), bn2(planes), downsample(downsample)
    {
        addModule("conv1", conv1);
        addModule("bn1", bn1);
        addModule("conv2", conv2);
        addModule("bn2", bn2);
        if (!downsample.empty())
            addModule("downsample", this->downsample);
    }

    Tensor forward(Tensor x) override
    {
        Tensor identity(x.clone());

        x = conv1.forward(x);
        x = bn1.forward(x);
        TensorOps::relu_(x);

        x = conv2.forward(x);
        x = bn2.forward(x);

        if (!downsample.empty())
            identity = downsample.forward(identity);

        TensorOps::add_(x, identity);
        TensorOps::relu_(x);

        return x;
    }
};

class BasicBlock : public ModuleHolder<BasicBlockImpl>
{
};

class ResNet18Impl : public ModuleListImpl
{
private:
    int64_t inplanes = 64;
    Conv2d conv1;
    BatchNorm2d bn1;
    MaxPool2d maxpool;
    ModuleList layer1;
    ModuleList layer2;
    ModuleList layer3;
    ModuleList layer4;
    AdaptiveAvgPool2d avgpool;
    Linear fc;

public:
    ResNet18Impl(const std::string &model_path,
                 std::vector<int> layers = {2, 2, 2, 2},
                 int64_t num_classes = 1000)
        : conv1(3, inplanes, 7, 2, 3),
          bn1(inplanes), maxpool(3, 2, 1),
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
        TensorOps::relu_(x);
        x = maxpool.forward(x);

        x = layer1.forward(x);
        x = layer2.forward(x);
        x = layer3.forward(x);
        x = layer4.forward(x);

        x = avgpool.forward(x);
        auto x_shape = x.sizes();
        x.view({x_shape[0], x.totalSize() / x_shape[0]});
        x = fc.forward(x);

        return x;
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
        // FIXME:
        layers.addModule("0", std::make_shared<BasicBlock>(
                                  inplanes, planes, stride, downsample));
        inplanes = planes * BasicBlock::expansion;
        for (int64_t i = 1; i < blocks; i++)
            layers.addModule(std::to_string(i), std::make_shared<BasicBlock>(inplanes, planes));
        return layers;
    }
};

NETWORK_MODULE(ResNet18);

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
