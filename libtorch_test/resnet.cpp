#include <torch/torch.h>
#include <iostream>

torch::nn::Conv2dOptions convOptions(
    int64_t in_planes, int64_t out_planes, int64_t kernel_size,
    int64_t stride = 1, int64_t padding = 1, bool with_bias = false)
{
    auto conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size);
    conv_options.stride(stride).padding(padding).bias(with_bias);
    return conv_options;
}

struct BasicBlock : torch::nn::Module
{
private:
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Sequential downsample;

public:
    static const int expansion = 1;

public:
    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
               torch::nn::Sequential downsample = torch::nn::Sequential())
        : conv1(convOptions(inplanes, planes, 3, stride)), bn1(planes),
          conv2(convOptions(planes, planes, 3)), bn2(planes), downsample(downsample)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        if (!downsample->is_empty())
            register_module("downsample", downsample);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        at::Tensor identity(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        if (!downsample->is_empty())
            identity = downsample->forward(identity);

        x += identity;
        x = torch::relu(x);

        return x;
    }
};

template <class Block>
class ResNet : public torch::nn::Module
{
private:
    int64_t inplanes = 64;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::MaxPool2d maxpool;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::AdaptiveAvgPool2d avgpool;
    torch::nn::Linear fc;

public:
    ResNet(torch::IntList layers, int64_t num_classes = 1000)
        : conv1(convOptions(3, inplanes, 7, 2, 3)), bn1(inplanes),
          maxpool(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),
          layer1(makeLayer(64, layers[0])),
          layer2(makeLayer(128, layers[1], 2)),
          layer3(makeLayer(256, layers[2], 2)),
          layer4(makeLayer(512, layers[3], 2)),
          avgpool(torch::nn::AdaptiveAvgPool2dOptions(1)),
          fc(512 * Block::expansion, num_classes)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("maxpool", maxpool);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("avgpool", avgpool);
        register_module("fc", fc);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = maxpool->forward(x);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = avgpool->forward(x);
        x = torch::flatten(x, 1);
        x = fc->forward(x);

        return x;
    }

private:
    torch::nn::Sequential makeLayer(int64_t planes, int64_t blocks, int64_t stride = 1)
    {
        torch::nn::Sequential downsample;
        if (stride != 1 || inplanes != planes * Block::expansion)
        {
            downsample = torch::nn::Sequential(
                torch::nn::Conv2d(convOptions(inplanes, planes * Block::expansion, 1, stride, 0)),
                torch::nn::BatchNorm2d(planes * Block::expansion));
        }
        torch::nn::Sequential layers;
        layers->push_back(Block(inplanes, planes, stride, downsample));
        inplanes = planes * Block::expansion;
        for (int64_t i = 1; i < blocks; i++)
            layers->push_back(Block(inplanes, planes));

        return layers;
    }
};

ResNet<BasicBlock> resnet18()
{
    ResNet<BasicBlock> model({2, 2, 2, 2});
    return model;
}

int main()
{
    // NOTE: getting started with libtorch
    // https://gist.github.com/CWHer/42de632afa2c0453c071c4b826377192
    // TODO: custom dataset
    // https://github.com/pytorch/examples/tree/main/cpp/custom-dataset
    // TODO: load pre-trained model

    std::cout << "Test" << std::endl;
    auto device = torch::Device("cuda");

    torch::Tensor x = torch::rand({16, 3, 224, 224}).to(device);
    ResNet<BasicBlock> resnet = resnet18();
    resnet.to(device);

    x = resnet.forward(x);
    std::cout << x.sizes() << std::endl;
}