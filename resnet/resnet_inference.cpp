#include "common.h"
#include "dataset.hpp"
#include "tensor.hpp"
#include "ops.hpp"

#include "conv.hpp"
#include "batchnorm.hpp"
#include "linear.hpp"
#include "pooling.hpp"

class BasicBlock : public ModuleList
{
private:
    std::shared_ptr<Conv2d> conv1;
    std::shared_ptr<BatchNorm2dRelu> bn1;
    std::shared_ptr<Conv2d> conv2;
    std::shared_ptr<BatchNorm2d> bn2;
    std::shared_ptr<ModuleList> downsample;

public:
    static const int expansion = 1;

public:
    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1,
               ModuleList downsample = ModuleList())
        : conv1(std::make_shared<Conv2d>(inplanes, planes, 3, stride, 1)),
          bn1(std::make_shared<BatchNorm2dRelu>(planes)),
          conv2(std::make_shared<Conv2d>(planes, planes, 3, 1, 1)),
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

    Tensor forward(Tensor x) override
    {
        Tensor identity(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        // TensorOps::relu_(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        if (!downsample->empty())
            identity = downsample->forward(identity);

        // TensorOps::add_(x, identity);
        // TensorOps::relu_(x);
        TensorOps::addRelu_(x, identity);

        return x;
    }
};

class ResNet18 : public ModuleList
{
private:
    int64_t inplanes = 64;
    std::shared_ptr<Conv2d> conv1;
    std::shared_ptr<BatchNorm2dRelu> bn1;
    std::shared_ptr<MaxPool2d> maxpool;
    std::shared_ptr<ModuleList> layer1;
    std::shared_ptr<ModuleList> layer2;
    std::shared_ptr<ModuleList> layer3;
    std::shared_ptr<ModuleList> layer4;
    std::shared_ptr<AvgPool2d> avgpool;
    std::shared_ptr<Linear> fc;

public:
    ResNet18(std::vector<int> layers = {2, 2, 2, 2},
             int64_t num_classes = 1000)
        : conv1(std::make_shared<Conv2d>(3, inplanes, 7, 2, 3)),
          bn1(std::make_shared<BatchNorm2dRelu>(inplanes)),
          maxpool(std::make_shared<MaxPool2d>(3, 2, 1)),
          layer1(std::make_shared<ModuleList>(makeLayer(64, layers[0]))),
          layer2(std::make_shared<ModuleList>(makeLayer(128, layers[1], 2))),
          layer3(std::make_shared<ModuleList>(makeLayer(256, layers[2], 2))),
          layer4(std::make_shared<ModuleList>(makeLayer(512, layers[3], 2))),
          avgpool(std::make_shared<AvgPool2d>(7)),
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
    }

    Tensor forward(Tensor x) override
    {
        x = conv1->forward(x);
        x = bn1->forward(x);
        // TensorOps::relu_(x);
        x = maxpool->forward(x);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = avgpool->forward(x);
        auto x_shape = x.sizes();
        x.reshape({x_shape[0], x.totalSize() / x_shape[0]});
        x = fc->forward(x);

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
#ifdef DATASET_ROOT
    const int N_IMAGES = 5000;
    const int BATCH_SIZE = 16;
    constexpr int NUM_BATCHES = (N_IMAGES - 1) / BATCH_SIZE + 1;
    ImageDataset dataset(DeviceType::CUDA);
    dataset.load(DATASET_ROOT, NUM_BATCHES);
#else
    std::cout << "Please set DATASET_ROOT to the root directory of the dataset." << std::endl;
    throw std::runtime_error("DATASET_ROOT not set");
#endif

#ifdef RESNET18_ROOT
    std::string model_dir = RESNET18_ROOT;

    ResNet18 resnet18;
    resnet18.loadWeights(model_dir + "/resnet18");
    resnet18.to(DeviceType::CUDA);
    resnet18.printModule("resnet18");
    std::cout << std::endl;

    SimpleTimer timer;
    int num_correct = 0;
    int num_total = 0;
    timer.start("total_time");
    for (int i = 0; i < NUM_BATCHES; i++)
    {
        auto data = dataset.next();
        auto inputs = data.first, std_logits = data.second;

        auto logits = resnet18.forward(inputs);
        logits.to(DeviceType::CPU);

        Tensor label = TensorOps::argmax(logits, 1);
        Tensor std_label = TensorOps::argmax(std_logits, 1);

        num_correct += TensorOps::sum_equal(label, std_label);
        num_total += inputs.sizes()[0];
    }
    // clang-format off
    std::cout << "Accuracy Compared to PyTorch ResNet18 Implementation: "
              << num_correct << "/" << num_total << '\n' << std::endl;
    // clang-format on
    timer.end("total_time");
    timer.printStat("total_time");

    Tensor x({16, 3, 224, 224}, DeviceType::CPU);
    Tensor y({16, 1000}, DeviceType::CPU);
    for (int i = 0; i < 100; i++)
    {
        timer.start("inference");
        x.to(DeviceType::CUDA);
        timer.start("forward");
        auto logits = resnet18.forward(x);
        timer.end("forward");
        logits.to(DeviceType::CPU);
        Tensor label = TensorOps::argmax(logits, 1);
        Tensor std_label = TensorOps::argmax(y, 1);
        TensorOps::sum_equal(label, std_label);
        timer.end("inference");
        x.to(DeviceType::CPU);
    }
    timer.printStat("inference");
    timer.printStat("forward");
    resnet18.printStat("resnet18");

    freeMemCache();
#else
    std::cout << "Please set RESNET18_ROOT to the directory of resnet18 model" << std::endl;
    throw std::runtime_error("RESNET18_ROOT not set");
#endif

    return 0;
}
