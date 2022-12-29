/** @file RESNET.hpp
*/#ifndef TENSORCORE_RESNET_COMMON_RESNET_HPP
#define TENSORCORE_RESNET_COMMON_RESNET_HPP

#include "module.hpp"
#include "conv.hpp"
#include "batchnorm.hpp"
#include "ops.hpp"
#include "pooling.hpp"
#include "linear.hpp"

namespace Impl {

class BasicBlock : public ModuleList {
private:
  std::shared_ptr<Conv2d> conv1;
#if DISABLE_FUSE
  std::shared_ptr<BatchNorm2d> bn1;
#else
  std::shared_ptr<BatchNorm2dRelu> bn1;
#endif
  std::shared_ptr<Conv2d> conv2;
  std::shared_ptr<BatchNorm2d> bn2;
  std::shared_ptr<ModuleList> downsample;

public:
  static const int expansion = 1;

public:
  BasicBlock(int64_t inplanes, int64_t planes, int64_t stride = 1, ModuleList downsample = ModuleList());

  Tensor forward(const Tensor &x) override;

  Tensor forward(Tensor &&x) override;
};

class ResNet18 : public ModuleList {
private:
  int64_t inplanes = 64;
  std::shared_ptr<Conv2d> conv1;
#if DISABLE_FUSE
  std::shared_ptr<BatchNorm2d> bn1;
#else
  std::shared_ptr<BatchNorm2dRelu> bn1;
#endif
  std::shared_ptr<MaxPool2d> maxpool;
  std::shared_ptr<ModuleList> layer1;
  std::shared_ptr<ModuleList> layer2;
  std::shared_ptr<ModuleList> layer3;
  std::shared_ptr<ModuleList> layer4;
  std::shared_ptr<AvgPool2d> avgpool;
  std::shared_ptr<Linear> fc;

public:
  ResNet18(const std::string &model_path, std::vector<int> layers = {2, 2, 2, 2}, int64_t num_classes = 1000);

  Tensor forward(const Tensor &x) override;

  Tensor forward(Tensor &&x) override;

private:
  ModuleList makeLayer(int64_t planes, int64_t blocks, int64_t stride = 1);
};
}

#endif //TENSORCORE_RESNET_COMMON_RESNET_HPP
