/** @file RESNET.cpp
*/

#include "resnet.hpp"

using namespace Impl;

BasicBlock::BasicBlock(int64_t inplanes, int64_t planes, int64_t stride, ModuleList downsample)
    : conv1(std::make_shared<Conv2d>(inplanes, planes, 3, stride, 1, 1)),
      bn1(std::make_shared<BatchNorm2d>(planes)),
      conv2(std::make_shared<Conv2d>(planes, planes, 3, 1, 1, 1)),
      bn2(std::make_shared<BatchNorm2d>(planes)),
      downsample(std::make_shared<ModuleList>(downsample)) {
  addModule("conv1", conv1);
  addModule("bn1", bn1);
  addModule("conv2", conv2);
  addModule("bn2", bn2);
  if (!downsample.empty())
    addModule("downsample", this->downsample);
}

Tensor BasicBlock::forward(const Tensor &x) {
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

Tensor BasicBlock::forward(Tensor &&x) {
  Tensor identity = x, result;

  result = conv1->forward(std::move(x));
  result = bn1->forward(std::move(result));
  TensorOps::relu_(result);

  result = conv2->forward(std::move(result));
  result = bn2->forward(std::move(result));

  if (!downsample->empty())
    identity = downsample->forward(std::move(identity));

  TensorOps::add_(result, identity);
  TensorOps::relu_(result);

  return result;
}

ResNet18::ResNet18(const std::string &model_path, std::vector<int> layers, int64_t num_classes)
    : conv1(std::make_shared<Conv2d>(3, inplanes, 7, 2, 3)),
      bn1(std::make_shared<BatchNorm2d>(inplanes)),
      maxpool(std::make_shared<MaxPool2d>(3, 2, 1)),
      layer1(std::make_shared<ModuleList>(makeLayer(64, layers[0]))),
      layer2(std::make_shared<ModuleList>(makeLayer(128, layers[1], 2))),
      layer3(std::make_shared<ModuleList>(makeLayer(256, layers[2], 2))),
      layer4(std::make_shared<ModuleList>(makeLayer(512, layers[3], 2))),
      avgpool(std::make_shared<AvgPool2d>(7)),
      fc(std::make_shared<Linear>(512 * BasicBlock::expansion, num_classes)) {
  addModule("conv1", conv1);
  addModule("bn1", bn1);
  addModule("maxpool", maxpool);
  addModule("layer1", layer1);
  addModule("layer2", layer2);
  addModule("layer3", layer3);
  addModule("layer4", layer4);
  addModule("avgpool", avgpool);
  addModule("fc", fc);

  this->loadWeights(model_path);
}

Tensor ResNet18::forward(const Tensor &x) {
  Tensor result = x;
  return forward(result);
}

Tensor ResNet18::forward(Tensor &&x) {
  x = conv1->forward(std::move(x));
  x = bn1->forward(std::move(x));
  TensorOps::relu_(x);
  x = maxpool->forward(std::move(x));
  x = layer1->forward(std::move(x));
  x = layer2->forward(std::move(x));
  x = layer3->forward(std::move(x));
  x = layer4->forward(std::move(x));
  x = avgpool->forward(std::move(x));
  auto x_shape = x.sizes();
  x.reshape({x_shape[0], static_cast<int>(x.totalSize() / x_shape[0])});
  x = fc->forward(std::move(x));
  return x;
}

ModuleList ResNet18::makeLayer(int64_t planes, int64_t blocks, int64_t stride) {
  ModuleList downsample;
  if (stride != 1 || inplanes != planes * BasicBlock::expansion) {
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
