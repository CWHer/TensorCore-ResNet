#pragma once

#include <utility>

#include "common.hpp"
#include "tensor.hpp"

namespace Impl {

class Module {
private:
  struct NamedTensor {
    std::string name;
    Tensor &tensor;
  };

  std::vector<NamedTensor> tensor_list;

protected:
  SimpleTimer timer;

public:
  virtual Tensor forward(const Tensor &x) = 0;

  virtual Tensor forward(Tensor &&x) {
    return forward(x);
  }

  virtual void printModule(const std::string &prefix) = 0;

  virtual void printStat(const std::string &prefix) = 0;

  void addTensor(const std::string &name, Tensor &tensor) {
    tensor_list.emplace_back(NamedTensor{name, tensor});
  }

  virtual void loadWeights(const std::string &path) {
    for (auto &named_tensor : tensor_list)
      named_tensor.tensor.load(path + "_" + named_tensor.name + ".bin");
  }

  virtual void to(Impl::DeviceType device) {
    for (auto &named_tensor : tensor_list)
      named_tensor.tensor.to(device);
  }
};

class ModuleList : public Module {
private:
  struct NamedModule {
    std::string name;
    std::shared_ptr<Module> module;
  };

  std::vector<NamedModule> module_list;

public:
  void addModule(std::string name, std::shared_ptr<Module> module) {
    module_list.emplace_back(NamedModule{std::move(name), std::move(module)});
  }

  bool empty() { return module_list.empty(); }

  void loadWeights(const std::string &path) override {
    for (auto &named_module : module_list)
      named_module.module->loadWeights(path + "_" + named_module.name);
  }

  Tensor forward(const Tensor &x) override {
    Tensor result = x;
    checkCppErrorsMsg(module_list.empty(), "ModuleList is empty");
    for (auto &named_module : module_list)
      result = named_module.module->forward(result);
    return result;
  }

  Tensor forward(Tensor &&x) override {
    for (auto &named_module : module_list)
      x = named_module.module->forward(std::move(x));
    return x;
  }

  void to(Impl::DeviceType device) override {
    for (auto &named_module : module_list)
      named_module.module->to(device);
  }

  void printModule(const std::string &prefix) override {
    for (auto &named_module : module_list)
      named_module.module->printModule(prefix + "_" + named_module.name);
  }

  void printStat(const std::string &prefix) override {
    for (auto &named_module : module_list)
      named_module.module->printStat(prefix + "_" + named_module.name);
  }
};

}
