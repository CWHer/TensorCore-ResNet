#pragma once

#include "common.h"
#include "tensor.hpp"

class Module
{
protected:
    struct NamedTensor
    {
        std::string name;
        Tensor tensor;
    };

    std::vector<NamedTensor> tensor_list;

public:
    void addTensor(const std::string &name, Tensor tensor)
    {
        tensor_list.emplace_back(NamedTensor{name, tensor});
    }

    virtual Tensor forward(Tensor x) = 0;

    virtual void printModule(const std::string &prefix) = 0;

    virtual void loadWeights(const std::string &path)
    {
        for (auto &named_tensor : tensor_list)
            named_tensor.tensor.load(path + "_" + named_tensor.name);
    }

    virtual void to(DeviceType device)
    {
        for (auto &named_tensor : tensor_list)
            named_tensor.tensor.to(device);
    }
};

template <typename ImplType>
class ModuleHolder
{
protected:
    std::shared_ptr<Module> module_impl;

public:
    /// Constructs the `ModuleHolder` with a contained module, forwarding all
    /// arguments to its constructor.
    // NOTE: borrowed from PyTorch
    template <typename Head, typename... Tail>
    ModuleHolder(Head &&head, Tail &&...tail)
        : module_impl(new ImplType(
              std::forward<Head>(head),
              std::forward<Tail>(tail)...)) {}

    ModuleHolder(std::shared_ptr<ImplType> module_impl)
        : module_impl(std::move(module_impl)) {}

    virtual Tensor forward(Tensor x)
    {
        return module_impl->forward(x);
    }

    void printModule(const std::string &prefix)
    {
        module_impl->printModule(prefix);
    }

    void loadWeights(const std::string &path)
    {
        module_impl->loadWeights(path);
    }

    void to(DeviceType device)
    {
        module_impl->to(device);
    }

    std::shared_ptr<Module> get() { return module_impl; }
};

// NOTE: borrowed from PyTorch
#define NETWORK_MODULE_IMPL(Name, ImplType)    \
    class Name : public ModuleHolder<ImplType> \
    {                                          \
    public:                                    \
        using Impl = ImplType;                 \
    }

/// Like `TORCH_MODULE_IMPL`, but defaults the `ImplType` name to `<Name>Impl`.
#define NETWORK_MODULE(Name) NETWORK_MODULE_IMPL(Name, Name##Impl)

class ModuleListImpl : public Module
{
private:
    struct NamedModule
    {
        std::string name;
        std::shared_ptr<Module> module;
    };

    std::vector<NamedModule> module_list;

public:
    ModuleListImpl() = default;

    template <typename M>
    void addModule(std::string name, ModuleHolder<M> holder)
    {
        module_list.emplace_back(NamedModule{name, holder.get()});
    }

    bool empty() { return module_list.empty(); }

    void loadWeights(const std::string &path) override
    {
        for (auto &named_module : module_list)
            named_module.module->loadWeights(path + "_" + named_module.name);
    }

    virtual Tensor forward(Tensor x) override
    {
        for (auto &named_module : module_list)
            x = named_module.module->forward(x);
        return x;
    }

    void to(DeviceType device) override
    {
        for (auto &named_module : module_list)
            named_module.module->to(device);
    }

    void printModule(const std::string &prefix) override
    {
        for (auto &named_module : module_list)
            named_module.module->printModule(prefix + "_" + named_module.name);
    }
};

class ModuleList : public ModuleHolder<ModuleListImpl>
{
public:
    using Impl = ModuleListImpl;

    ModuleList() : ModuleHolder(std::make_shared<Impl>()) {}

    void addModule(std::string name, ModuleHolder module)
    {
        std::dynamic_pointer_cast<Impl>(module_impl)->addModule(name, module);
    }

    bool empty()
    {
        return std::dynamic_pointer_cast<Impl>(module_impl)->empty();
    }
};