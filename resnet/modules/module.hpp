#pragma once

#include "common.h"
#include "tensor.hpp"

class Module
{
    // NOTE: HACK: implicit copy is a shallow copy

    // TODO
public:
    virtual void loadWeights(const std::string &path) {}
    virtual Tensor forward(Tensor x) {}
    virtual void to(DeviceType device) {}
};

class ModuleList : public Module
{
    // TODO
private:
    struct NamedModule
    {
        std::string name;
        Module module;
    };

    std::vector<NamedModule> module_list;

public:
    void addModule(std::string name, Module module)
    {
        module_list.emplace_back((NamedModule){name, module});
    }

    bool empty() { return module_list.empty(); }

    void loadWeights(const std::string &path)
    {
        for (auto &named_module : module_list)
            named_module.module.loadWeights(path + "_" + named_module.name);
    }

    virtual Tensor forward(Tensor x)
    {
        for (auto &named_module : module_list)
            x = named_module.module.forward(x);
        return x;
    }

    void to(DeviceType device)
    {
        for (auto &named_module : module_list)
            named_module.module.to(device);
    }
};