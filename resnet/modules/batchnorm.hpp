#pragma once


#include "module.hpp"
#include <string>

namespace Impl {
class BatchNorm2d : public Module
{
protected:
    int num_features;
    double eps;
    Tensor weight, bias, running_mean, running_var;

public:
    // FIXME: these parameters are not used,
    //  recommend to add conflict check or remove them
    BatchNorm2d(int num_features, double eps = 1e-5, double momentum = 0.1,
                bool affine = true, bool track_running_stats = true);

    Tensor forward(const Tensor &x) override;
    Tensor forward(Tensor &&x) override;

    void printModule(const std::string &prefix) override;
};

class BatchNorm2dRelu : public Module
{
protected:
  int num_features;
  double eps;
  Tensor weight, bias, running_mean, running_var;


public:
  BatchNorm2dRelu(int num_features, double eps = 1e-5, double momentum = 0.1,
                  bool affine = true, bool track_running_stats = true);

  Tensor forward(const Tensor &x) override;
  Tensor forward(Tensor &&x) override;

  void printModule(const std::string &prefix) override;
};
}
