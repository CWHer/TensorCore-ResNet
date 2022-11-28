#pragma once

#include "common.h"
#include "module.hpp"
#include <cmath>

class BatchNorm2d : public Module
{
private:
	int num_features;
	double eps, momentum;
	bool affine, track_running_stats;
public:
    BatchNorm2d(int num_features, double eps = 1e-5, double momentum = 0.1,
                bool affine = true, bool track_running_stats = true)
		: num_features(num_features), eps(eps), momentum(momentum), affine(affine), 
		  track_running_stats(track_running_stats)
    {
    }

    Tensor forward(Tensor x) override
    {
		int batch = x.sizes()[0];
		int channel = x.sizes()[1];
		int num_elements = x.sizes()[2] * x.sizes()[3];
		float* input_data = x.data_ptr();

		// gpu execution
		dim3 dim_grid(1, 1);
		dim3 dim_block(1, ceil(num_features/num_elements));
		BatchNorm2dKernel<<<>>>(input_data, num_features, num_elements, eps);

        return Tensor(x.sizes());
    }

    void printModule(const std::string &prefix) override
    {
        std::cout << prefix << ":BatchNorm2d" << std::endl;
    }
};
