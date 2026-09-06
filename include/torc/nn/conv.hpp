// conv.hpp
#pragma once
#include "torc/nn.hpp"
#include <vector>
#include <random>

namespace torc::nn {

class Conv2d : public Module {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size,
           int stride = 1, int padding = 0,
           float init_std = 0.0f, unsigned int seed = std::random_device{}());

    Variable forward(const Variable& x) const override;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
};

class Flatten : public Module {
public:
    Flatten() = default;
    Variable forward(const Variable& x) const override;
};

} // namespace torc::nn
