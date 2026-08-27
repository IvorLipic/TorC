// activations.hpp
#pragma once
#include "torc/nn.hpp"

namespace torc::nn {

class ReLU : public Module {
public:
    ReLU() = default;
    Variable forward(const Variable& x) const override;
};

class Sigmoid : public Module {
public:
    Sigmoid() = default;
    Variable forward(const Variable& x) const override;
};

class Softmax : public Module {
public:
    explicit Softmax(int axis = -1) : axis_(axis) {}
    Variable forward(const Variable& x) const override;

private:
    int axis_;
};

} // namespace torc::nn
