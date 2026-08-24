// optim.hpp
#pragma once
#include "torc/autograd.hpp"
#include <vector>

namespace torc::optim {

class SGD {
public:
    SGD(std::vector<torc::Variable*>& params, float lr, float momentum = 0.0f);
    void step();
    void zero_grad();

private:
    std::vector<torc::Variable*> params_;
    float lr_;
    float momentum_;
    std::vector<torc::Tensor> velocities_;
};

class Adam {
public:
    Adam(std::vector<torc::Variable*>& params, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step();
    void zero_grad();

private:
    std::vector<torc::Variable*> params_;
    float lr_;
    float beta1_;
    float beta2_;
    float eps_;
    int step_;
    std::vector<torc::Tensor> m_;
    std::vector<torc::Tensor> v_;
};

} // namespace torc::optim
