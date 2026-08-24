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

} // namespace torc::optim
