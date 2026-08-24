// optim.cpp
#include "torc/optim.hpp"

namespace torc::optim {

SGD::SGD(std::vector<torc::Variable*>& params, float lr, float momentum)
    : params_(params), lr_(lr), momentum_(momentum) {
    velocities_.reserve(params_.size());
    for (torc::Variable* param : params_) {
        velocities_.emplace_back(param->data().shape());
    }
}

void SGD::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        torc::Variable* param = params_[i];
        if (!param->has_grad()) continue;

        const torc::Tensor& grad = param->grad();
        if (momentum_ != 0.0f) {
            velocities_[i] = velocities_[i].mul(momentum_).add(grad);
            param->data() = param->data().sub(velocities_[i].mul(lr_));
        } else {
            param->data() = param->data().sub(grad.mul(lr_));
        }
    }
}

void SGD::zero_grad() {
    for (torc::Variable* param : params_) {
        param->zero_grad();
    }
}

} // namespace torc::optim
