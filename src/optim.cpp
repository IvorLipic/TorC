// optim.cpp
#include "torc/optim.hpp"
#include <cmath>

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

Adam::Adam(std::vector<torc::Variable*>& params, float lr, float beta1, float beta2, float eps)
    : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), step_(1) {
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (torc::Variable* param : params_) {
        m_.emplace_back(param->data().shape());
        v_.emplace_back(param->data().shape());
    }
}

void Adam::step() {
    float beta1 = beta1_;
    float beta2 = beta2_;
    float lr = lr_;
    float eps = eps_;
    int t = step_++;

    for (size_t i = 0; i < params_.size(); ++i) {
        torc::Variable* param = params_[i];
        if (!param->has_grad()) continue;

        const torc::Tensor& grad = param->grad();
        m_[i] = m_[i].mul(beta1).add(grad.mul(1.0f - beta1));
        v_[i] = v_[i].mul(beta2).add(grad.mul(grad).mul(1.0f - beta2));

        float bias_correction1 = 1.0f - std::pow(beta1, t);
        float bias_correction2 = 1.0f - std::pow(beta2, t);

        torc::Tensor m_hat = m_[i].div(bias_correction1);
        torc::Tensor v_hat = v_[i].div(bias_correction2);
        torc::Tensor update = m_hat.div(v_hat.sqrt().add(eps));

        param->data() = param->data().sub(update.mul(lr));
    }
}

void Adam::zero_grad() {
    for (torc::Variable* param : params_) {
        param->zero_grad();
    }
}

AdamW::AdamW(std::vector<torc::Variable*>& params, float lr, float weight_decay, float beta1, float beta2, float eps)
    : params_(params), lr_(lr), weight_decay_(weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps), step_(1) {
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (torc::Variable* param : params_) {
        m_.emplace_back(param->data().shape());
        v_.emplace_back(param->data().shape());
    }
}

void AdamW::step() {
    float beta1 = beta1_;
    float beta2 = beta2_;
    float lr = lr_;
    float eps = eps_;
    float wd = weight_decay_;
    int t = step_++;

    for (size_t i = 0; i < params_.size(); ++i) {
        torc::Variable* param = params_[i];
        if (!param->has_grad()) continue;

        const torc::Tensor& grad = param->grad();
        m_[i] = m_[i].mul(beta1).add(grad.mul(1.0f - beta1));
        v_[i] = v_[i].mul(beta2).add(grad.mul(grad).mul(1.0f - beta2));

        float bias_correction1 = 1.0f - std::pow(beta1, t);
        float bias_correction2 = 1.0f - std::pow(beta2, t);

        torc::Tensor m_hat = m_[i].div(bias_correction1);
        torc::Tensor v_hat = v_[i].div(bias_correction2);
        torc::Tensor update = m_hat.div(v_hat.sqrt().add(eps));

        param->data() = param->data().sub(update.mul(lr)).sub(param->data().mul(lr * wd));
    }
}

void AdamW::zero_grad() {
    for (torc::Variable* param : params_) {
        param->zero_grad();
    }
}

} // namespace torc::optim
