// autograd.cpp
#include "torc/autograd.hpp"
#include <algorithm>

namespace torc {

Variable add_scalar(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    float av = a.data_.data()[0];
    float bv = b.data_.data()[0];
    Variable out(av + bv, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = grad_output;  // dz/da = 1
            input_grads[1] = grad_output;  // dz/db = 1
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable sub_scalar(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    float av = a.data_.data()[0];
    float bv = b.data_.data()[0];
    Variable out(av - bv, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = grad_output;  // dz/da = 1
            input_grads[1] = Tensor(std::vector<int>{1});
            input_grads[1].data()[0] = -grad_output.data()[0];  // dz/db = -1
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable mul_scalar(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    float av = a.data_.data()[0];
    float bv = b.data_.data()[0];
    Variable out(av * bv, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        entry.backward = [av, bv](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = Tensor(std::vector<int>{1});
            input_grads[0].data()[0] = grad_output.data()[0] * bv;  // dz/da = b
            input_grads[1] = Tensor(std::vector<int>{1});
            input_grads[1].data()[0] = grad_output.data()[0] * av;  // dz/db = a
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable div_scalar(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    float av = a.data_.data()[0];
    float bv = b.data_.data()[0];
    Variable out(av / bv, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        entry.backward = [av, bv](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = Tensor(std::vector<int>{1});
            input_grads[0].data()[0] = grad_output.data()[0] / bv;  // dz/da = 1/b
            input_grads[1] = Tensor(std::vector<int>{1});
            input_grads[1].data()[0] = -grad_output.data()[0] * av / (bv * bv);  // dz/db = -a/b^2
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable neg_scalar(const Variable& a) {
    bool needs_grad = a.requires_grad_;
    float av = a.data_.data()[0];
    Variable out(-av, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = Tensor(std::vector<int>{1});
            input_grads[0].data()[0] = -grad_output.data()[0];  // dz/da = -1
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

} // namespace torc
