// autograd.cpp
#include "torc/autograd.hpp"
#include <algorithm>

namespace torc {

Tensor reduce_sum_to_shape(const Tensor& grad, const std::vector<int>& target_shape) {
    Tensor result = grad;
    int g_rank = (int)result.shape().size();
    int t_rank = (int)target_shape.size();
    int offset = g_rank - t_rank;
    for (int i = 0; i < offset; ++i)
        result = result.sum(0);
    for (int i = 0; i < t_rank; ++i) {
        if (target_shape[i] == 1 && result.shape()[i] > 1) {
            result = result.sum(i);
            std::vector<int> new_shape = result.shape();
            new_shape.insert(new_shape.begin() + i, 1);
            result = result.reshape(new_shape);
        }
    }
    return result;
}

Variable add(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    Tensor out = a.data_.add(b.data_);
    Variable vout(std::move(out), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = grad_output;
            input_grads[1] = grad_output;
        };
        vout.tape_.push_back(std::move(entry));
    }

    return vout;
}

Variable sub(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    Tensor out = a.data_.sub(b.data_);
    Variable vout(std::move(out), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = grad_output;
            input_grads[1] = grad_output.mul(-1.0f);
        };
        vout.tape_.push_back(std::move(entry));
    }

    return vout;
}

Variable mul(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    Tensor out = a.data_.mul(b.data_);
    Variable vout(std::move(out), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        Tensor a_data = a.data();
        Tensor b_data = b.data();
        entry.backward = [a_data, b_data](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = grad_output.mul(b_data);
            input_grads[1] = grad_output.mul(a_data);
        };
        vout.tape_.push_back(std::move(entry));
    }

    return vout;
}

Variable div(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    Tensor out = a.data_.div(b.data_);
    Variable vout(std::move(out), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        Tensor a_data = a.data();
        Tensor b_data = b.data();
        entry.backward = [a_data, b_data](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = grad_output.div(b_data);
            Tensor b_sq = b_data.mul(b_data);
            input_grads[1] = grad_output.mul(a_data).mul(-1.0f).div(b_sq);
        };
        vout.tape_.push_back(std::move(entry));
    }

    return vout;
}

Variable neg(const Variable& a) {
    bool needs_grad = a.requires_grad_;
    Tensor out = a.data_.operator-();
    Variable vout(std::move(out), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = grad_output.mul(-1.0f);
        };
        vout.tape_.push_back(std::move(entry));
    }

    return vout;
}

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
