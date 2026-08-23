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

Tensor expand_grad_along_axis(const Tensor& grad_output, int axis, int axis_size, const std::vector<int>& input_shape) {
    Tensor result(input_shape);
    int outer_stride = shape_product(std::span<const int>(input_shape).subspan(axis + 1));
    int inner_stride = shape_product(std::span<const int>(input_shape).subspan(0, axis));
    for (int outer = 0; outer < inner_stride; ++outer) {
        for (int j = 0; j < outer_stride; ++j) {
            int out_idx = outer * outer_stride + j;
            float val = grad_output.data()[out_idx];
            int base = outer * axis_size * outer_stride + j;
            for (int a = 0; a < axis_size; ++a)
                result.data()[base + a * outer_stride] = val;
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

Variable sum(const Variable& a) {
    bool needs_grad = a.requires_grad_;
    float out_val = a.data_.sum();
    Variable out(out_val, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        Tensor a_data = a.data();
        entry.backward = [a_data](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            float g = grad_output.data()[0];
            Tensor result(a_data.shape());
            for (int i = 0; i < result.numel(); ++i)
                result.data()[i] = g;
            input_grads[0] = result;
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable sum(const Variable& a, int axis) {
    bool needs_grad = a.requires_grad_;
    Tensor out_data = a.data_.sum(axis);
    Variable out(std::move(out_data), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        Tensor a_data = a.data();
        int axis_size = a.data_.shape()[axis];
        entry.backward = [a_data, axis, axis_size](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = expand_grad_along_axis(grad_output, axis, axis_size, a_data.shape());
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable mean(const Variable& a) {
    bool needs_grad = a.requires_grad_;
    float out_val = a.data_.mean();
    Variable out(out_val, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        Tensor a_data = a.data();
        entry.backward = [a_data](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            float g = grad_output.data()[0] / a_data.numel();
            Tensor result(a_data.shape());
            for (int i = 0; i < result.numel(); ++i)
                result.data()[i] = g;
            input_grads[0] = result;
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable mean(const Variable& a, int axis) {
    bool needs_grad = a.requires_grad_;
    Tensor out_data = a.data_.mean(axis);
    Variable out(std::move(out_data), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        Tensor a_data = a.data();
        int axis_size = a.data_.shape()[axis];
        entry.backward = [a_data, axis, axis_size](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            Tensor scaled_grad(grad_output.shape());
            float scale = 1.0f / axis_size;
            for (int i = 0; i < grad_output.numel(); ++i)
                scaled_grad.data()[i] = grad_output.data()[i] * scale;
            input_grads[0] = expand_grad_along_axis(scaled_grad, axis, axis_size, a_data.shape());
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable max(const Variable& a) {
    bool needs_grad = a.requires_grad_;
    float out_val = a.data_.max();
    Variable out(out_val, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            throw ShapeError("max backward not yet implemented: argmax tracking is required");
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Variable min(const Variable& a) {
    bool needs_grad = a.requires_grad_;
    float out_val = a.data_.min();
    Variable out(out_val, needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a) };
        entry.backward = [](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            throw ShapeError("min backward not yet implemented: argmax tracking is required");
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

Tensor swap_last_two_axes(const Tensor& t) {
    int r = (int)t.shape().size();
    if (r < 2) return t;
    std::vector<int> axes(r);
    for (int i = 0; i < r; ++i) axes[i] = i;
    std::swap(axes[r - 2], axes[r - 1]);
    return t.transpose(std::move(axes));
}

Variable matmul(const Variable& a, const Variable& b) {
    bool needs_grad = a.requires_grad_ || b.requires_grad_;
    Tensor out_data = a.data_.matmul(b.data_);
    Variable out(std::move(out_data), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&a), const_cast<Variable*>(&b) };
        Tensor a_data = a.data();
        Tensor b_data = b.data();
        entry.backward = [a_data, b_data](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            Tensor bt = swap_last_two_axes(b_data);
            Tensor da_raw = grad_output.matmul(bt);
            input_grads[0] = reduce_sum_to_shape(da_raw, a_data.shape());

            Tensor at = swap_last_two_axes(a_data);
            Tensor db_raw = at.matmul(grad_output);
            input_grads[1] = reduce_sum_to_shape(db_raw, b_data.shape());
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

} // namespace torc
