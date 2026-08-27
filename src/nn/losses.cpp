// losses.cpp
#include "torc/nn/losses.hpp"
#include "torc/autograd.hpp"
#include "torc/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <format>

namespace torc::nn {

Variable MSELoss::forward(const Variable& input, const Variable& target) const {
    bool needs_grad = (input.requires_grad_ || target.requires_grad_) && Variable::grad_enabled();
    Tensor diff_data = input.data_.sub(target.data_);
    Tensor squared_data = diff_data.mul(diff_data);
    float mean_val = squared_data.mean();
    Tensor loss_data(std::vector<int>{1});
    loss_data.data()[0] = mean_val;

    Variable out(std::move(loss_data), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&input), const_cast<Variable*>(&target) };

        Tensor input_data = input.data();
        Tensor target_data = target.data();

        entry.backward = [input_data, target_data](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            float n = static_cast<float>(input_data.numel());
            Tensor grad = input_data.sub(target_data);
            grad = grad.mul(2.0f / n);
            input_grads[0] = grad.mul(grad_output);
            input_grads[1] = grad.mul(-1.0f).mul(grad_output);
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

static Tensor row_softmax(const Tensor& logits, int batch_size, int num_classes) {
    Tensor out(logits.shape());
    for (int i = 0; i < batch_size; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < num_classes; ++j) {
            max_val = std::max(max_val, logits.data()[i * num_classes + j]);
        }
        double sum_exp = 0.0;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += std::exp(static_cast<double>(logits.data()[i * num_classes + j]) - max_val);
        }
        for (int j = 0; j < num_classes; ++j) {
            out.data()[i * num_classes + j] = static_cast<float>(
                std::exp(static_cast<double>(logits.data()[i * num_classes + j]) - max_val) / sum_exp);
        }
    }
    return out;
}

Variable CrossEntropyLoss::forward(const Variable& logits, const Variable& targets) const {
    const auto& logits_shape = logits.data().shape();
    const auto& targets_shape = targets.data().shape();
    if (logits_shape.size() != 2) {
        throw ShapeError(std::format("CrossEntropyLoss expects rank-2 logits, got shape {}",
                                     shape_to_string(logits_shape)));
    }
    if (targets_shape.size() != 1) {
        throw ShapeError(std::format("CrossEntropyLoss expects rank-1 targets, got shape {}",
                                     shape_to_string(targets_shape)));
    }

    bool needs_grad = logits.requires_grad_ && Variable::grad_enabled();
    int batch_size = logits_shape[0];
    int num_classes = logits_shape[1];
    if (batch_size <= 0 || num_classes <= 0) {
        throw ShapeError(std::format("CrossEntropyLoss requires non-empty batch and classes, got logits shape {}",
                                     shape_to_string(logits_shape)));
    }
    if (targets_shape[0] != batch_size) {
        throw ShapeError(std::format("CrossEntropyLoss target length {} does not match batch size {}",
                                     targets_shape[0], batch_size));
    }

    for (int i = 0; i < logits.data().numel(); ++i) {
        if (!std::isfinite(logits.data().data()[i])) {
            throw ShapeError("CrossEntropyLoss logits must be finite");
        }
    }
    for (int i = 0; i < batch_size; ++i) {
        float target_value = targets.data().data()[i];
        if (!std::isfinite(target_value) || std::floor(target_value) != target_value) {
            throw ShapeError(std::format("CrossEntropyLoss target at index {} must be an integer", i));
        }
        if (target_value < 0.0f || target_value >= static_cast<float>(num_classes)) {
            throw ShapeError(std::format("CrossEntropyLoss target {} at index {} is out of range [0, {})",
                                         target_value, i, num_classes));
        }
    }

    Tensor softmax_data = row_softmax(logits.data(), batch_size, num_classes);

    Tensor loss_data(std::vector<int>{1});
    double total_loss = 0.0;

    for (int i = 0; i < batch_size; ++i) {
        int target_class = static_cast<int>(targets.data().data()[i]);
        float max_val = logits.data().data()[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            max_val = std::max(max_val, logits.data().data()[i * num_classes + j]);
        }
        double sum_exp = 0.0;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += std::exp(static_cast<double>(logits.data().data()[i * num_classes + j]) - max_val);
        }
        double log_sum_exp = static_cast<double>(max_val) + std::log(sum_exp);
        total_loss += log_sum_exp - logits.data().data()[i * num_classes + target_class];
    }
    loss_data.data()[0] = static_cast<float>(total_loss / batch_size);

    Variable out(std::move(loss_data), needs_grad);

    if (needs_grad) {
        TapeEntry entry;
        entry.inputs = { const_cast<Variable*>(&logits) };

        Tensor softmax_copy = softmax_data;
        Tensor targets_copy = targets.data();

        entry.backward = [softmax_copy, targets_copy, batch_size, num_classes](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
            input_grads[0] = Tensor(softmax_copy.shape());
            for (int i = 0; i < batch_size; ++i) {
                int target_class = static_cast<int>(targets_copy.data()[i]);
                for (int j = 0; j < num_classes; ++j) {
                    float grad = softmax_copy.data()[i * num_classes + j];
                    if (j == target_class) {
                        grad -= 1.0f;
                    }
                    input_grads[0].data()[i * num_classes + j] = grad / batch_size;
                }
            }
            input_grads[0] = input_grads[0].mul(grad_output);
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

} // namespace torc::nn
