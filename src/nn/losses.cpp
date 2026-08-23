// losses.cpp
#include "torc/nn/losses.hpp"
#include "torc/autograd.hpp"
#include "torc/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace torc::nn {

Variable MSELoss::forward(const Variable& input, const Variable& target) const {
    bool needs_grad = input.requires_grad_ || target.requires_grad_;
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
            input_grads[0] = grad;
            input_grads[1] = grad.mul(-1.0f);
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
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            sum_exp += std::exp(logits.data()[i * num_classes + j] - max_val);
        }
        for (int j = 0; j < num_classes; ++j) {
            out.data()[i * num_classes + j] = std::exp(logits.data()[i * num_classes + j] - max_val) / sum_exp;
        }
    }
    return out;
}

Variable CrossEntropyLoss::forward(const Variable& logits, const Variable& targets) const {
    bool needs_grad = logits.requires_grad_;
    int batch_size = logits.data().shape()[0];
    int num_classes = logits.data().shape()[1];

    Tensor softmax_data = row_softmax(logits.data(), batch_size, num_classes);
    Tensor log_softmax_data = softmax_data.log();

    Tensor loss_data(std::vector<int>{1});
    loss_data.data()[0] = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        int target_class = static_cast<int>(targets.data().data()[i]);
        loss_data.data()[0] -= log_softmax_data.data()[i * num_classes + target_class];
    }
    loss_data.data()[0] /= batch_size;

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
        };
        out.tape_.push_back(std::move(entry));
    }

    return out;
}

} // namespace torc::nn
