// autograd.hpp
#pragma once
#include "torc/tensor.hpp"
#include <vector>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

namespace torc {

struct Variable;

struct TapeEntry {
    std::vector<Variable*> inputs;
    std::function<void(const Tensor& grad_output, std::vector<Tensor>& input_grads)> backward;
};

Tensor reduce_sum_to_shape(const Tensor& grad, const std::vector<int>& target_shape);

class Variable {
public:
    Tensor data_;
    Tensor grad_;
    bool requires_grad_;
    bool has_grad_;
    std::vector<TapeEntry> tape_;

    static bool g_grad_enabled;

    explicit Variable(Tensor data, bool requires_grad)
        : data_(std::move(data)),
          grad_(std::vector<int>{}),
          requires_grad_(requires_grad),
          has_grad_(false) {}

    Variable(float scalar, bool requires_grad)
        : data_(Tensor(std::vector<int>{1})),
          grad_(std::vector<int>{}),
          requires_grad_(requires_grad),
          has_grad_(false) {
        data_.data()[0] = scalar;
    }

    [[nodiscard]] Tensor& data() { return data_; }
    [[nodiscard]] const Tensor& data() const { return data_; }
    [[nodiscard]] bool requires_grad() const { return requires_grad_; }
    [[nodiscard]] bool has_grad() const { return has_grad_; }
    [[nodiscard]] const Tensor& grad() const { return grad_; }

    [[nodiscard]] Variable detach() const {
        Variable out(data_, false);
        return out;
    }

    void fill(float val) {
        if (requires_grad_)
            throw TorcError("in-place operation forbidden on a Variable that requires grad");
        data_.fill(val);
    }

    [[nodiscard]] static bool grad_enabled() { return g_grad_enabled; }
    static void set_grad_enabled(bool enabled) { g_grad_enabled = enabled; }

    void backward() {
        if (!requires_grad_) return;

        if (data_.numel() != 1)
            throw ShapeError(std::format(
                "backward() requires scalar output, got shape {}",
                torc::shape_to_string(data_.shape())));

        Tensor ones({1.0f}, std::vector<int>{1});
        backward_with_grad(ones);
    }

    void backward(const Tensor& grad_output) {
        if (!requires_grad_) return;
        if (grad_output.numel() != data_.numel())
            throw ShapeError(std::format(
                "grad_output shape {} does not match output shape {}",
                torc::shape_to_string(grad_output.shape()),
                torc::shape_to_string(data_.shape())));
        backward_with_grad(grad_output);
    }

    void zero_grad() {
        has_grad_ = false;
        grad_ = Tensor(std::vector<int>{});
    }

private:
    void accumulate_grad(const Tensor& local_grad) {
        if (!has_grad_) {
            grad_ = Tensor(std::vector<int>(local_grad.shape().begin(), local_grad.shape().end()));
            for (int i = 0; i < local_grad.numel(); ++i)
                grad_.data()[i] = local_grad.data()[i];
            has_grad_ = true;
        } else {
            for (int i = 0; i < grad_.numel(); ++i)
                grad_.data()[i] += local_grad.data()[i];
        }
    }

    std::vector<Variable*> build_topo() const {
        std::vector<Variable*> topo;
        std::unordered_set<const Variable*> visited;
        
        std::function<void(const Variable*)> dfs = [&](const Variable* v) {
            if (visited.count(v)) return;
            visited.insert(v);
            for (const auto& entry : v->tape_) {
                for (Variable* input : entry.inputs) {
                    dfs(input);
                }
            }
            topo.push_back(const_cast<Variable*>(v));
        };
        
        dfs(this);
        return topo;
    }

    void backward_with_grad(const Tensor& upstream_grad) {
        if (tape_.empty()) return;

        std::vector<Variable*> topo = build_topo();
        std::reverse(topo.begin(), topo.end());

        std::unordered_map<Variable*, Tensor> grad_map;
        grad_map.emplace(this, upstream_grad);

        for (Variable* v : topo) {
            if (v->tape_.empty()) continue;
            
            auto it = grad_map.find(v);
            if (it == grad_map.end()) continue;
            
            const Tensor& grad = it->second;
            Tensor current_grad = grad;
            
            for (auto eit = v->tape_.rbegin(); eit != v->tape_.rend(); ++eit) {
                const TapeEntry& entry = *eit;
                std::vector<Tensor> input_grads;
                input_grads.reserve(entry.inputs.size());
                for (size_t i = 0; i < entry.inputs.size(); ++i) {
                    input_grads.push_back(Tensor(std::vector<int>{1}));
                }
                entry.backward(current_grad, input_grads);
                
                for (size_t i = 0; i < entry.inputs.size(); ++i) {
                    Variable* input = entry.inputs[i];
                    if (input->requires_grad_) {
                        Tensor input_grad = input_grads[i];
                        if (input_grad.shape() != input->data_.shape())
                            input_grad = reduce_sum_to_shape(input_grad, input->data_.shape());
                        input->accumulate_grad(input_grad);
                        auto insert_result = grad_map.emplace(input, input_grad);
                        if (!insert_result.second) {
                            insert_result.first->second = input_grad;
                        }
                    }
                }
                
                current_grad = input_grads.back();
            }
        }

        tape_.clear();
    }
};

} // namespace torc

namespace torc {

Variable add_scalar(const Variable& a, const Variable& b);
Variable sub_scalar(const Variable& a, const Variable& b);
Variable mul_scalar(const Variable& a, const Variable& b);
Variable div_scalar(const Variable& a, const Variable& b);
Variable neg_scalar(const Variable& a);

Tensor reduce_sum_to_shape(const Tensor& grad, const std::vector<int>& target_shape);

Variable add(const Variable& a, const Variable& b);
Variable sub(const Variable& a, const Variable& b);
Variable mul(const Variable& a, const Variable& b);
Variable div(const Variable& a, const Variable& b);
Variable neg(const Variable& a);
Variable sum(const Variable& a);
Variable sum(const Variable& a, int axis);
Variable mean(const Variable& a);
Variable mean(const Variable& a, int axis);
Variable max(const Variable& a);
Variable min(const Variable& a);
Variable max(const Variable& a, int axis);
Variable min(const Variable& a, int axis);
Variable matmul(const Variable& a, const Variable& b);
Variable transpose(const Variable& a);
Variable transpose(const Variable& a, std::vector<int> axes);
Variable reshape(const Variable& a, std::vector<int> new_shape);
Variable slice(const Variable& a, std::vector<std::pair<int, int>> slices);

} // namespace torc
