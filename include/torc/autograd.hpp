// autograd.hpp
#pragma once
#include "torc/tensor.hpp"
#include <vector>
#include <initializer_list>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <memory>

namespace torc {

struct Variable;
struct VariableLifetime {};

struct TapeEntry {
    std::vector<Variable*> inputs;
    std::vector<std::weak_ptr<VariableLifetime>> input_lifetimes;
    std::vector<bool> input_requires_grad;
    std::function<void(const Tensor& grad_output, std::vector<Tensor>& input_grads)> backward;

    void set_inputs(std::initializer_list<Variable*> values);
};

Tensor reduce_sum_to_shape(const Tensor& grad, const std::vector<int>& target_shape);

class Variable {
public:
    Tensor data_;
    Tensor grad_;
    bool requires_grad_;
    bool has_grad_;
    std::vector<TapeEntry> tape_;

    static thread_local bool g_grad_enabled;

    explicit Variable(Tensor data, bool requires_grad)
        : data_(std::move(data)),
          grad_(std::vector<int>{}),
          requires_grad_(requires_grad),
          has_grad_(false),
          lifetime_(std::make_shared<VariableLifetime>()) {}

    Variable(float scalar, bool requires_grad)
        : data_(Tensor(std::vector<int>{1})),
          grad_(std::vector<int>{}),
          requires_grad_(requires_grad),
          has_grad_(false),
          lifetime_(std::make_shared<VariableLifetime>()) {
        data_.data()[0] = scalar;
    }

    Variable(const Variable& other)
        : data_(other.data_),
          grad_(other.grad_),
          requires_grad_(other.requires_grad_),
          has_grad_(other.has_grad_),
          tape_(other.tape_),
          lifetime_(std::make_shared<VariableLifetime>()) {}

    Variable(Variable&& other) noexcept
        : data_(std::move(other.data_)),
          grad_(std::move(other.grad_)),
          requires_grad_(other.requires_grad_),
          has_grad_(other.has_grad_),
          tape_(std::move(other.tape_)),
          lifetime_(std::make_shared<VariableLifetime>()) {
        other.lifetime_.reset();
    }

    Variable& operator=(const Variable& other) {
        if (this != &other) {
            data_ = other.data_;
            grad_ = other.grad_;
            requires_grad_ = other.requires_grad_;
            has_grad_ = other.has_grad_;
            tape_ = other.tape_;
        }
        return *this;
    }

    Variable& operator=(Variable&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            grad_ = std::move(other.grad_);
            requires_grad_ = other.requires_grad_;
            has_grad_ = other.has_grad_;
            tape_ = std::move(other.tape_);
            // A moved-from Variable must not remain a valid target for existing
            // tape edges.  Resetting its token makes such edges fail safely.
            other.lifetime_.reset();
        }
        return *this;
    }

    [[nodiscard]] Tensor& data() { return data_; }
    [[nodiscard]] const Tensor& data() const { return data_; }
    [[nodiscard]] bool requires_grad() const { return requires_grad_; }
    [[nodiscard]] bool has_grad() const { return has_grad_; }
    [[nodiscard]] const Tensor& grad() const { return grad_; }
    [[nodiscard]] std::weak_ptr<VariableLifetime> lifetime_token() const { return lifetime_; }

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

    class NoGradGuard {
    public:
        NoGradGuard() : previous_(g_grad_enabled) { g_grad_enabled = false; }
        ~NoGradGuard() { g_grad_enabled = previous_; }
        NoGradGuard(const NoGradGuard&) = delete;
        NoGradGuard& operator=(const NoGradGuard&) = delete;
    private:
        bool previous_;
    };

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
    std::shared_ptr<VariableLifetime> lifetime_;

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
                for (size_t i = 0; i < entry.inputs.size(); ++i) {
                    if (entry.input_requires_grad.size() != entry.inputs.size()) {
                        throw TorcError("autograd tape entry lacks input lifetime metadata");
                    }
                    if (!entry.input_requires_grad[i]) continue;
                    if (entry.input_lifetimes.size() != entry.inputs.size() ||
                        entry.input_lifetimes[i].expired()) {
                        throw TorcError("autograd graph input expired before backward(); keep all graph Variables alive");
                    }
                    Variable* input = entry.inputs[i];
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
                    if (entry.input_requires_grad.size() != entry.inputs.size()) {
                        throw TorcError("autograd tape entry lacks input lifetime metadata");
                    }
                    if (!entry.input_requires_grad[i]) continue;
                    if (entry.input_lifetimes.size() != entry.inputs.size() ||
                        entry.input_lifetimes[i].expired()) {
                        throw TorcError("autograd graph input expired before backward(); keep all graph Variables alive");
                    }
                    Variable* input = entry.inputs[i];
                    Tensor input_grad = input_grads[i];
                    if (input_grad.shape() != input->data_.shape())
                        input_grad = reduce_sum_to_shape(input_grad, input->data_.shape());
                    input->accumulate_grad(input_grad);
                    auto insert_result = grad_map.emplace(input, input_grad);
                    if (!insert_result.second) {
                        insert_result.first->second = insert_result.first->second.add(input_grad);
                    }
                }
                
            }
        }

        tape_.clear();
    }
};

inline void TapeEntry::set_inputs(std::initializer_list<Variable*> values) {
    inputs.assign(values.begin(), values.end());
    input_lifetimes.clear();
    input_lifetimes.reserve(inputs.size());
    input_requires_grad.clear();
    input_requires_grad.reserve(inputs.size());
    for (Variable* input : inputs) {
        input_lifetimes.push_back(input->lifetime_token());
        input_requires_grad.push_back(input->requires_grad());
    }
}

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
Variable exp(const Variable& a);
Variable relu(const Variable& a);
Variable sigmoid(const Variable& a);
Variable softmax(const Variable& a);
Variable softmax(const Variable& a, int axis);
Variable log(const Variable& a);
Variable sqrt(const Variable& a);

} // namespace torc
