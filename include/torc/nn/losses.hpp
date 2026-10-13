// losses.hpp
#pragma once
#include "torc/autograd.hpp"

namespace torc::nn {

// Mean squared error: mean((input - target)^2).
//
// `input` and `target` may have different shapes as long as they broadcast (NumPy rules). The
// loss is the mean over the *broadcast* result, i.e. it divides by the number of elements of
// the broadcast shape, not by input.numel(). For example an input of shape (N, 1) against a
// target of shape (N) is compared as an (N, N) matrix and divided by N * N. Gradients are
// taken with respect to that same loss: each operand receives 2 * (input - target) / N_broadcast
// (negated for the target), summed over the broadcast dimensions back to its own shape.
// When the shapes are equal this is the ordinary mean over all elements.
class MSELoss {
public:
    MSELoss() = default;
    Variable forward(const Variable& input, const Variable& target) const;
    Variable operator()(const Variable& input, const Variable& target) const {
        return forward(input, target);
    }
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default;
    Variable forward(const Variable& logits, const Variable& targets) const;
    Variable operator()(const Variable& logits, const Variable& targets) const {
        return forward(logits, targets);
    }
};

} // namespace torc::nn
