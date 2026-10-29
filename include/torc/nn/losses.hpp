// losses.hpp
#pragma once
#include "torc/autograd.hpp"

namespace torc::nn {

// Mean squared error: mean((input - target)^2) over all elements.
//
// `input` and `target` must have exactly the same shape; otherwise forward() throws ShapeError.
// MSELoss does not broadcast: a prediction of shape (N, 1) against targets of shape (N) would
// otherwise be compared as an (N, N) matrix, a silent and common mistake. Reshape one operand
// first if the shapes differ only by a size-1 dimension. The gradient is 2 * (input - target) / n
// to `input` and its negation to `target`, where n is the element count.
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
