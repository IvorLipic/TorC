// losses.hpp
#pragma once
#include "torc/autograd.hpp"

namespace torc::nn {

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
