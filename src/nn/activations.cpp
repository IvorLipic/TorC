// activations.cpp
#include "torc/nn/activations.hpp"
#include "torc/autograd.hpp"

namespace torc::nn {

Variable ReLU::forward(const Variable& x) const {
    return torc::relu(x);
}

Variable Sigmoid::forward(const Variable& x) const {
    return torc::sigmoid(x);
}

Variable Softmax::forward(const Variable& x) const {
    return torc::softmax(x);
}

} // namespace torc::nn
