// linear.cpp
#include "torc/nn/linear.hpp"
#include "torc/autograd.hpp"
#include "torc/utils.hpp"
#include <cmath>

namespace torc::nn {

Linear::Linear(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {
    
    if (in_features <= 0 || out_features <= 0) {
        throw TorcError("Linear features must be positive");
    }
    
    Tensor weight_data(std::vector<int>{out_features, in_features});
    for (int i = 0; i < weight_data.numel(); ++i) {
        weight_data.data()[i] = 0.01f;
    }
    register_parameter("weight", Variable(std::move(weight_data), true));
    
    Tensor bias_data(std::vector<int>{out_features});
    for (int i = 0; i < bias_data.numel(); ++i) {
        bias_data.data()[i] = 0.0f;
    }
    register_parameter("bias", Variable(std::move(bias_data), true));
}

Variable Linear::forward(const Variable& x) const {
    const Variable& weight = named_parameters().at("weight");
    const Variable& bias = named_parameters().at("bias");
    
    Variable matmul_result = torc::matmul(x, torc::transpose(weight));
    return torc::add(matmul_result, bias);
}

} // namespace torc::nn
