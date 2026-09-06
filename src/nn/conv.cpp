// conv.cpp
#include "torc/nn/conv.hpp"
#include "torc/autograd.hpp"
#include "torc/utils.hpp"
#include <cmath>

namespace torc::nn {

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size,
               int stride, int padding, float init_std, unsigned int seed)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {

    if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0)
        throw TorcError("Conv2d channels and kernel_size must be positive");
    if (stride < 1)
        throw TorcError("Conv2d stride must be >= 1");
    if (padding < 0)
        throw TorcError("Conv2d padding must be >= 0");

    Tensor weight_data(std::vector<int>{out_channels, in_channels, kernel_size, kernel_size});
    int fan_in = in_channels * kernel_size * kernel_size;
    if (init_std <= 0.0f) {
        torc::fill_kaiming_normal(weight_data.data(), weight_data.numel(), fan_in, seed);
    } else {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, init_std);
        for (int i = 0; i < weight_data.numel(); ++i) {
            weight_data.data()[i] = dist(rng);
        }
    }
    register_parameter("weight", Variable(std::move(weight_data), true));

    Tensor bias_data(std::vector<int>{out_channels});
    for (int i = 0; i < bias_data.numel(); ++i) {
        bias_data.data()[i] = 0.0f;
    }
    register_parameter("bias", Variable(std::move(bias_data), true));
}

Variable Conv2d::forward(const Variable& x) const {
    const Variable& weight = named_parameters().at("weight");
    const Variable& bias = named_parameters().at("bias");

    forward_cache_.emplace_back(torc::conv2d(x, weight, stride_, padding_));
    Variable& conv_result = forward_cache_.back();

    forward_cache_.emplace_back(torc::reshape(bias, std::vector<int>{1, out_channels_, 1, 1}));
    Variable& reshaped_bias = forward_cache_.back();

    forward_cache_.emplace_back(torc::add(conv_result, reshaped_bias));
    Variable& out = forward_cache_.back();

    return out;
}

Variable Flatten::forward(const Variable& x) const {
    const auto& shape = x.data().shape();
    if (shape.empty())
        throw TorcError("Flatten requires non-empty input");
    int batch = shape[0];
    int flat = 1;
    for (size_t i = 1; i < shape.size(); ++i)
        flat *= shape[i];
    return torc::reshape(x, std::vector<int>{batch, flat});
}

} // namespace torc::nn
