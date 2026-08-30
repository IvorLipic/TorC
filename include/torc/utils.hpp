// utils.hpp
#pragma once
#include <vector>
#include <span>
#include <string>
#include <stdexcept>
#include <format>
#include <algorithm>
#include <numeric>
#include <random>

namespace torc {

class TorcError : public std::runtime_error {
public:
    explicit TorcError(const std::string& msg) : std::runtime_error(msg) {}
};

class ShapeError : public TorcError {
public:
    explicit ShapeError(const std::string& msg) : TorcError(msg) {}
};

class NumericalError : public TorcError {
public:
    explicit NumericalError(const std::string& msg) : TorcError(msg) {}
};

class IndexError : public TorcError {
public:
    explicit IndexError(const std::string& msg) : TorcError(msg) {}
};

inline int shape_product(std::span<const int> shape) {
    return std::ranges::fold_left(shape, 1, std::multiplies<>{});
}

inline std::string shape_to_string(std::span<const int> shape) {
    std::string result = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::format("{}", shape[i]);
    }
    result += ")";
    return result;
}

inline std::vector<int> broadcast_shape(std::span<const int> a, std::span<const int> b) {
    size_t max_rank = std::max(a.size(), b.size());
    std::vector<int> result;
    result.reserve(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
        int dim_a = (i < max_rank - a.size()) ? 1 : a[i - (max_rank - a.size())];
        int dim_b = (i < max_rank - b.size()) ? 1 : b[i - (max_rank - b.size())];
        if (dim_a == dim_b)      result.push_back(dim_a);
        else if (dim_a == 1)     result.push_back(dim_b);
        else if (dim_b == 1)     result.push_back(dim_a);
        else
            throw ShapeError(std::format(
                "Cannot broadcast shapes {} and {}", shape_to_string(a), shape_to_string(b)));
    }
    return result;
}

inline float kaiming_normal_std(int fan_in) {
    if (fan_in <= 0) throw TorcError("kaiming_normal_std: fan_in must be positive");
    return std::sqrt(2.0f / static_cast<float>(fan_in));
}

inline void fill_kaiming_normal(float* data, size_t count, int fan_in, unsigned int seed = std::random_device{}()) {
    std::mt19937 rng(seed);
    float stddev = kaiming_normal_std(fan_in);
    std::normal_distribution<float> dist(0.0f, stddev);
    for (size_t i = 0; i < count; ++i) data[i] = dist(rng);
}

} // namespace torc
