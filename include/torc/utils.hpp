#pragma once
#include <vector>
#include <string>
#include <stdexcept>

namespace torc {

class TorcError : public std::runtime_error {
public:
    explicit TorcError(const std::string& msg) : std::runtime_error(msg) {}
};

class ShapeError : public TorcError {
public:
    explicit ShapeError(const std::string& msg) : TorcError(msg) {}
};

inline int shape_product(const std::vector<int>& shape) {
    int total = 1;
    for (int s : shape) total *= s;
    return total;
}

inline std::string shape_to_string(const std::vector<int>& shape) {
    std::string result = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += ")";
    return result;
}

inline std::vector<int> broadcast_shape(const std::vector<int>& a, const std::vector<int>& b) {
    size_t rank_a = a.size();
    size_t rank_b = b.size();
    size_t max_rank = std::max(rank_a, rank_b);
    std::vector<int> result;
    result.reserve(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
        int dim_a = (i < max_rank - rank_a) ? 1 : a[i - (max_rank - rank_a)];
        int dim_b = (i < max_rank - rank_b) ? 1 : b[i - (max_rank - rank_b)];
        if (dim_a == dim_b) {
            result.push_back(dim_a);
        } else if (dim_a == 1) {
            result.push_back(dim_b);
        } else if (dim_b == 1) {
            result.push_back(dim_a);
        } else {
            throw ShapeError("Cannot broadcast shapes " + shape_to_string(a) + " and " + shape_to_string(b));
        }
    }
    return result;
}

} // namespace torc
