#pragma once
#include <vector>
#include <string>

namespace torc {

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

} // namespace torc
