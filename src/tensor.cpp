#include "torc/tensor.hpp"
#include <stdexcept>

namespace torc {

Tensor::Tensor(std::vector<int> shape) : shape_(shape) {
    int total = 1;
    for (int s : shape) total *= s;
    storage_.resize(total, 0.0f);
}

Tensor::Tensor(std::initializer_list<float> data, std::vector<int> shape)
    : shape_(shape), storage_(data) {
    int total = 1;
    for (int s : shape) total *= s;
    if (total != (int)storage_.size()) {
        throw std::runtime_error("Shape mismatch with data size");
    }
}

float* Tensor::data() { return storage_.data(); }
const float* Tensor::data() const { return storage_.data(); }

int Tensor::numel() const {
    int total = 1;
    for (int s : shape_) total *= s;
    return total;
}

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] + other.storage_[i];
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] * other.storage_[i];
    return out;
}

}
