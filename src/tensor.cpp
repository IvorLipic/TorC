#include "torc/tensor.hpp"
#include "torc/utils.hpp"
#include <ostream>

namespace torc {

Tensor::Tensor(std::vector<int> shape) : shape_(shape) {
    storage_.resize(shape_product(shape_), 0.0f);
}

Tensor::Tensor(std::initializer_list<float> data, std::vector<int> shape)
    : shape_(shape), storage_(data) {
    if (shape_product(shape_) != (int)storage_.size())
        throw ShapeError("Initializer list size " + std::to_string(storage_.size()) +
                         " does not match shape product " + std::to_string(shape_product(shape_)));
}

float* Tensor::data() { return storage_.data(); }
const float* Tensor::data() const { return storage_.data(); }

int Tensor::numel() const {
    return shape_product(shape_);
}

void Tensor::check_same_shape(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw ShapeError("Shape mismatch: " + shape_to_string(shape_) +
                         " vs " + shape_to_string(other.shape_));
}

Tensor Tensor::add(const Tensor& other) const {
    check_same_shape(other);
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] + other.storage_[i];
    return out;
}

Tensor Tensor::sub(const Tensor& other) const {
    check_same_shape(other);
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] - other.storage_[i];
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    check_same_shape(other);
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] * other.storage_[i];
    return out;
}

Tensor Tensor::div(const Tensor& other) const {
    check_same_shape(other);
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] / other.storage_[i];
    return out;
}

Tensor Tensor::add(float scalar) const {
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] + scalar;
    return out;
}

Tensor Tensor::sub(float scalar) const {
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] - scalar;
    return out;
}

Tensor Tensor::mul(float scalar) const {
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] * scalar;
    return out;
}

Tensor Tensor::div(float scalar) const {
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = storage_[i] / scalar;
    return out;
}

bool Tensor::operator==(const Tensor& other) const {
    if (shape_ != other.shape_) return false;
    for (int i = 0; i < numel(); ++i)
        if (storage_[i] != other.storage_[i]) return false;
    return true;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape=" << shape_to_string(t.shape()) << ", data=[";
    for (int i = 0; i < t.numel(); ++i) {
        if (i > 0) os << ", ";
        os << t.data()[i];
    }
    os << "])";
    return os;
}

}
