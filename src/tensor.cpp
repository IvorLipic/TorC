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

Tensor Tensor::operator-() const {
    Tensor out(shape_);
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = -storage_[i];
    return out;
}

bool Tensor::operator==(const Tensor& other) const {
    if (shape_ != other.shape_) return false;
    for (int i = 0; i < numel(); ++i)
        if (storage_[i] != other.storage_[i]) return false;
    return true;
}

float Tensor::sum() const {
    float total = 0.0f;
    for (int i = 0; i < numel(); ++i)
        total += storage_[i];
    return total;
}

float Tensor::mean() const {
    if (numel() == 0)
        throw ShapeError("Cannot compute mean of empty tensor");
    return sum() / numel();
}

float Tensor::max() const {
    if (numel() == 0)
        throw ShapeError("Cannot compute max of empty tensor");
    float m = storage_[0];
    for (int i = 1; i < numel(); ++i)
        if (storage_[i] > m) m = storage_[i];
    return m;
}

float Tensor::min() const {
    if (numel() == 0)
        throw ShapeError("Cannot compute min of empty tensor");
    float m = storage_[0];
    for (int i = 1; i < numel(); ++i)
        if (storage_[i] < m) m = storage_[i];
    return m;
}

Tensor Tensor::sum(int axis) const {
    if (axis < 0 || axis >= (int)shape_.size())
        throw ShapeError("Invalid axis " + std::to_string(axis) +
                         " for tensor with shape " + shape_to_string(shape_));
    if (numel() == 0)
        throw ShapeError("Cannot compute sum of empty tensor");

    std::vector<int> out_shape = shape_;
    out_shape.erase(out_shape.begin() + axis);
    Tensor out(out_shape);

    int outer_stride = shape_product(std::vector<int>(shape_.begin() + axis + 1, shape_.end()));
    int inner_stride = shape_product(std::vector<int>(shape_.begin(), shape_.begin() + axis));
    int axis_size = shape_[axis];

    for (int outer = 0; outer < inner_stride; ++outer) {
        for (int a = 0; a < axis_size; ++a) {
            for (int j = 0; j < outer_stride; ++j) {
                int in_idx = outer * axis_size * outer_stride + a * outer_stride + j;
                int out_idx = outer * outer_stride + j;
                out.storage_[out_idx] += storage_[in_idx];
            }
        }
    }
    return out;
}

Tensor Tensor::mean(int axis) const {
    Tensor s = sum(axis);
    for (int i = 0; i < s.numel(); ++i)
        s.storage_[i] /= shape_[axis];
    return s;
}

Tensor Tensor::max(int axis) const {
    if (axis < 0 || axis >= (int)shape_.size())
        throw ShapeError("Invalid axis " + std::to_string(axis) +
                         " for tensor with shape " + shape_to_string(shape_));
    if (numel() == 0)
        throw ShapeError("Cannot compute max of empty tensor");

    std::vector<int> out_shape = shape_;
    out_shape.erase(out_shape.begin() + axis);
    Tensor out(out_shape);

    int outer_stride = shape_product(std::vector<int>(shape_.begin() + axis + 1, shape_.end()));
    int inner_stride = shape_product(std::vector<int>(shape_.begin(), shape_.begin() + axis));
    int axis_size = shape_[axis];

    for (int outer = 0; outer < inner_stride; ++outer) {
        for (int j = 0; j < outer_stride; ++j) {
            int out_idx = outer * outer_stride + j;
            int first_in_idx = outer * axis_size * outer_stride + j;
            out.storage_[out_idx] = storage_[first_in_idx];
            for (int a = 1; a < axis_size; ++a) {
                int in_idx = outer * axis_size * outer_stride + a * outer_stride + j;
                if (storage_[in_idx] > out.storage_[out_idx])
                    out.storage_[out_idx] = storage_[in_idx];
            }
        }
    }
    return out;
}

Tensor Tensor::min(int axis) const {
    if (axis < 0 || axis >= (int)shape_.size())
        throw ShapeError("Invalid axis " + std::to_string(axis) +
                         " for tensor with shape " + shape_to_string(shape_));
    if (numel() == 0)
        throw ShapeError("Cannot compute min of empty tensor");

    std::vector<int> out_shape = shape_;
    out_shape.erase(out_shape.begin() + axis);
    Tensor out(out_shape);

    int outer_stride = shape_product(std::vector<int>(shape_.begin() + axis + 1, shape_.end()));
    int inner_stride = shape_product(std::vector<int>(shape_.begin(), shape_.begin() + axis));
    int axis_size = shape_[axis];

    for (int outer = 0; outer < inner_stride; ++outer) {
        for (int j = 0; j < outer_stride; ++j) {
            int out_idx = outer * outer_stride + j;
            int first_in_idx = outer * axis_size * outer_stride + j;
            out.storage_[out_idx] = storage_[first_in_idx];
            for (int a = 1; a < axis_size; ++a) {
                int in_idx = outer * axis_size * outer_stride + a * outer_stride + j;
                if (storage_[in_idx] < out.storage_[out_idx])
                    out.storage_[out_idx] = storage_[in_idx];
            }
        }
    }
    return out;
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
