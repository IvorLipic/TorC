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

int Tensor::flat_index(const std::vector<int>& indices) const {
    int idx = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        idx = idx * shape_[i] + indices[i];
    }
    return idx;
}

Tensor Tensor::add(const Tensor& other) const {
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor out(out_shape);
    for (int i = 0; i < out.numel(); ++i) {
        std::vector<int> out_indices(out_shape.size());
        int tmp = i;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            out_indices[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int a_rank = shape_.size();
        int b_rank = other.shape_.size();
        int a_offset = (int)out_shape.size() - a_rank;
        int b_offset = (int)out_shape.size() - b_rank;
        int a_idx = 0;
        for (int d = 0; d < a_rank; ++d) {
            a_idx = a_idx * shape_[d] + (shape_[d] == 1 ? 0 : out_indices[d + a_offset]);
        }
        int b_idx = 0;
        for (int d = 0; d < b_rank; ++d) {
            b_idx = b_idx * other.shape_[d] + (other.shape_[d] == 1 ? 0 : out_indices[d + b_offset]);
        }
        out.storage_[i] = storage_[a_idx] + other.storage_[b_idx];
    }
    return out;
}

Tensor Tensor::sub(const Tensor& other) const {
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor out(out_shape);
    for (int i = 0; i < out.numel(); ++i) {
        std::vector<int> out_indices(out_shape.size());
        int tmp = i;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            out_indices[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int a_rank = shape_.size();
        int b_rank = other.shape_.size();
        int a_offset = (int)out_shape.size() - a_rank;
        int b_offset = (int)out_shape.size() - b_rank;
        int a_idx = 0;
        for (int d = 0; d < a_rank; ++d) {
            a_idx = a_idx * shape_[d] + (shape_[d] == 1 ? 0 : out_indices[d + a_offset]);
        }
        int b_idx = 0;
        for (int d = 0; d < b_rank; ++d) {
            b_idx = b_idx * other.shape_[d] + (other.shape_[d] == 1 ? 0 : out_indices[d + b_offset]);
        }
        out.storage_[i] = storage_[a_idx] - other.storage_[b_idx];
    }
    return out;
}

Tensor Tensor::mul(const Tensor& other) const {
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor out(out_shape);
    for (int i = 0; i < out.numel(); ++i) {
        std::vector<int> out_indices(out_shape.size());
        int tmp = i;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            out_indices[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int a_rank = shape_.size();
        int b_rank = other.shape_.size();
        int a_offset = (int)out_shape.size() - a_rank;
        int b_offset = (int)out_shape.size() - b_rank;
        int a_idx = 0;
        for (int d = 0; d < a_rank; ++d) {
            a_idx = a_idx * shape_[d] + (shape_[d] == 1 ? 0 : out_indices[d + a_offset]);
        }
        int b_idx = 0;
        for (int d = 0; d < b_rank; ++d) {
            b_idx = b_idx * other.shape_[d] + (other.shape_[d] == 1 ? 0 : out_indices[d + b_offset]);
        }
        out.storage_[i] = storage_[a_idx] * other.storage_[b_idx];
    }
    return out;
}

Tensor Tensor::div(const Tensor& other) const {
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor out(out_shape);
    for (int i = 0; i < out.numel(); ++i) {
        std::vector<int> out_indices(out_shape.size());
        int tmp = i;
        for (int d = (int)out_shape.size() - 1; d >= 0; --d) {
            out_indices[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int a_rank = shape_.size();
        int b_rank = other.shape_.size();
        int a_offset = (int)out_shape.size() - a_rank;
        int b_offset = (int)out_shape.size() - b_rank;
        int a_idx = 0;
        for (int d = 0; d < a_rank; ++d) {
            a_idx = a_idx * shape_[d] + (shape_[d] == 1 ? 0 : out_indices[d + a_offset]);
        }
        int b_idx = 0;
        for (int d = 0; d < b_rank; ++d) {
            b_idx = b_idx * other.shape_[d] + (other.shape_[d] == 1 ? 0 : out_indices[d + b_offset]);
        }
        out.storage_[i] = storage_[a_idx] / other.storage_[b_idx];
    }
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

Tensor Tensor::transpose(std::vector<int> axes) const {
    int rank = shape_.size();
    if (axes.empty()) {
        axes.resize(rank);
        for (int i = 0; i < rank; ++i) axes[i] = rank - 1 - i;
    }
    if ((int)axes.size() != rank)
        throw ShapeError("Transpose axes size " + std::to_string(axes.size()) + " does not match rank " + std::to_string(rank));
    std::vector<bool> seen(rank, false);
    for (int i = 0; i < rank; ++i) {
        if (axes[i] < 0 || axes[i] >= rank)
            throw ShapeError("Transpose axis " + std::to_string(axes[i]) + " out of bounds for rank " + std::to_string(rank));
        if (seen[axes[i]])
            throw ShapeError("Duplicate transpose axis " + std::to_string(axes[i]));
        seen[axes[i]] = true;
    }

    std::vector<int> new_shape(rank);
    for (int i = 0; i < rank; ++i) new_shape[i] = shape_[axes[i]];
    Tensor out(std::move(new_shape));

    std::vector<int> out_indices(rank, 0);
    for (int out_flat = 0; out_flat < out.numel(); ++out_flat) {
        std::vector<int> in_indices(rank);
        for (int i = 0; i < rank; ++i) in_indices[axes[i]] = out_indices[i];
        out.storage_[out_flat] = storage_[flat_index(in_indices)];
        for (int i = rank - 1; i >= 0; --i) {
            ++out_indices[i];
            if (out_indices[i] < out.shape()[i]) break;
            out_indices[i] = 0;
        }
    }
    return out;
}

Tensor Tensor::slice(const std::vector<Slice>& slices) const {
    int rank = shape_.size();
    if ((int)slices.size() != rank)
        throw ShapeError("Slice count " + std::to_string(slices.size()) + " does not match rank " + std::to_string(rank));
    std::vector<int> out_shape(rank);
    std::vector<int> offsets(rank);
    for (int i = 0; i < rank; ++i) {
        if (slices[i].start < 0 || slices[i].end > shape_[i] || slices[i].start >= slices[i].end)
            throw ShapeError("Invalid slice at dim " + std::to_string(i) + ": [" + std::to_string(slices[i].start) + ", " + std::to_string(slices[i].end) + ")");
        out_shape[i] = slices[i].end - slices[i].start;
        offsets[i] = slices[i].start;
    }
    Tensor out(std::move(out_shape));
    std::vector<int> out_indices(rank, 0);
    for (int out_flat = 0; out_flat < out.numel(); ++out_flat) {
        std::vector<int> in_indices(rank);
        for (int i = 0; i < rank; ++i) in_indices[i] = offsets[i] + out_indices[i];
        out.storage_[out_flat] = storage_[flat_index(in_indices)];
        for (int i = rank - 1; i >= 0; --i) {
            ++out_indices[i];
            if (out_indices[i] < out.shape()[i]) break;
            out_indices[i] = 0;
        }
    }
    return out;
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

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    if (shape_product(new_shape) != numel())
        throw ShapeError("Cannot reshape tensor of shape " + shape_to_string(shape_) +
                         " with " + std::to_string(numel()) + " elements into shape " +
                         shape_to_string(new_shape) + " with " +
                         std::to_string(shape_product(new_shape)) + " elements");
    Tensor out(std::move(new_shape));
    out.storage_ = storage_;
    return out;
}

Tensor Tensor::view(std::vector<int> new_shape) const {
    return reshape(std::move(new_shape));
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
