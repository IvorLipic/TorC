// tensor.cpp
#include "torc/tensor.hpp"
#include <ostream>
#include <format>
#include <algorithm>
#include <numeric>
#include <functional>

namespace torc {

namespace {
// odometer-style increment of a multi-index in row-major order; shared by transpose/slice
void advance_indices(std::vector<int>& idx, std::span<const int> shape) {
    for (int i = (int)idx.size() - 1; i >= 0; --i) {
        if (++idx[i] < shape[i]) return;
        idx[i] = 0;
    }
}
} // namespace

Tensor::Tensor(std::vector<int> shape) : shape_(std::move(shape)) {
    validate_shape(shape_);
    storage_.resize(shape_product(shape_), 0.0f);
}

Tensor::Tensor(std::initializer_list<float> data, std::vector<int> shape)
    : shape_(std::move(shape)), storage_(data) {
    validate_shape(shape_);
    if (shape_product(shape_) != (int)storage_.size())
        throw ShapeError(std::format(
            "Initializer list size {} does not match shape product {}",
            storage_.size(), shape_product(shape_)));
}

void Tensor::validate_shape(std::span<const int> shape) {
    for (int d : shape)
        if (d < 0)
            throw ShapeError(std::format(
                "Shape dimensions must be non-negative, got {}", shape_to_string(shape)));
}

float* Tensor::data() { return storage_.data(); }
const float* Tensor::data() const { return storage_.data(); }

int Tensor::numel() const { return shape_product(shape_); }

int Tensor::flat_index(std::span<const int> indices) const {
    int idx = 0;
    for (size_t i = 0; i < shape_.size(); ++i)
        idx = idx * shape_[i] + indices[i];
    return idx;
}

void Tensor::check_index(std::span<const int> indices) const {
    if (indices.size() != shape_.size())
        throw ShapeError(std::format(
            "Rank mismatch: got {} indices for rank {}", indices.size(), shape_.size()));
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i])
            throw ShapeError(std::format(
                "Index out of bounds at dim {}: {} not in [0, {})", i, indices[i], shape_[i]));
    }
}

template<typename BinOp>
Tensor Tensor::elementwise_binary_op(const Tensor& other, BinOp op) const {
    auto out_shape = broadcast_shape(shape_, other.shape_);
    Tensor out(out_shape);
    int rank = (int)out_shape.size();

    for (int i = 0; i < out.numel(); ++i) {
        std::vector<int> out_indices(rank);
        int tmp = i;
        for (int d = rank - 1; d >= 0; --d) {
            out_indices[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int a_rank = (int)shape_.size(), b_rank = (int)other.shape_.size();
        int a_offset = rank - a_rank, b_offset = rank - b_rank;

        int a_idx = 0;
        for (int d = 0; d < a_rank; ++d)
            a_idx = a_idx * shape_[d] + (shape_[d] == 1 ? 0 : out_indices[d + a_offset]);
        int b_idx = 0;
        for (int d = 0; d < b_rank; ++d)
            b_idx = b_idx * other.shape_[d] + (other.shape_[d] == 1 ? 0 : out_indices[d + b_offset]);

        out.storage_[i] = op(storage_[a_idx], other.storage_[b_idx]);
    }
    return out;
}

Tensor Tensor::add(const Tensor& other) const { return elementwise_binary_op(other, std::plus<>{}); }
Tensor Tensor::sub(const Tensor& other) const { return elementwise_binary_op(other, std::minus<>{}); }
Tensor Tensor::mul(const Tensor& other) const { return elementwise_binary_op(other, std::multiplies<>{}); }
Tensor Tensor::div(const Tensor& other) const { return elementwise_binary_op(other, std::divides<>{}); }

Tensor Tensor::add(float scalar) const {
    Tensor out(shape_);
    std::ranges::transform(storage_, out.storage_.begin(), [scalar](float x) { return x + scalar; });
    return out;
}
Tensor Tensor::sub(float scalar) const {
    Tensor out(shape_);
    std::ranges::transform(storage_, out.storage_.begin(), [scalar](float x) { return x - scalar; });
    return out;
}
Tensor Tensor::mul(float scalar) const {
    Tensor out(shape_);
    std::ranges::transform(storage_, out.storage_.begin(), [scalar](float x) { return x * scalar; });
    return out;
}
Tensor Tensor::div(float scalar) const {
    Tensor out(shape_);
    std::ranges::transform(storage_, out.storage_.begin(), [scalar](float x) { return x / scalar; });
    return out;
}

Tensor Tensor::operator-() const {
    Tensor out(shape_);
    std::ranges::transform(storage_, out.storage_.begin(), std::negate<>{});
    return out;
}

#ifndef TORC_USE_BLAS

Tensor Tensor::matmul(const Tensor& other) const {
    int r = (int)shape_.size();
    int r2 = (int)other.shape_.size();
    if (r < 2)
        throw ShapeError(std::format("matmul requires rank >= 2 operands, got {} and {}",
                                     shape_to_string(shape_), shape_to_string(other.shape_)));
    if (r2 < 2)
        throw ShapeError(std::format("matmul requires rank >= 2 operands, rank {}, got {} and {}",
                                     r2, shape_to_string(shape_), shape_to_string(other.shape_)));

    int m = shape_[r - 2], k = shape_[r - 1];
    int k2 = other.shape_[r2 - 2], n = other.shape_[r2 - 1];
    if (k != k2)
        throw ShapeError(std::format("matmul inner dimensions must match, got {} and {}",
                                     shape_to_string(shape_), shape_to_string(other.shape_)));

    std::vector<int> batch;
    try {
        batch = broadcast_shape(std::span(shape_).subspan(0, r - 2),
                                std::span(other.shape_).subspan(0, r2 - 2));
    } catch (const ShapeError& e) {
        throw ShapeError(std::format("matmul batch dimensions cannot broadcast for shapes {} and {}: {}",
                                     shape_to_string(shape_), shape_to_string(other.shape_),
                                     e.what()));
    }

    std::vector<int> out_shape = batch;
    out_shape.push_back(m);
    out_shape.push_back(n);
    Tensor out(std::move(out_shape));

    int a_batch = shape_product(std::span(shape_).subspan(0, r - 2));
    int b_batch = shape_product(std::span(other.shape_).subspan(0, r2 - 2));
    int out_batch = shape_product(batch);
    int batch_rank = (int)batch.size();

    std::vector<int> batch_idx(batch_rank, 0);
    for (int b = 0; b < out_batch; ++b) {
        int a_off = 0, b_off = 0;
        if (r > 2) {
            for (int d = 0; d < (int)batch.size(); ++d) {
                int dim_a = (d < batch_rank - (r - 2)) ? 1 : shape_[d - (batch_rank - (r - 2))];
                int idx_a = dim_a == 1 ? 0 : batch_idx[d];
                a_off = a_off * dim_a + idx_a;
            }
            a_off *= m * k;
        }
        if (r2 > 2) {
            for (int d = 0; d < (int)batch.size(); ++d) {
                int dim_b = (d < batch_rank - (r2 - 2)) ? 1 : other.shape_[d - (batch_rank - (r2 - 2))];
                int idx_b = dim_b == 1 ? 0 : batch_idx[d];
                b_off = b_off * dim_b + idx_b;
            }
            b_off *= k * n;
        }
        int out_off = b * m * n;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int p = 0; p < k; ++p)
                    acc += storage_[a_off + i * k + p] * other.storage_[b_off + p * n + j];
                out.storage_[out_off + i * n + j] = acc;
            }
        }
        advance_indices(batch_idx, batch);
    }
    return out;
}

#endif // TORC_USE_BLAS

Tensor Tensor::transpose(std::vector<int> axes) const {
    int rank = (int)shape_.size();
    if (axes.empty()) {
        axes.resize(rank);
        for (int i = 0; i < rank; ++i) axes[i] = rank - 1 - i;
    }
    if ((int)axes.size() != rank)
        throw ShapeError(std::format("Transpose axes size {} does not match rank {}", axes.size(), rank));

    std::vector<bool> seen(rank, false);
    for (int i = 0; i < rank; ++i) {
        if (axes[i] < 0 || axes[i] >= rank)
            throw ShapeError(std::format("Transpose axis {} out of bounds for rank {}", axes[i], rank));
        if (seen[axes[i]])
            throw ShapeError(std::format("Duplicate transpose axis {}", axes[i]));
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
        advance_indices(out_indices, out.shape());
    }
    return out;
}

Tensor Tensor::slice(const std::vector<Slice>& slices) const {
    int rank = (int)shape_.size();
    if ((int)slices.size() != rank)
        throw ShapeError(std::format("Slice count {} does not match rank {}", slices.size(), rank));

    std::vector<int> out_shape(rank), offsets(rank);
    for (int i = 0; i < rank; ++i) {
        if (slices[i].start < 0 || slices[i].end > shape_[i] || slices[i].start >= slices[i].end)
            throw ShapeError(std::format("Invalid slice at dim {}: [{}, {})", i, slices[i].start, slices[i].end));
        out_shape[i] = slices[i].end - slices[i].start;
        offsets[i] = slices[i].start;
    }
    Tensor out(std::move(out_shape));

    std::vector<int> out_indices(rank, 0);
    for (int out_flat = 0; out_flat < out.numel(); ++out_flat) {
        std::vector<int> in_indices(rank);
        for (int i = 0; i < rank; ++i) in_indices[i] = offsets[i] + out_indices[i];
        out.storage_[out_flat] = storage_[flat_index(in_indices)];
        advance_indices(out_indices, out.shape());
    }
    return out;
}

float Tensor::sum() const { return std::ranges::fold_left(storage_, 0.0f, std::plus<>{}); }

float Tensor::mean() const {
    if (numel() == 0) throw ShapeError("Cannot compute mean of empty tensor");
    return sum() / numel();
}

float Tensor::max() const {
    if (storage_.empty()) throw ShapeError("Cannot compute max of empty tensor");
    return std::ranges::max(storage_);
}

float Tensor::min() const {
    if (storage_.empty()) throw ShapeError("Cannot compute min of empty tensor");
    return std::ranges::min(storage_);
}

template<typename BinOp>
Tensor Tensor::reduce_axis(int axis, BinOp op) const {
    if (axis < 0 || axis >= (int)shape_.size())
        throw ShapeError(std::format("Invalid axis {} for tensor with shape {}", axis, shape_to_string(shape_)));
    if (numel() == 0)
        throw ShapeError("Cannot reduce empty tensor");

    std::vector<int> out_shape = shape_;
    out_shape.erase(out_shape.begin() + axis);
    Tensor out(out_shape);

    int outer_stride = shape_product(std::span(shape_).subspan(axis + 1));
    int inner_stride = shape_product(std::span(shape_).subspan(0, axis));
    int axis_size = shape_[axis];

    for (int outer = 0; outer < inner_stride; ++outer) {
        for (int j = 0; j < outer_stride; ++j) {
            int out_idx = outer * outer_stride + j;
            int base = outer * axis_size * outer_stride + j;
            float acc = storage_[base];
            for (int a = 1; a < axis_size; ++a)
                acc = op(acc, storage_[base + a * outer_stride]);
            out.storage_[out_idx] = acc;
        }
    }
    return out;
}

Tensor Tensor::sum(int axis) const { return reduce_axis(axis, std::plus<>{}); }
Tensor Tensor::max(int axis) const { return reduce_axis(axis, [](float a, float b) { return std::max(a, b); }); }
Tensor Tensor::min(int axis) const { return reduce_axis(axis, [](float a, float b) { return std::min(a, b); }); }

Tensor Tensor::mean(int axis) const {
    Tensor s = sum(axis);
    std::ranges::transform(s.storage_, s.storage_.begin(),
                            [n = shape_[axis]](float x) { return x / n; });
    return s;
}

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    if (shape_product(new_shape) != numel())
        throw ShapeError(std::format(
            "Cannot reshape tensor of shape {} with {} elements into shape {} with {} elements",
            shape_to_string(shape_), numel(), shape_to_string(new_shape), shape_product(new_shape)));
    Tensor out(std::move(new_shape));
    out.storage_ = storage_;
    return out;
}

Tensor Tensor::view(std::vector<int> new_shape) const { return reshape(std::move(new_shape)); }

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << std::format("Tensor(shape={}, data=[", shape_to_string(t.shape()));
    for (int i = 0; i < t.numel(); ++i) {
        if (i > 0) os << ", ";
        os << t.data()[i];
    }
    os << "])";
    return os;
}

} // namespace torc