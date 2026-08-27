// tensor.cpp
#include "torc/tensor.hpp"
#include "simd_ops.hpp"
#include <ostream>
#include <format>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <string_view>

namespace torc {

namespace {
void require_finite(const Tensor& tensor, std::string_view operation) {
    for (int i = 0; i < tensor.numel(); ++i) {
        if (!std::isfinite(tensor.data()[i]))
            throw NumericalError(std::format("{} requires finite input values", operation));
    }
}

void require_nonzero(const Tensor& tensor, std::string_view operation) {
    for (int i = 0; i < tensor.numel(); ++i) {
        if (tensor.data()[i] == 0.0f)
            throw NumericalError(std::format("{} does not allow division by zero", operation));
    }
}

// odometer-style increment of a multi-index in row-major order; shared by transpose/slice
void advance_indices(std::vector<int>& idx, std::span<const int> shape) {
    for (int i = (int)idx.size() - 1; i >= 0; --i) {
        if (++idx[i] < shape[i]) return;
        idx[i] = 0;
    }
}
} // namespace

Tensor::Tensor(std::vector<int> shape) : shape_(std::move(shape)), numel_valid_(false) {
    validate_shape(shape_);
    storage_.resize(shape_product(shape_), 0.0f);
}

Tensor::Tensor(std::initializer_list<float> data, std::vector<int> shape)
    : shape_(std::move(shape)), storage_(data), numel_valid_(false) {
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

int Tensor::numel() const {
    if (!numel_valid_) {
        numel_ = shape_product(shape_);
        numel_valid_ = true;
    }
    return numel_;
}

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
    int n = out.numel();

    std::vector<int> a_strides(rank), b_strides(rank);
    int a_rank = (int)shape_.size(), b_rank = (int)other.shape_.size();
    int a_offset = rank - a_rank, b_offset = rank - b_rank;

    int a_stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
        int a_dim = (d < a_offset) ? 1 : shape_[d - a_offset];
        a_strides[d] = (a_dim == 1) ? 0 : a_stride;
        if (d >= a_offset) a_stride *= shape_[d - a_offset];
    }

    int b_stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
        int b_dim = (d < b_offset) ? 1 : other.shape_[d - b_offset];
        b_strides[d] = (b_dim == 1) ? 0 : b_stride;
        if (d >= b_offset) b_stride *= other.shape_[d - b_offset];
    }

#if defined(_OPENMP)
    #pragma omp parallel for if(n > 65536) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        std::vector<int> out_indices(rank);
        int tmp = i;
        for (int d = rank - 1; d >= 0; --d) {
            out_indices[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
        int a_idx = 0, b_idx = 0;
        for (int d = 0; d < rank; ++d) {
            a_idx += out_indices[d] * a_strides[d];
            b_idx += out_indices[d] * b_strides[d];
        }
        out.storage_[i] = op(storage_[a_idx], other.storage_[b_idx]);
    }
    return out;
}

    Tensor Tensor::add(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor out(shape_);
            simd::add(storage_.data(), other.storage_.data(), out.storage_.data(), numel());
            return out;
        }
        return elementwise_binary_op(other, std::plus<>{});
    }
    Tensor Tensor::sub(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor out(shape_);
            simd::sub(storage_.data(), other.storage_.data(), out.storage_.data(), numel());
            return out;
        }
        return elementwise_binary_op(other, std::minus<>{});
    }
    Tensor Tensor::mul(const Tensor& other) const {
        if (shape_ == other.shape_) {
            Tensor out(shape_);
            simd::mul(storage_.data(), other.storage_.data(), out.storage_.data(), numel());
            return out;
        }
        return elementwise_binary_op(other, std::multiplies<>{});
    }
    Tensor Tensor::div(const Tensor& other) const {
        require_finite(*this, "division");
        require_finite(other, "division");
        require_nonzero(other, "division");
        if (shape_ == other.shape_) {
            Tensor out(shape_);
            simd::div(storage_.data(), other.storage_.data(), out.storage_.data(), numel());
            return out;
        }
        return elementwise_binary_op(other, std::divides<>{});
    }

Tensor Tensor::add(float scalar) const {
    Tensor out(shape_);
    simd::add_scalar(storage_.data(), scalar, out.storage_.data(), numel());
    return out;
}
Tensor Tensor::sub(float scalar) const {
    Tensor out(shape_);
    simd::sub_scalar(storage_.data(), scalar, out.storage_.data(), numel());
    return out;
}
Tensor Tensor::mul(float scalar) const {
    Tensor out(shape_);
    simd::mul_scalar(storage_.data(), scalar, out.storage_.data(), numel());
    return out;
}
Tensor Tensor::div(float scalar) const {
    require_finite(*this, "division");
    if (!std::isfinite(scalar)) throw NumericalError("division requires a finite scalar");
    if (scalar == 0.0f) throw NumericalError("division does not allow division by zero");
    Tensor out(shape_);
    simd::div_scalar(storage_.data(), scalar, out.storage_.data(), numel());
    return out;
}

Tensor Tensor::operator-() const {
    Tensor out(shape_);
    simd::neg(storage_.data(), out.storage_.data(), numel());
    return out;
}

Tensor Tensor::exp() const {
    require_finite(*this, "exp");
    Tensor out(shape_);
    simd::exp(storage_.data(), out.storage_.data(), numel());
    return out;
}

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

    constexpr int Mc = 32, Nc = 32, Kc = 32;

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

        for (int i0 = 0; i0 < m; i0 += Mc) {
            int imax = std::min(i0 + Mc, m);
            for (int k0 = 0; k0 < k; k0 += Kc) {
                int kmax = std::min(k0 + Kc, k);
                for (int j0 = 0; j0 < n; j0 += Nc) {
                    int jmax = std::min(j0 + Nc, n);
                    for (int i = i0; i < imax; ++i) {
                        for (int p = k0; p < kmax; ++p) {
                            float a_val = storage_[a_off + i * k + p];
                            float* out_row = out.storage_.data() + out_off + i * n;
                            const float* b_row = other.storage_.data() + b_off + p * n;
                            int j = j0;
#if TORC_HAS_AVX2
                            for (; j + 8 <= jmax; j += 8) {
                                __m256 b_vec = _mm256_loadu_ps(b_row + j);
                                __m256 a_vec = _mm256_set1_ps(a_val);
                                __m256 out_vec = _mm256_loadu_ps(out_row + j);
                                __m256 prod = _mm256_mul_ps(a_vec, b_vec);
                                __m256 res = _mm256_add_ps(out_vec, prod);
                                _mm256_storeu_ps(out_row + j, res);
                            }
#endif
                            for (; j < jmax; ++j) {
                                out_row[j] += a_val * b_row[j];
                            }
                        }
                    }
                }
            }
        }

        advance_indices(batch_idx, batch);
    }
    return out;
}



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

    std::vector<int> in_strides(rank);
    int stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
        in_strides[d] = stride;
        stride *= shape_[d];
    }

    std::vector<int> out_strides(rank);
    stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
        out_strides[d] = stride;
        stride *= out.shape()[d];
    }

    std::vector<int> out_indices(rank, 0);
    for (int out_flat = 0; out_flat < out.numel(); ++out_flat) {
        int in_flat = 0;
        for (int i = 0; i < rank; ++i)
            in_flat += out_indices[i] * in_strides[axes[i]];
        out.storage_[out_flat] = storage_[in_flat];
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

    std::vector<int> in_strides(rank);
    int stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
        in_strides[d] = stride;
        stride *= shape_[d];
    }

    std::vector<int> out_indices(rank, 0);
    for (int out_flat = 0; out_flat < out.numel(); ++out_flat) {
        int in_flat = 0;
        for (int i = 0; i < rank; ++i)
            in_flat += (offsets[i] + out_indices[i]) * in_strides[i];
        out.storage_[out_flat] = storage_[in_flat];
        advance_indices(out_indices, out.shape());
    }
    return out;
}

float Tensor::sum() const {
    require_finite(*this, "sum");
    return std::ranges::fold_left(storage_, 0.0f, std::plus<>{});
}

float Tensor::mean() const {
    if (numel() == 0) throw ShapeError("Cannot compute mean of empty tensor");
    return sum() / numel();
}

float Tensor::max() const {
    require_finite(*this, "max");
    if (storage_.empty()) throw ShapeError("Cannot compute max of empty tensor");
    return std::ranges::max(storage_);
}

float Tensor::min() const {
    require_finite(*this, "min");
    if (storage_.empty()) throw ShapeError("Cannot compute min of empty tensor");
    return std::ranges::min(storage_);
}

template<typename BinOp>
Tensor Tensor::reduce_axis(int axis, BinOp op) const {
    if (axis < 0 || axis >= (int)shape_.size())
        throw ShapeError(std::format("Invalid axis {} for tensor with shape {}", axis, shape_to_string(shape_)));
    if (numel() == 0)
        throw ShapeError("Cannot reduce empty tensor");
    require_finite(*this, "reduction");

    std::vector<int> out_shape = shape_;
    out_shape.erase(out_shape.begin() + axis);
    Tensor out(out_shape);

    int outer_stride = shape_product(std::span(shape_).subspan(axis + 1));
    int inner_stride = shape_product(std::span(shape_).subspan(0, axis));
    int axis_size = shape_[axis];

#if defined(_OPENMP)
    #pragma omp parallel for if(inner_stride > 65536) schedule(static)
#endif
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

void Tensor::fill(float val) {
    std::ranges::fill(storage_, val);
}

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

Tensor Tensor::softmax() const {
    if (storage_.empty())
        throw ShapeError("Cannot compute softmax of an empty tensor");
    Tensor out(shape_);
    float max_val = storage_[0];
    for (float value : storage_) {
        if (!std::isfinite(value))
            throw NumericalError("softmax requires finite input values");
        max_val = std::max(max_val, value);
    }
    std::vector<float> exp_vals(numel());
#if defined(_OPENMP)
    #pragma omp parallel for if(numel() > 65536) schedule(static)
#endif
    for (int i = 0; i < numel(); ++i) {
        exp_vals[i] = std::exp(storage_[i] - max_val);
    }
    float sum_exp = std::accumulate(exp_vals.begin(), exp_vals.end(), 0.0f);
#if defined(_OPENMP)
    #pragma omp parallel for if(numel() > 65536) schedule(static)
#endif
    for (int i = 0; i < numel(); ++i) {
        out.storage_[i] = exp_vals[i] / sum_exp;
    }
    return out;
}

Tensor Tensor::softmax(int axis) const {
    int rank = static_cast<int>(shape_.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw ShapeError(std::format("Invalid softmax axis {} for tensor with shape {}",
                                     axis, shape_to_string(shape_)));
    if (storage_.empty())
        throw ShapeError("Cannot compute softmax of an empty tensor");

    Tensor out(shape_);
    int axis_size = shape_[axis];
    int inner_stride = shape_product(std::span<const int>(shape_).subspan(axis + 1));
    int outer_stride = axis_size * inner_stride;
    int outer_count = shape_product(std::span<const int>(shape_).subspan(0, axis));

    for (int outer = 0; outer < outer_count; ++outer) {
        int base = outer * outer_stride;
        for (int inner = 0; inner < inner_stride; ++inner) {
            float max_val = storage_[base + inner];
            for (int a = 0; a < axis_size; ++a) {
                float value = storage_[base + a * inner_stride + inner];
                if (!std::isfinite(value))
                    throw NumericalError("softmax requires finite input values");
                max_val = std::max(max_val, value);
            }

            float sum_exp = 0.0f;
            for (int a = 0; a < axis_size; ++a) {
                float value = std::exp(storage_[base + a * inner_stride + inner] - max_val);
                out.storage_[base + a * inner_stride + inner] = value;
                sum_exp += value;
            }
            for (int a = 0; a < axis_size; ++a)
                out.storage_[base + a * inner_stride + inner] /= sum_exp;
        }
    }
    return out;
}

Tensor Tensor::log() const {
    require_finite(*this, "log");
    for (float value : storage_) {
        if (value <= 0.0f)
            throw NumericalError("log requires strictly positive input values");
    }
    Tensor out(shape_);
#if defined(_OPENMP)
    #pragma omp parallel for if(numel() > 65536) schedule(static)
#endif
    for (int i = 0; i < numel(); ++i)
        out.storage_[i] = std::log(storage_[i]);
    return out;
}

Tensor Tensor::sqrt() const {
    require_finite(*this, "sqrt");
    for (float value : storage_) {
        if (value < 0.0f)
            throw NumericalError("sqrt requires non-negative input values");
    }
    Tensor out(shape_);
    simd::sqrt(storage_.data(), out.storage_.data(), numel());
    return out;
}

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
