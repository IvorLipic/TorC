// matmul_blas.cpp
#include "torc/tensor.hpp"
#include <cblas.h>
#include <algorithm>
#include <stdexcept>

namespace torc {

namespace {

// Compute leading dimension for row-major matrix with given shape
// For row-major: lda = number of columns (inner dimension)
inline int lda_row_major(int rows, int cols) {
    return cols;
}

// Copy a submatrix from strided source to contiguous destination
// src: pointer to source data with leading dimension src_ld
// dst: pointer to destination (contiguous, lda = cols)
// m, n: matrix dimensions
void copy_submatrix(const float* src, int src_ld, float* dst, int m, int n) {
    for (int i = 0; i < m; ++i) {
        std::copy(src + i * src_ld, src + i * src_ld + n, dst + i * n);
    }
}

// Copy contiguous source to strided destination
void copy_to_strided(const float* src, float* dst, int dst_ld, int m, int n) {
    for (int i = 0; i < m; ++i) {
        std::copy(src + i * n, src + i * n + n, dst + i * dst_ld);
    }
}

} // namespace

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

    // Temporary buffers for contiguous matrices (CBLAS expects contiguous row-major)
    std::vector<float> a_contig(m * k);
    std::vector<float> b_contig(k * n);
    std::vector<float> c_contig(m * n);

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

        // Copy A submatrix to contiguous buffer
        int a_lda = (r > 2 && k > 0) ? k : k;  // row-major: lda = cols = k
        if (r == 2 || a_batch == 1) {
            // Single matrix or no batch dim: source is already contiguous
            std::copy(storage_.data() + a_off, storage_.data() + a_off + m * k, a_contig.data());
        } else {
            copy_submatrix(storage_.data() + a_off, k, a_contig.data(), m, k);
        }

        // Copy B submatrix to contiguous buffer
        int b_lda = (r2 > 2 && n > 0) ? n : n;  // row-major: ldb = cols = n
        if (r2 == 2 || b_batch == 1) {
            std::copy(other.storage_.data() + b_off, other.storage_.data() + b_off + k * n, b_contig.data());
        } else {
            copy_submatrix(other.storage_.data() + b_off, n, b_contig.data(), k, n);
        }

        // cblas_sgemm: C = A * B (row-major, no transpose)
        // A: m x k, B: k x n, C: m x n
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f,
                    a_contig.data(), k,  // lda = k
                    b_contig.data(), n,  // ldb = n
                    0.0f,
                    c_contig.data(), n); // ldc = n

        // Copy result to output
        std::copy(c_contig.begin(), c_contig.end(), out.storage_.data() + out_off);

        // Advance batch index
        for (int i = batch_rank - 1; i >= 0; --i) {
            if (++batch_idx[i] < batch[i]) break;
            batch_idx[i] = 0;
        }
    }
    return out;
}

} // namespace torc