#pragma once
#include <vector>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include "torc/utils.hpp"

namespace torc {

class Tensor {
public:
    Tensor(std::vector<int> shape);
    Tensor(std::initializer_list<float> data, std::vector<int> shape);

    float* data();
    const float* data() const;

    const std::vector<int>& shape() const { return shape_; }
    int numel() const;

    // elementwise tensor-tensor ops (broadcasting)
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;

    // scalar ops
    Tensor add(float scalar) const;
    Tensor sub(float scalar) const;
    Tensor mul(float scalar) const;
    Tensor div(float scalar) const;

    // unary negation
    Tensor operator-() const;

    // comparison
    bool operator==(const Tensor& other) const;

    // indexing
    template<typename... Args>
    float& operator()(Args... args) {
        std::vector<int> idx;
        idx.reserve(sizeof...(Args));
        (void)std::initializer_list<int>{0, (idx.push_back(args), 0)...};
        if ((int)idx.size() != (int)shape_.size())
            throw ShapeError("Rank mismatch: got " + std::to_string(idx.size()) + " indices for rank " + std::to_string(shape_.size()));
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] < 0 || idx[i] >= shape_[i])
                throw ShapeError("Index out of bounds at dim " + std::to_string(i) + ": " + std::to_string(idx[i]) + " not in [0, " + std::to_string(shape_[i]) + ")");
        }
        return storage_[flat_index(idx)];
    }
    template<typename... Args>
    const float& operator()(Args... args) const {
        std::vector<int> idx;
        idx.reserve(sizeof...(Args));
        (void)std::initializer_list<int>{0, (idx.push_back(args), 0)...};
        if ((int)idx.size() != (int)shape_.size())
            throw ShapeError("Rank mismatch: got " + std::to_string(idx.size()) + " indices for rank " + std::to_string(shape_.size()));
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] < 0 || idx[i] >= shape_[i])
                throw ShapeError("Index out of bounds at dim " + std::to_string(i) + ": " + std::to_string(idx[i]) + " not in [0, " + std::to_string(shape_[i]) + ")");
        }
        return storage_[flat_index(idx)];
    }

    // transpose
    Tensor transpose(std::vector<int> axes) const;

    // slice
    struct Slice {
        int start;
        int end;
    };
    Tensor slice(const std::vector<Slice>& slices) const;

    // rule of five: std::vector members handle everything; defaulted for clarity
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    // reshape / view — no-copy where possible (moves flat storage)
    Tensor reshape(std::vector<int> new_shape) const;
    Tensor view(std::vector<int> new_shape) const;

    // reductions
    float sum() const;
    float mean() const;
    float max() const;
    float min() const;

    Tensor sum(int axis) const;
    Tensor mean(int axis) const;
    Tensor max(int axis) const;
    Tensor min(int axis) const;

private:
    std::vector<int> shape_;
    std::vector<float> storage_;

    int flat_index(const std::vector<int>& indices) const;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace torc
