// tensor.hpp
#pragma once
#include <vector>
#include <array>
#include <span>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <concepts>
#include "torc/utils.hpp"

namespace torc {

class Tensor {
public:
    explicit Tensor(std::vector<int> shape);
    Tensor(std::initializer_list<float> data, std::vector<int> shape);

    [[nodiscard]] float* data();
    [[nodiscard]] const float* data() const;

    [[nodiscard]] const std::vector<int>& shape() const { return shape_; }
    [[nodiscard]] int numel() const;

    // elementwise tensor-tensor ops (broadcasting)
    [[nodiscard]] Tensor add(const Tensor& other) const;
    [[nodiscard]] Tensor sub(const Tensor& other) const;
    [[nodiscard]] Tensor mul(const Tensor& other) const;
    [[nodiscard]] Tensor div(const Tensor& other) const;

    // scalar ops
    [[nodiscard]] Tensor add(float scalar) const;
    [[nodiscard]] Tensor sub(float scalar) const;
    [[nodiscard]] Tensor mul(float scalar) const;
    [[nodiscard]] Tensor div(float scalar) const;

    [[nodiscard]] Tensor operator-() const;

    bool operator==(const Tensor& other) const = default;

    // indexing — one implementation, deduced-this const/non-const, arg types checked at compile time
    template<typename Self, typename... Args>
        requires (std::integral<Args> && ...)
    decltype(auto) operator[](this Self&& self, Args... args) {
        std::array<int, sizeof...(Args)> idx{static_cast<int>(args)...};
        self.check_index(idx);
        return self.storage_[self.flat_index(idx)];
    }

    [[nodiscard]] Tensor matmul(const Tensor& other) const;

    [[nodiscard]] Tensor transpose(std::vector<int> axes) const;

    struct Slice { int start; int end; };
    [[nodiscard]] Tensor slice(const std::vector<Slice>& slices) const;

    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    [[nodiscard]] Tensor reshape(std::vector<int> new_shape) const;
    [[nodiscard]] Tensor view(std::vector<int> new_shape) const;

    [[nodiscard]] float sum() const;
    [[nodiscard]] float mean() const;
    [[nodiscard]] float max() const;
    [[nodiscard]] float min() const;

    [[nodiscard]] Tensor sum(int axis) const;
    [[nodiscard]] Tensor mean(int axis) const;
    [[nodiscard]] Tensor max(int axis) const;
    [[nodiscard]] Tensor min(int axis) const;
private:
    std::vector<int> shape_;
    std::vector<float> storage_;

    static void validate_shape(std::span<const int> shape);

    int flat_index(std::span<const int> indices) const;
    void check_index(std::span<const int> indices) const;

    // implementation details — defined in tensor.cpp only, never instantiated elsewhere
    template<typename BinOp>
    Tensor elementwise_binary_op(const Tensor& other, BinOp op) const;

    template<typename BinOp>
    Tensor reduce_axis(int axis, BinOp op) const;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace torc