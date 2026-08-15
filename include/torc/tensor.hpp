#pragma once
#include <vector>
#include <initializer_list>
#include <iosfwd>

namespace torc {

class Tensor {
public:
    Tensor(std::vector<int> shape);
    Tensor(std::initializer_list<float> data, std::vector<int> shape);

    float* data();
    const float* data() const;

    const std::vector<int>& shape() const { return shape_; }
    int numel() const;

    // elementwise tensor-tensor ops (require identical shape)
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

    // reductions: whole-tensor return float, axis-wise return Tensor
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
    std::vector<float> storage_;  // naive: always float32

    void check_same_shape(const Tensor& other) const;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace torc
