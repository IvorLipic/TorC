#pragma once
#include <vector>
#include <memory>
#include <initializer_list>

namespace torc {

class Tensor {
public:
    Tensor(std::vector<int> shape);
    Tensor(std::initializer_list<float> data, std::vector<int> shape);

    float* data();
    const float* data() const;

    const std::vector<int>& shape() const { return shape_; }
    int numel() const;

    // simple ops
    Tensor add(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;

private:
    std::vector<int> shape_;
    std::vector<float> storage_;  // naive: always float32
};

} // namespace torc
