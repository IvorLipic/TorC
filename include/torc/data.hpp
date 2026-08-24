// data.hpp
#pragma once
#include "torc/tensor.hpp"
#include <vector>
#include <utility>
#include <random>

namespace torc::data {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t len() const = 0;
    virtual std::pair<Tensor, Tensor> get(size_t idx) const = 0;
};

class TensorDataset : public Dataset {
public:
    TensorDataset(Tensor xs, Tensor ys);
    size_t len() const override;
    std::pair<Tensor, Tensor> get(size_t idx) const override;

private:
    Tensor xs_;
    Tensor ys_;
};

class DataLoader {
public:
    DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle = false);
    std::pair<Tensor, Tensor> next_batch();
    bool has_next() const;
    void reset();

private:
    const Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
    size_t current_;
    mutable std::mt19937 rng_;
};

} // namespace torc::data
