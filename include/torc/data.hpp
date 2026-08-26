// data.hpp
#pragma once
#include "torc/tensor.hpp"
#include <vector>
#include <utility>
#include <random>
#include <string>

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

class SyntheticRegression : public Dataset {
public:
    SyntheticRegression(size_t num_samples, int num_features, float weight, float bias, float noise_std, unsigned int seed = 42);
    size_t len() const override;
    std::pair<Tensor, Tensor> get(size_t idx) const override;

private:
    Tensor xs_;
    Tensor ys_;
};

class CSVDataset : public Dataset {
public:
    struct Options {
        bool has_header = false;
        char delimiter = ',';
        size_t feature_cols = 0;
        size_t target_col = 0;
    };

    CSVDataset(const std::string& filepath, Options opts);
    CSVDataset(const std::string& filepath);
    size_t len() const override;
    std::pair<Tensor, Tensor> get(size_t idx) const override;

    static std::vector<std::string> split_line(const std::string& line, char delimiter);
    static float parse_float(const std::string& token);

private:
    Tensor xs_;
    Tensor ys_;
};

class MNISTDataset : public Dataset {
public:
    MNISTDataset(const std::string& filepath, size_t max_samples = 0);
    size_t len() const override;
    std::pair<Tensor, Tensor> get(size_t idx) const override;

private:
    Tensor xs_;
    Tensor ys_;
    size_t len_ = 0;
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
