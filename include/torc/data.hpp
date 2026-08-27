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
    virtual std::pair<Tensor, Tensor> get_batch(size_t start, size_t end) const {
        if (end > len()) end = len();
        if (start >= end) return {Tensor(std::vector<int>{0}), Tensor(std::vector<int>{0})};
        std::vector<Tensor> xs, ys;
        xs.reserve(end - start);
        ys.reserve(end - start);
        for (size_t i = start; i < end; ++i) {
            auto [x, y] = get(i);
            xs.push_back(std::move(x));
            ys.push_back(std::move(y));
        }
        return {stack_samples(xs), stack_samples(ys)};
    }

private:
    static Tensor stack_samples(const std::vector<Tensor>& samples) {
        if (samples.empty()) return Tensor(std::vector<int>{0});
        size_t batch_size = samples.size();
        auto sample_shape = samples[0].shape();
        std::vector<int> batch_shape = sample_shape;
        batch_shape.insert(batch_shape.begin(), static_cast<int>(batch_size));
        Tensor result(batch_shape);
        size_t sample_size = samples[0].numel();
        for (size_t i = 0; i < batch_size; ++i) {
            const float* src = samples[i].data();
            float* dst = result.data() + i * sample_size;
            std::copy(src, src + sample_size, dst);
        }
        return result;
    }
};

class TensorDataset : public Dataset {
public:
    TensorDataset(Tensor xs, Tensor ys);
    size_t len() const override;
    std::pair<Tensor, Tensor> get(size_t idx) const override;
    std::pair<Tensor, Tensor> get_batch(size_t start, size_t end) const override;

private:
    Tensor xs_;
    Tensor ys_;
};

class SyntheticRegression : public Dataset {
public:
    SyntheticRegression(size_t num_samples, int num_features, float weight, float bias, float noise_std, unsigned int seed = 42);
    size_t len() const override;
    std::pair<Tensor, Tensor> get(size_t idx) const override;
    std::pair<Tensor, Tensor> get_batch(size_t start, size_t end) const override;

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
    std::pair<Tensor, Tensor> get_batch(size_t start, size_t end) const override;

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
    std::pair<Tensor, Tensor> get_batch(size_t start, size_t end) const override;

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
