// data.cpp
#include "torc/data.hpp"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <cmath>

namespace torc::data {

TensorDataset::TensorDataset(Tensor xs, Tensor ys)
    : xs_(std::move(xs)), ys_(std::move(ys)) {
    if (xs_.shape().empty() || ys_.shape().empty()) {
        throw std::invalid_argument("TensorDataset: xs and ys must have at least one dimension");
    }
    if (xs_.shape().front() != ys_.shape().front()) {
        throw std::invalid_argument("TensorDataset: xs and ys must have the same number of samples");
    }
}

size_t TensorDataset::len() const {
    return xs_.shape().front();
}

std::pair<Tensor, Tensor> TensorDataset::get(size_t idx) const {
    if (idx >= len()) {
        throw std::out_of_range("TensorDataset::get: index out of range");
    }

    auto xs_shape = xs_.shape();
    auto ys_shape = ys_.shape();

    std::vector<Tensor::Slice> x_slices(xs_shape.size());
    x_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < xs_shape.size(); ++i) {
        x_slices[i] = Tensor::Slice{0, xs_shape[i]};
    }
    Tensor x = xs_.slice(x_slices);
    std::vector<int> x_sample_shape(xs_shape.begin() + 1, xs_shape.end());
    x = x.reshape(x_sample_shape);

    std::vector<Tensor::Slice> y_slices(ys_shape.size());
    y_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < ys_shape.size(); ++i) {
        y_slices[i] = Tensor::Slice{0, ys_shape[i]};
    }
    Tensor y = ys_.slice(y_slices);
    std::vector<int> y_sample_shape(ys_shape.begin() + 1, ys_shape.end());
    y = y.reshape(y_sample_shape);

    return {std::move(x), std::move(y)};
}

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), current_(0) {
    if (batch_size_ == 0) {
        throw std::invalid_argument("DataLoader: batch_size must be > 0");
    }
    size_t n = dataset_.len();
    indices_.resize(n);
    for (size_t i = 0; i < n; ++i) indices_[i] = i;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}

std::pair<Tensor, Tensor> DataLoader::next_batch() {
    if (!has_next()) {
        throw std::runtime_error("DataLoader::next_batch: no more batches in current epoch");
    }

    size_t end = std::min(current_ + batch_size_, indices_.size());
    size_t actual_batch_size = end - current_;

    std::vector<Tensor> x_samples;
    std::vector<Tensor> y_samples;
    x_samples.reserve(actual_batch_size);
    y_samples.reserve(actual_batch_size);

    for (size_t i = current_; i < end; ++i) {
        auto [x, y] = dataset_.get(indices_[i]);
        x_samples.push_back(std::move(x));
        y_samples.push_back(std::move(y));
    }
    current_ = end;

    auto stack = [](const std::vector<Tensor>& samples) -> Tensor {
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
    };

    return {stack(x_samples), stack(y_samples)};
}

bool DataLoader::has_next() const {
    return current_ < indices_.size();
}

void DataLoader::reset() {
    current_ = 0;
    if (shuffle_) {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
}

SyntheticRegression::SyntheticRegression(size_t num_samples, int num_features, float weight, float bias, float noise_std, unsigned int seed)
    : xs_(std::vector<int>{static_cast<int>(num_samples), num_features}),
      ys_(std::vector<int>{static_cast<int>(num_samples), 1}) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, noise_std);

    for (size_t i = 0; i < num_samples; ++i) {
        float x_val = x_dist(rng);
        float y_val = weight * x_val + bias + noise_dist(rng);
        xs_.data()[i * num_features] = x_val;
        ys_.data()[i] = y_val;
    }
}

size_t SyntheticRegression::len() const {
    return xs_.shape().front();
}

std::pair<Tensor, Tensor> SyntheticRegression::get(size_t idx) const {
    if (idx >= len()) {
        throw std::out_of_range("SyntheticRegression::get: index out of range");
    }

    auto xs_shape = xs_.shape();
    auto ys_shape = ys_.shape();

    std::vector<Tensor::Slice> x_slices(xs_shape.size());
    x_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < xs_shape.size(); ++i) {
        x_slices[i] = Tensor::Slice{0, xs_shape[i]};
    }
    Tensor x = xs_.slice(x_slices);
    std::vector<int> x_sample_shape(xs_shape.begin() + 1, xs_shape.end());
    x = x.reshape(x_sample_shape);

    std::vector<Tensor::Slice> y_slices(ys_shape.size());
    y_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < ys_shape.size(); ++i) {
        y_slices[i] = Tensor::Slice{0, ys_shape[i]};
    }
    Tensor y = ys_.slice(y_slices);
    std::vector<int> y_sample_shape(ys_shape.begin() + 1, ys_shape.end());
    y = y.reshape(y_sample_shape);

    return {std::move(x), std::move(y)};
}

std::vector<std::string> CSVDataset::split_line(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

float CSVDataset::parse_float(const std::string& token) {
    try {
        return std::stof(token);
    } catch (const std::exception&) {
        throw std::invalid_argument("CSVDataset: cannot parse float from token: '" + token + "'");
    }
}

CSVDataset::CSVDataset(const std::string& filepath)
    : CSVDataset(filepath, Options{}) {
}

CSVDataset::CSVDataset(const std::string& filepath, Options opts)
    : xs_(std::vector<int>{0}), ys_(std::vector<int>{0}) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("CSVDataset: cannot open file: " + filepath);
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    size_t line_num = 0;
    size_t expected_cols = 0;

    while (std::getline(file, line)) {
        ++line_num;
        if (line_num == 1 && opts.has_header) {
            continue;
        }

        if (line.empty()) continue;

        auto tokens = split_line(line, opts.delimiter);
        if (tokens.empty()) continue;

        size_t total_cols = tokens.size();
        if (expected_cols == 0) {
            expected_cols = total_cols;
        } else if (total_cols != expected_cols) {
            throw std::runtime_error("CSVDataset: inconsistent column count at line " + std::to_string(line_num));
        }

        if (opts.target_col >= total_cols) {
            throw std::runtime_error("CSVDataset: target_col out of range at line " + std::to_string(line_num));
        }
        if (opts.feature_cols > total_cols) {
            throw std::runtime_error("CSVDataset: feature_cols exceeds column count at line " + std::to_string(line_num));
        }

        std::vector<float> row;
        row.reserve(total_cols);
        for (const auto& token : tokens) {
            row.push_back(parse_float(token));
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        throw std::runtime_error("CSVDataset: no data rows found in file: " + filepath);
    }

    size_t num_samples = rows.size();
    size_t num_features = opts.feature_cols;

    xs_ = Tensor(std::vector<int>{static_cast<int>(num_samples), static_cast<int>(num_features)});
    ys_ = Tensor(std::vector<int>{static_cast<int>(num_samples), 1});

    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            xs_.data()[i * num_features + j] = rows[i][j];
        }
        ys_.data()[i] = rows[i][opts.target_col];
    }
}

size_t CSVDataset::len() const {
    return xs_.shape().front();
}

std::pair<Tensor, Tensor> CSVDataset::get(size_t idx) const {
    if (idx >= len()) {
        throw std::out_of_range("CSVDataset::get: index out of range");
    }

    auto xs_shape = xs_.shape();
    auto ys_shape = ys_.shape();

    std::vector<Tensor::Slice> x_slices(xs_shape.size());
    x_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < xs_shape.size(); ++i) {
        x_slices[i] = Tensor::Slice{0, xs_shape[i]};
    }
    Tensor x = xs_.slice(x_slices);
    std::vector<int> x_sample_shape(xs_shape.begin() + 1, xs_shape.end());
    x = x.reshape(x_sample_shape);

    std::vector<Tensor::Slice> y_slices(ys_shape.size());
    y_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < ys_shape.size(); ++i) {
        y_slices[i] = Tensor::Slice{0, ys_shape[i]};
    }
    Tensor y = ys_.slice(y_slices);
    std::vector<int> y_sample_shape(ys_shape.begin() + 1, ys_shape.end());
    y = y.reshape(y_sample_shape);

    return {std::move(x), std::move(y)};
}

MNISTDataset::MNISTDataset(const std::string& filepath, size_t max_samples)
    : xs_(std::vector<int>{0}), ys_(std::vector<int>{0}), len_(0) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("MNISTDataset: cannot open file: " + filepath);
    }

    std::vector<std::vector<float>> rows;
    std::string line;
    size_t line_num = 0;
    size_t expected_cols = 0;
    size_t loaded_samples = 0;

    while (std::getline(file, line) && (max_samples == 0 || loaded_samples < max_samples)) {
        ++line_num;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        auto tokens = CSVDataset::split_line(line, ',');
        if (tokens.empty()) continue;

        size_t total_cols = tokens.size();
        if (expected_cols == 0) {
            expected_cols = total_cols;
        } else if (total_cols != expected_cols) {
            throw std::runtime_error("MNISTDataset: inconsistent column count at line " + std::to_string(line_num));
        }

        if (expected_cols < 2) {
            throw std::runtime_error("MNISTDataset: expected at least 2 columns (label + pixels), got " + std::to_string(expected_cols) + " at line " + std::to_string(line_num));
        }

        std::vector<float> row;
        row.reserve(total_cols);
        for (const auto& token : tokens) {
            row.push_back(CSVDataset::parse_float(token));
        }
        rows.push_back(std::move(row));
        ++loaded_samples;
    }

    if (rows.empty()) {
        throw std::runtime_error("MNISTDataset: no data rows found in file: " + filepath);
    }

    size_t num_samples = rows.size();
    int num_pixels = static_cast<int>(expected_cols - 1);
    xs_ = Tensor(std::vector<int>{static_cast<int>(num_samples), num_pixels});
    ys_ = Tensor(std::vector<int>{static_cast<int>(num_samples), 1});

    for (size_t i = 0; i < num_samples; ++i) {
        ys_.data()[i] = rows[i][0];
        for (size_t j = 0; j < static_cast<size_t>(num_pixels); ++j) {
            xs_.data()[i * num_pixels + j] = rows[i][j + 1] / 255.0f;
        }
    }

    len_ = num_samples;
}

size_t MNISTDataset::len() const {
    return len_;
}

std::pair<Tensor, Tensor> MNISTDataset::get(size_t idx) const {
    if (idx >= len()) {
        throw std::out_of_range("MNISTDataset::get: index out of range");
    }

    auto xs_shape = xs_.shape();
    auto ys_shape = ys_.shape();

    std::vector<Tensor::Slice> x_slices(xs_shape.size());
    x_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < xs_shape.size(); ++i) {
        x_slices[i] = Tensor::Slice{0, xs_shape[i]};
    }
    Tensor x = xs_.slice(x_slices);
    std::vector<int> x_sample_shape(xs_shape.begin() + 1, xs_shape.end());
    x = x.reshape(x_sample_shape);

    std::vector<Tensor::Slice> y_slices(ys_shape.size());
    y_slices[0] = Tensor::Slice{static_cast<int>(idx), static_cast<int>(idx) + 1};
    for (size_t i = 1; i < ys_shape.size(); ++i) {
        y_slices[i] = Tensor::Slice{0, ys_shape[i]};
    }
    Tensor y = ys_.slice(y_slices);
    std::vector<int> y_sample_shape(ys_shape.begin() + 1, ys_shape.end());
    y = y.reshape(y_sample_shape);

    return {std::move(x), std::move(y)};
}

} // namespace torc::data
