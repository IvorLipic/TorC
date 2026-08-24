// data.cpp
#include "torc/data.hpp"
#include <algorithm>
#include <stdexcept>

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

} // namespace torc::data
