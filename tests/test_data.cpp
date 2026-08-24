#include <gtest/gtest.h>
#include "torc/data.hpp"
#include <algorithm>

using torc::Tensor;
using torc::data::Dataset;
using torc::data::TensorDataset;
using torc::data::DataLoader;

TEST(TensorDatasetTest, LenReturnsNumberOfSamples) {
    Tensor xs({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor ys({0.0f, 1.0f}, {2});
    TensorDataset ds(std::move(xs), std::move(ys));
    EXPECT_EQ(ds.len(), 2);
}

TEST(TensorDatasetTest, GetReturnsCorrectSample) {
    Tensor xs({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor ys({0.0f, 1.0f}, {2});
    TensorDataset ds(std::move(xs), std::move(ys));

    auto [x0, y0] = ds.get(0);
    EXPECT_FLOAT_EQ(x0.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(x0.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(y0.data()[0], 0.0f);

    auto [x1, y1] = ds.get(1);
    EXPECT_FLOAT_EQ(x1.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(x1.data()[1], 4.0f);
    EXPECT_FLOAT_EQ(y1.data()[0], 1.0f);
}

TEST(TensorDatasetTest, GetOutOfRangeThrows) {
    Tensor xs({1.0f, 2.0f}, {2});
    Tensor ys({0.0f, 1.0f}, {2});
    TensorDataset ds(std::move(xs), std::move(ys));
    EXPECT_THROW(ds.get(2), std::out_of_range);
}

TEST(TensorDatasetTest, MismatchedSampleCountThrows) {
    Tensor xs({1.0f, 2.0f}, {2});
    Tensor ys({0.0f}, {1});
    EXPECT_THROW(TensorDataset(std::move(xs), std::move(ys)), std::invalid_argument);
}

TEST(DataLoaderTest, IteratesAllSamples) {
    Tensor xs({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    Tensor ys({0.0f, 1.0f, 2.0f}, {3});
    TensorDataset ds(std::move(xs), std::move(ys));
    DataLoader loader(ds, 2, false);

    size_t count = 0;
    while (loader.has_next()) {
        auto [x_batch, y_batch] = loader.next_batch();
        count += x_batch.shape().front();
    }
    EXPECT_EQ(count, 3);
}

TEST(DataLoaderTest, BatchSizeAffectsOutputShape) {
    Tensor xs({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor ys({0.0f, 1.0f}, {2, 1});
    TensorDataset ds(std::move(xs), std::move(ys));
    DataLoader loader(ds, 2, false);

    auto [x_batch, y_batch] = loader.next_batch();
    ASSERT_EQ(x_batch.shape().size(), 2);
    EXPECT_EQ(x_batch.shape().front(), 2);
    EXPECT_EQ(x_batch.shape().back(), 2);
    ASSERT_EQ(y_batch.shape().size(), 2);
    EXPECT_EQ(y_batch.shape().front(), 2);
    EXPECT_EQ(y_batch.shape().back(), 1);
}

TEST(DataLoaderTest, LastBatchIsSmaller) {
    Tensor xs({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    Tensor ys({0.0f, 1.0f, 2.0f}, {3});
    TensorDataset ds(std::move(xs), std::move(ys));
    DataLoader loader(ds, 2, false);

    auto [x1, y1] = loader.next_batch();
    EXPECT_EQ(x1.shape().front(), 2);

    auto [x2, y2] = loader.next_batch();
    EXPECT_EQ(x2.shape().front(), 1);
}

TEST(DataLoaderTest, ShuffleProducesValidPermutation) {
    Tensor xs({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
    Tensor ys({0.0f, 1.0f, 2.0f}, {3});
    TensorDataset ds(std::move(xs), std::move(ys));
    DataLoader loader(ds, 1, true);

    std::vector<float> first_elements;
    while (loader.has_next()) {
        auto [x, y] = loader.next_batch();
        first_elements.push_back(x.data()[0]);
    }

    std::vector<float> sorted = first_elements;
    std::sort(sorted.begin(), sorted.end());
    std::vector<float> expected = {1.0f, 3.0f, 5.0f};
    EXPECT_EQ(sorted, expected);
}

TEST(DataLoaderTest, ResetStartsNewEpoch) {
    Tensor xs({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor ys({0.0f, 1.0f}, {2});
    TensorDataset ds(std::move(xs), std::move(ys));
    DataLoader loader(ds, 2, false);

    auto [x1, y1] = loader.next_batch();
    EXPECT_FALSE(loader.has_next());

    loader.reset();
    EXPECT_TRUE(loader.has_next());

    auto [x2, y2] = loader.next_batch();
    EXPECT_FLOAT_EQ(x2.data()[0], 1.0f);
}

TEST(DataLoaderTest, BatchSizeLargerThanDataset) {
    Tensor xs({1.0f, 2.0f}, {1, 2});
    Tensor ys({0.0f}, {1});
    TensorDataset ds(std::move(xs), std::move(ys));
    DataLoader loader(ds, 4, false);

    auto [x_batch, y_batch] = loader.next_batch();
    EXPECT_EQ(x_batch.shape().front(), 1);
    EXPECT_FALSE(loader.has_next());
}

TEST(DataLoaderTest, EmptyDataset) {
    Tensor xs({}, {0, 2});
    Tensor ys({}, {0});
    TensorDataset ds(std::move(xs), std::move(ys));
    DataLoader loader(ds, 2, false);

    EXPECT_FALSE(loader.has_next());
}
