#include <gtest/gtest.h>
#include "torc/data.hpp"
#include <algorithm>
#include <fstream>
#include <string>

using torc::Tensor;
using torc::data::Dataset;
using torc::data::TensorDataset;
using torc::data::SyntheticRegression;
using torc::data::CSVDataset;
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

TEST(SyntheticRegressionTest, LenReturnsNumSamples) {
    SyntheticRegression ds(100, 2, 3.0f, 1.0f, 0.1f);
    EXPECT_EQ(ds.len(), 100);
}

TEST(SyntheticRegressionTest, GetReturnsCorrectShape) {
    SyntheticRegression ds(10, 3, 2.0f, -1.0f, 0.1f);
    auto [x, y] = ds.get(0);
    ASSERT_EQ(x.shape().size(), 1);
    EXPECT_EQ(x.shape()[0], 3);
    ASSERT_EQ(y.shape().size(), 1);
    EXPECT_EQ(y.shape()[0], 1);
}

TEST(SyntheticRegressionTest, ReproducibleWithSameSeed) {
    SyntheticRegression ds1(10, 2, 3.0f, 1.0f, 0.1f, 123);
    SyntheticRegression ds2(10, 2, 3.0f, 1.0f, 0.1f, 123);
    for (size_t i = 0; i < ds1.len(); ++i) {
        auto [x1, y1] = ds1.get(i);
        auto [x2, y2] = ds2.get(i);
        for (size_t j = 0; j < x1.numel(); ++j) {
            EXPECT_FLOAT_EQ(x1.data()[j], x2.data()[j]);
        }
        EXPECT_FLOAT_EQ(y1.data()[0], y2.data()[0]);
    }
}

TEST(SyntheticRegressionTest, DifferentSeedProducesDifferentData) {
    SyntheticRegression ds1(10, 1, 3.0f, 1.0f, 0.1f, 123);
    SyntheticRegression ds2(10, 1, 3.0f, 1.0f, 0.1f, 456);
    auto [x1, y1] = ds1.get(0);
    auto [x2, y2] = ds2.get(0);
    EXPECT_NE(x1.data()[0], x2.data()[0]);
}

TEST(CSVDatasetTest, LoadSimpleCSV) {
    std::string path = "test_simple.csv";
    {
        std::ofstream file(path);
        file << "1.0,2.0,3.0\n";
        file << "4.0,5.0,6.0\n";
        file << "7.0,8.0,9.0\n";
    }

    CSVDataset::Options opts;
    opts.feature_cols = 2;
    opts.target_col = 2;
    CSVDataset ds(path, opts);
    EXPECT_EQ(ds.len(), 3);

    auto [x0, y0] = ds.get(0);
    EXPECT_FLOAT_EQ(x0.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(x0.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(y0.data()[0], 3.0f);

    auto [x1, y1] = ds.get(1);
    EXPECT_FLOAT_EQ(x1.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(x1.data()[1], 5.0f);
    EXPECT_FLOAT_EQ(y1.data()[0], 6.0f);

    std::remove(path.c_str());
}

TEST(CSVDatasetTest, SkipHeader) {
    std::string path = "test_header.csv";
    {
        std::ofstream file(path);
        file << "x1,x2,y\n";
        file << "1.0,2.0,3.0\n";
        file << "4.0,5.0,6.0\n";
    }

    CSVDataset::Options opts;
    opts.has_header = true;
    opts.feature_cols = 2;
    opts.target_col = 2;
    CSVDataset ds(path, opts);
    EXPECT_EQ(ds.len(), 2);

    auto [x, y] = ds.get(0);
    EXPECT_FLOAT_EQ(x.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[0], 3.0f);

    std::remove(path.c_str());
}

TEST(CSVDatasetTest, CustomDelimiter) {
    std::string path = "test_semicolon.csv";
    {
        std::ofstream file(path);
        file << "1.0;2.0;3.0\n";
        file << "4.0;5.0;6.0\n";
    }

    CSVDataset::Options opts;
    opts.delimiter = ';';
    opts.feature_cols = 2;
    opts.target_col = 2;
    CSVDataset ds(path, opts);
    EXPECT_EQ(ds.len(), 2);

    auto [x, y] = ds.get(1);
    EXPECT_FLOAT_EQ(x.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(y.data()[0], 6.0f);

    std::remove(path.c_str());
}

TEST(CSVDatasetTest, MalformedLineThrows) {
    std::string path = "test_malformed.csv";
    {
        std::ofstream file(path);
        file << "1.0,2.0,not_a_number\n";
    }

    EXPECT_THROW(CSVDataset ds(path, CSVDataset::Options()), std::invalid_argument);

    std::remove(path.c_str());
}

TEST(CSVDatasetTest, FileNotFoundThrows) {
    EXPECT_THROW(CSVDataset ds("nonexistent_file.csv", CSVDataset::Options()), std::runtime_error);
}

TEST(CSVDatasetTest, InconsistentColumnsThrows) {
    std::string path = "test_inconsistent.csv";
    {
        std::ofstream file(path);
        file << "1.0,2.0,3.0\n";
        file << "4.0,5.0\n";
    }

    EXPECT_THROW(CSVDataset ds(path, CSVDataset::Options()), std::runtime_error);

    std::remove(path.c_str());
}

