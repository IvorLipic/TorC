#include <gtest/gtest.h>
#include "torc/tensor.hpp"
#include "torc/utils.hpp"
#include <sstream>
#include <string>
#include <cstdio>
#include <cmath>

using torc::Tensor;

TEST(TensorConstruction, ZeroFilledFromShape) {
    Tensor t({2, 3});
    EXPECT_EQ(t.numel(), 6);
    for (int i = 0; i < t.numel(); ++i)
        EXPECT_EQ(t.data()[i], 0.0f);
}

TEST(TensorConstruction, InitializerListWithMatchingShape) {
    Tensor t({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_EQ(t.numel(), 3);
    EXPECT_FLOAT_EQ(t.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(t.data()[2], 3.0f);
}

TEST(TensorConstruction, InitializerListSizeMismatchThrows) {
    EXPECT_THROW(Tensor({1.0f, 2.0f}, {3}), torc::ShapeError);
}

TEST(TensorConstruction, ShapeIsAccessible) {
    Tensor t({2, 3, 4});
    const auto& s = t.shape();
    ASSERT_EQ(s.size(), 3);
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(s[2], 4);
    EXPECT_EQ(t.numel(), 24);
}

TEST(TensorData, MutableAndConstAccess) {
    Tensor t({5});
    t.data()[0] = 42.0f;
    const Tensor ct({1.0f}, {1});
    EXPECT_FLOAT_EQ(ct.data()[0], 1.0f);
}

TEST(TensorOps, AddIdenticalShape) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({4.0f, 5.0f, 6.0f}, {3});
    Tensor c = a.add(b);
    EXPECT_EQ(c.numel(), 3);
    EXPECT_FLOAT_EQ(c.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 7.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 9.0f);
}

TEST(TensorOps, SubIdenticalShape) {
    Tensor a({10.0f, 20.0f, 30.0f}, {3});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    Tensor c = a.sub(b);
    EXPECT_FLOAT_EQ(c.data()[0], 9.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 18.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 27.0f);
}

TEST(TensorOps, MulIdenticalShape) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({4.0f, 5.0f, 6.0f}, {3});
    Tensor c = a.mul(b);
    EXPECT_FLOAT_EQ(c.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 10.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 18.0f);
}

TEST(TensorOps, DivIdenticalShape) {
    Tensor a({10.0f, 20.0f, 30.0f}, {3});
    Tensor b({2.0f, 4.0f, 5.0f}, {3});
    Tensor c = a.div(b);
    EXPECT_FLOAT_EQ(c.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 6.0f);
}

TEST(TensorOps, AddShapeMismatchThrows) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.add(b), torc::ShapeError);
}

TEST(TensorOps, SubShapeMismatchThrows) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.sub(b), torc::ShapeError);
}

TEST(TensorOps, MulShapeMismatchThrows) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.mul(b), torc::ShapeError);
}

TEST(TensorOps, DivShapeMismatchThrows) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.div(b), torc::ShapeError);
}

TEST(TensorOps, MultiDimElementwise) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2});
    Tensor c = a.add(b);
    EXPECT_FLOAT_EQ(c.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 44.0f);
}

TEST(TensorScalarOps, AddScalar) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor c = a.add(10.0f);
    EXPECT_FLOAT_EQ(c.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 12.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 13.0f);
}

TEST(TensorScalarOps, SubScalar) {
    Tensor a({10.0f, 20.0f, 30.0f}, {3});
    Tensor c = a.sub(5.0f);
    EXPECT_FLOAT_EQ(c.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 15.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 25.0f);
}

TEST(TensorScalarOps, MulScalar) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor c = a.mul(3.0f);
    EXPECT_FLOAT_EQ(c.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 6.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 9.0f);
}

TEST(TensorScalarOps, DivScalar) {
    Tensor a({10.0f, 20.0f, 30.0f}, {3});
    Tensor c = a.div(2.0f);
    EXPECT_FLOAT_EQ(c.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 10.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 15.0f);
}

TEST(TensorComparison, OperatorEqual) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    Tensor c({1.0f, 2.0f, 4.0f}, {3});
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(TensorComparison, OperatorEqualDifferentShape) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f}, {1, 2});
    EXPECT_FALSE(a == b);
}

TEST(TensorPrinting, OstreamOutput) {
    Tensor t({1.0f, 2.0f, 3.0f}, {3});
    std::ostringstream oss;
    oss << t;
    EXPECT_EQ(oss.str(), "Tensor(shape=(3), data=[1, 2, 3])");
}

TEST(TensorErrors, ShapeErrorDerivesFromTorcError) {
    EXPECT_THROW({
        try {
            Tensor({1.0f, 2.0f}, {3});
        } catch (const torc::ShapeError& e) {
            EXPECT_TRUE(std::string(e.what()).find("does not match") != std::string::npos);
            throw;
        }
    }, torc::ShapeError);
}

TEST(TensorNegation, NegatesValues) {
    Tensor a({1.0f, -2.0f, 3.0f}, {3});
    Tensor c = -a;
    EXPECT_FLOAT_EQ(c.data()[0], -1.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(c.data()[2], -3.0f);
}

TEST(TensorNegation, PreservesShape) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor c = -a;
    const auto& s = c.shape();
    ASSERT_EQ(s.size(), 2);
    EXPECT_EQ(s[0], 2);
    EXPECT_EQ(s[1], 2);
    EXPECT_EQ(c.numel(), a.numel());
}

TEST(TensorNegation, DoubleNegation) {
    Tensor a({1.0f, -2.0f, 3.0f}, {3});
    Tensor c = -(-a);
    EXPECT_TRUE(c == a);
    Tensor b({-1.0f, 2.0f, -3.0f}, {3});
    EXPECT_TRUE((-a) == b);
}

TEST(TensorNegation, MultiDim) {
    Tensor a({1.0f, -2.0f, 3.0f, -4.0f}, {2, 2});
    Tensor c = -a;
    EXPECT_FLOAT_EQ(c.data()[0], -1.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(c.data()[2], -3.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 4.0f);
    EXPECT_TRUE((-a).shape() == a.shape());
}

TEST(TensorExp, BasicValues) {
    Tensor a({0.0f, 1.0f, 2.0f}, {3});
    Tensor c = a.exp();
    EXPECT_FLOAT_EQ(c.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(c.data()[1], std::exp(1.0f));
    EXPECT_FLOAT_EQ(c.data()[2], std::exp(2.0f));
}

TEST(TensorExp, PreservesShape) {
    Tensor a({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2});
    Tensor c = a.exp();
    EXPECT_TRUE(c.shape() == a.shape());
}

TEST(TensorExp, NegativeValues) {
    Tensor a({-1.0f, 0.0f, 1.0f}, {3});
    Tensor c = a.exp();
    EXPECT_FLOAT_EQ(c.data()[0], std::exp(-1.0f));
    EXPECT_FLOAT_EQ(c.data()[1], 1.0f);
    EXPECT_FLOAT_EQ(c.data()[2], std::exp(1.0f));
}

TEST(TensorSqrt, BasicValues) {
    Tensor a({0.0f, 1.0f, 4.0f, 9.0f}, {4});
    Tensor c = a.sqrt();
    EXPECT_FLOAT_EQ(c.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 1.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 2.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 3.0f);
}

TEST(TensorSqrt, PreservesShape) {
    Tensor a({1.0f, 4.0f, 9.0f, 16.0f}, {2, 2});
    Tensor c = a.sqrt();
    EXPECT_TRUE(c.shape() == a.shape());
}

TEST(TensorSqrt, NegativeValues) {
    Tensor a({-4.0f, -1.0f, 0.0f, 1.0f}, {4});
    Tensor c = a.sqrt();
    EXPECT_TRUE(std::isnan(c.data()[0]));
    EXPECT_TRUE(std::isnan(c.data()[1]));
    EXPECT_FLOAT_EQ(c.data()[2], 0.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 1.0f);
}

TEST(TensorReductions, WholeTensorSum) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_FLOAT_EQ(t.sum(), 10.0f);
}

TEST(TensorReductions, WholeTensorMean) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_FLOAT_EQ(t.mean(), 2.5f);
}

TEST(TensorReductions, WholeTensorMax) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_FLOAT_EQ(t.max(), 4.0f);
}

TEST(TensorReductions, WholeTensorMin) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_FLOAT_EQ(t.min(), 1.0f);
}

TEST(TensorReductions, WholeTensorNegativeValues) {
    Tensor t({-1.0f, -2.0f, 3.0f}, {3});
    EXPECT_FLOAT_EQ(t.sum(), 0.0f);
    EXPECT_FLOAT_EQ(t.mean(), 0.0f);
    EXPECT_FLOAT_EQ(t.max(), 3.0f);
    EXPECT_FLOAT_EQ(t.min(), -2.0f);
}

TEST(TensorReductions, WholeTensorSingleElement) {
    Tensor t({42.0f}, {1});
    EXPECT_FLOAT_EQ(t.sum(), 42.0f);
    EXPECT_FLOAT_EQ(t.mean(), 42.0f);
    EXPECT_FLOAT_EQ(t.max(), 42.0f);
    EXPECT_FLOAT_EQ(t.min(), 42.0f);
}

TEST(TensorReductions, WholeTensorMaxMinEmptyThrows) {
    Tensor t(std::vector<int>{0});
    EXPECT_THROW(t.max(), torc::ShapeError);
    EXPECT_THROW(t.min(), torc::ShapeError);
}

TEST(TensorReductions, AxisSum) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.sum(0);
    EXPECT_EQ(c.shape(), std::vector<int>({3}));
    EXPECT_FLOAT_EQ(c.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 7.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 9.0f);
}

TEST(TensorReductions, AxisSumLastDim) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.sum(1);
    EXPECT_EQ(c.shape(), std::vector<int>({2}));
    EXPECT_FLOAT_EQ(c.data()[0], 6.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 15.0f);
}

TEST(TensorReductions, AxisMean) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.mean(0);
    EXPECT_EQ(c.shape(), std::vector<int>({3}));
    EXPECT_FLOAT_EQ(c.data()[0], 2.5f);
    EXPECT_FLOAT_EQ(c.data()[1], 3.5f);
    EXPECT_FLOAT_EQ(c.data()[2], 4.5f);
}

TEST(TensorReductions, AxisMeanLastDim) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.mean(1);
    EXPECT_EQ(c.shape(), std::vector<int>({2}));
    EXPECT_FLOAT_EQ(c.data()[0], 2.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 5.0f);
}

TEST(TensorReductions, AxisMax) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.max(0);
    EXPECT_EQ(c.shape(), std::vector<int>({3}));
    EXPECT_FLOAT_EQ(c.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 6.0f);
}

TEST(TensorReductions, AxisMin) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.min(0);
    EXPECT_EQ(c.shape(), std::vector<int>({3}));
    EXPECT_FLOAT_EQ(c.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 3.0f);
}

TEST(TensorReductions, AxisMaxMinLastDim) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    EXPECT_EQ(t.max(1).shape(), std::vector<int>({2}));
    EXPECT_FLOAT_EQ(t.max(1).data()[0], 3.0f);
    EXPECT_FLOAT_EQ(t.max(1).data()[1], 6.0f);
    EXPECT_EQ(t.min(1).shape(), std::vector<int>({2}));
    EXPECT_FLOAT_EQ(t.min(1).data()[0], 1.0f);
    EXPECT_FLOAT_EQ(t.min(1).data()[1], 4.0f);
}

TEST(TensorReductions, AxisHigherDim) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor c = t.sum(2);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 6.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 15.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 24.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 33.0f);
}

TEST(TensorReductions, AxisInvalidThrows) {
    Tensor t({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(t.sum(-1), torc::ShapeError);
    EXPECT_THROW(t.sum(3), torc::ShapeError);
    EXPECT_THROW(t.mean(-1), torc::ShapeError);
    EXPECT_THROW(t.max(3), torc::ShapeError);
    EXPECT_THROW(t.min(3), torc::ShapeError);
}

TEST(TensorReductions, AxisEmptyThrows) {
    Tensor t(std::vector<int>{0});
    EXPECT_THROW(t.sum(0), torc::ShapeError);
    EXPECT_THROW(t.mean(0), torc::ShapeError);
}

TEST(TensorReshape, ValidReshape) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor r = t.reshape({4});
    EXPECT_EQ(r.shape(), std::vector<int>({4}));
    EXPECT_EQ(r.numel(), 4);
    EXPECT_FLOAT_EQ(r.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(r.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(r.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(r.data()[3], 4.0f);
}

TEST(TensorReshape, PreservesDataOrder) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor r = t.reshape({1, 4});
    EXPECT_EQ(r.shape(), std::vector<int>({1, 4}));
    EXPECT_FLOAT_EQ(r.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(r.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(r.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(r.data()[3], 4.0f);
}

TEST(TensorReshape, MismatchThrows) {
    Tensor t({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(t.reshape({2, 2}), torc::ShapeError);
}

TEST(TensorReshape, SameShape) {
    Tensor t({1.0f, 2.0f}, {2});
    Tensor r = t.reshape({2});
    EXPECT_TRUE(r == t);
}

TEST(TensorView, DelegatesToReshape) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor r = t.view({4});
    Tensor s = t.reshape({4});
    EXPECT_TRUE(r == s);
}

TEST(TensorMove, MoveConstructible) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b = std::move(a);
    EXPECT_EQ(b.shape(), std::vector<int>({3}));
    EXPECT_FLOAT_EQ(b.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(b.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(b.data()[2], 3.0f);
}

TEST(TensorMove, MoveAssignable) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({0.0f}, {1});
    b = std::move(a);
    EXPECT_EQ(b.shape(), std::vector<int>({3}));
    EXPECT_FLOAT_EQ(b.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(b.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(b.data()[2], 3.0f);
}

TEST(TensorCopy, IndependentCopies) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b = a;
    b.data()[0] = 99.0f;
    EXPECT_FLOAT_EQ(a.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(b.data()[0], 99.0f);
}

TEST(TensorBroadcasting, IdenticalShape) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({4.0f, 5.0f, 6.0f}, {3});
    Tensor c = a.add(b);
    EXPECT_EQ(c.shape(), std::vector<int>({3}));
    EXPECT_FLOAT_EQ(c.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 7.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 9.0f);
}

TEST(TensorBroadcasting, DimOneExpansion) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({10.0f, 20.0f}, {2, 1});
    Tensor c = a.add(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 12.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 23.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 24.0f);
}

TEST(TensorBroadcasting, MultiDimBroadcast) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({10.0f, 20.0f}, {1, 2});
    Tensor c = a.add(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 22.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 13.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 24.0f);
}

TEST(TensorBroadcasting, HigherRankBroadcast) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor b({10.0f, 20.0f, 30.0f}, {1, 3});
    Tensor c = a.add(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 3}));
    EXPECT_FLOAT_EQ(c.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 22.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 33.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 14.0f);
    EXPECT_FLOAT_EQ(c.data()[4], 25.0f);
    EXPECT_FLOAT_EQ(c.data()[5], 36.0f);
}

TEST(TensorBroadcasting, IncompatibleShapesThrow) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.add(b), torc::ShapeError);
    EXPECT_THROW(a.sub(b), torc::ShapeError);
    EXPECT_THROW(a.mul(b), torc::ShapeError);
    EXPECT_THROW(a.div(b), torc::ShapeError);
}

TEST(TensorBroadcasting, AllOpsBroadcast) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({10.0f, 20.0f, 30.0f}, {3, 1});
    EXPECT_EQ(a.add(b).shape(), std::vector<int>({3, 3}));
    EXPECT_EQ(a.sub(b).shape(), std::vector<int>({3, 3}));
    EXPECT_EQ(a.mul(b).shape(), std::vector<int>({3, 3}));
    EXPECT_EQ(a.div(b).shape(), std::vector<int>({3, 3}));
    EXPECT_FLOAT_EQ(a.add(b).data()[0], 11.0f);
    EXPECT_FLOAT_EQ(a.sub(b).data()[1], -8.0f);
    EXPECT_FLOAT_EQ(a.mul(b).data()[2], 30.0f);
    EXPECT_FLOAT_EQ(a.div(b).data()[3], 0.05f);
}

TEST(TensorIndexing, ReadWrite) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_FLOAT_EQ(static_cast<float>(t[0, 0]), 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t[1, 1]), 4.0f);
    t[0, 1] = 99.0f;
    EXPECT_FLOAT_EQ(t.data()[1], 99.0f);
}

TEST(TensorIndexing, OutOfBoundsThrow) {
    Tensor t({2, 2});
    EXPECT_THROW(static_cast<void>(t[2, 0]), torc::ShapeError);
    EXPECT_THROW(static_cast<void>(t[0, 2]), torc::ShapeError);
    EXPECT_THROW(static_cast<void>(t[-1, 0]), torc::ShapeError);
}

TEST(TensorIndexing, WrongRankThrow) {
    Tensor t({2, 2});
    EXPECT_THROW(static_cast<void>(t[0]), torc::ShapeError);
    EXPECT_THROW(static_cast<void>(t[0, 0, 0]), torc::ShapeError);
}

TEST(TensorIndexing, MultiDim) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    EXPECT_FLOAT_EQ(static_cast<float>(t[0, 2]), 3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t[1, 0]), 4.0f);
}

TEST(TensorIndexing, ConstAccess) {
    const Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_FLOAT_EQ(static_cast<float>(t[1, 0]), 3.0f);
}

TEST(TensorTranspose, Default2D) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.transpose({});
    EXPECT_EQ(c.shape(), std::vector<int>({3, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 4.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 2.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 5.0f);
    EXPECT_FLOAT_EQ(c.data()[4], 3.0f);
    EXPECT_FLOAT_EQ(c.data()[5], 6.0f);
}

TEST(TensorTranspose, Default3D) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor c = t.transpose({});
    EXPECT_EQ(c.shape(), std::vector<int>({3, 2, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 7.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 4.0f);
}

TEST(TensorTranspose, CustomAxes) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor c = t.transpose({0, 2, 1});
    EXPECT_EQ(c.shape(), std::vector<int>({2, 3, 2}));
}

TEST(TensorTranspose, InvalidAxesThrow) {
    Tensor t({2, 3});
    EXPECT_THROW(t.transpose({0}), torc::ShapeError);
    EXPECT_THROW(t.transpose({0, 0}), torc::ShapeError);
    EXPECT_THROW(t.transpose({0, 2}), torc::ShapeError);
}

TEST(TensorTranspose, PreservesValues) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = t.transpose({});
    EXPECT_TRUE(c.shape() == std::vector<int>({3, 2}));
    for (int i = 0; i < t.numel(); ++i)
        EXPECT_FLOAT_EQ(c.sum(), t.sum());
}

TEST(TensorSlice, FirstDimSlice) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor s = t.slice({Tensor::Slice{0, 1}, Tensor::Slice{0, 3}});
    EXPECT_EQ(s.shape(), std::vector<int>({1, 3}));
    EXPECT_FLOAT_EQ(s.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(s.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(s.data()[2], 3.0f);
}

TEST(TensorSlice, LastDimSlice) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor s = t.slice({Tensor::Slice{0, 2}, Tensor::Slice{1, 2}});
    EXPECT_EQ(s.shape(), std::vector<int>({2, 1}));
    EXPECT_FLOAT_EQ(s.data()[0], 2.0f);
    EXPECT_FLOAT_EQ(s.data()[1], 5.0f);
}

TEST(TensorSlice, InnerDimSlice) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor s = t.slice({Tensor::Slice{0, 2}, Tensor::Slice{0, 2}});
    EXPECT_EQ(s.shape(), std::vector<int>({2, 2}));
    EXPECT_FLOAT_EQ(s.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(s.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(s.data()[2], 4.0f);
    EXPECT_FLOAT_EQ(s.data()[3], 5.0f);
}

TEST(TensorSlice, CombinedSlice) {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor s = t.slice({Tensor::Slice{0, 2}, Tensor::Slice{1, 2}, Tensor::Slice{0, 2}});
    EXPECT_EQ(s.shape(), std::vector<int>({2, 1, 2}));
    EXPECT_FLOAT_EQ(s.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(s.data()[1], 5.0f);
    EXPECT_FLOAT_EQ(s.data()[2], 10.0f);
    EXPECT_FLOAT_EQ(s.data()[3], 11.0f);
}

TEST(TensorSlice, OutOfRangeThrow) {
    Tensor t({2, 3});
    EXPECT_THROW(t.slice({Tensor::Slice{0, 3}, Tensor::Slice{0, 3}}), torc::ShapeError);
    EXPECT_THROW(t.slice({Tensor::Slice{0, 2}, Tensor::Slice{-1, 2}}), torc::ShapeError);
    EXPECT_THROW(t.slice({Tensor::Slice{1, 0}, Tensor::Slice{0, 2}}), torc::ShapeError);
}

TEST(TensorSlice, WrongRankThrow) {
    Tensor t({2, 3});
    EXPECT_THROW(t.slice({Tensor::Slice{0, 1}}), torc::ShapeError);
    EXPECT_THROW(t.slice({Tensor::Slice{0, 2}, Tensor::Slice{0, 2}, Tensor::Slice{0, 2}}), torc::ShapeError);
}

TEST(TensorMatmul, Square2D) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 19.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 22.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 43.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 50.0f);
}

TEST(TensorMatmul, NonSquare2D) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 2}));
    EXPECT_FLOAT_EQ(c.data()[0], 58.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 64.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 139.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 154.0f);
}

TEST(TensorMatmul, IdentityLeftAndRight) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor il({1.0f, 0.0f, 0.0f, 1.0f}, {2, 2});
    Tensor ir({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {3, 3});
    EXPECT_TRUE(il.matmul(a) == a);
    EXPECT_TRUE(a.matmul(ir) == a);
}

TEST(TensorMatmul, OutputShapeOnly) {
    Tensor a({4, 3});
    Tensor b({3, 5});
    EXPECT_EQ(a.matmul(b).shape(), std::vector<int>({4, 5}));
}

TEST(TensorMatmul, InnerDimensionMismatchThrows) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor b({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_THROW(a.matmul(b), torc::ShapeError);
}

TEST(TensorMatmul, RankOneThrows) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({1.0f, 2.0f}, {2, 1});
    Tensor c({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    EXPECT_THROW(a.matmul(b), torc::ShapeError);
    EXPECT_THROW(b.matmul(a), torc::ShapeError);
    EXPECT_THROW(a.matmul(c), torc::ShapeError);
    EXPECT_THROW(c.matmul(a), torc::ShapeError);
}

TEST(TensorMatmul, MismatchMessageNamesBothShapes) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor b({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    try {
        a.matmul(b);
        FAIL() << "expected ShapeError";
    } catch (const torc::ShapeError& e) {
        std::string msg(e.what());
        EXPECT_TRUE(msg.find("(2, 3)") != std::string::npos);
        EXPECT_TRUE(msg.find("(2, 2)") != std::string::npos);
    }
}

TEST(TensorMatmul, TransposeProperty) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    Tensor lhs = a.matmul(b).transpose({});
    Tensor rhs = b.transpose({}).matmul(a.transpose({}));
    ASSERT_EQ(lhs.numel(), rhs.numel());
    for (int i = 0; i < lhs.numel(); ++i)
        EXPECT_NEAR(lhs.data()[i], rhs.data()[i], 1e-5f);
}

TEST(TensorMatmul, DistributiveProperty) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    Tensor c({9.0f, 10.0f, 11.0f, 12.0f}, {2, 2});
    Tensor lhs = a.matmul(b.add(c));
    Tensor rhs = a.matmul(b).add(a.matmul(c));
    ASSERT_EQ(lhs.numel(), rhs.numel());
    for (int i = 0; i < lhs.numel(); ++i)
        EXPECT_NEAR(lhs.data()[i], rhs.data()[i], 1e-5f);
}

TEST(TensorMatmul, TransposeOperandInterop) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor c = a.transpose({}).matmul(a);
    EXPECT_EQ(c.shape(), std::vector<int>({3, 3}));
    EXPECT_FLOAT_EQ(c.data()[0], 1.0f * 1.0f + 4.0f * 4.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 1.0f * 2.0f + 4.0f * 5.0f);
}

TEST(TensorMatmulBatched, DistinctPerBatch) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor b({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 3, 2});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 2, 2}));
    // batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
    EXPECT_FLOAT_EQ(c.data()[0], 22.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 28.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 49.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 64.0f);
    // batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]]
    EXPECT_FLOAT_EQ(c.data()[4], 220.0f);
    EXPECT_FLOAT_EQ(c.data()[5], 244.0f);
    EXPECT_FLOAT_EQ(c.data()[6], 301.0f);
    EXPECT_FLOAT_EQ(c.data()[7], 334.0f);
}

TEST(TensorMatmulBatched, BroadcastRankPromotion) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 2, 2}));
    Tensor flat = a.matmul(b.reshape({1, 3, 2}));
    EXPECT_TRUE(c == flat);
}

TEST(TensorMatmulBatched, BroadcastSizeOne) {
    // a: (2,1,2,3) -> batch (2,1), each a matrix is (2,3)
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 1, 2, 3});
    // b: (1,3,3,2) -> batch (1,3), each b matrix is (3,2)
    Tensor b({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f}, {1, 3, 3, 2});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 3, 2, 2}));

    // With batch broadcasting, result block (bi,bj) = a_block[bi] @ b_block[bj] (independent).
    // block (0,0): a0=[[1,2,3],[4,5,6]] @ b0=[[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    EXPECT_FLOAT_EQ(c.data()[0], 22.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 28.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 49.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 64.0f);
    // block (0,1): a0 @ b1=[[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    EXPECT_FLOAT_EQ(c.data()[4], 58.0f);
    EXPECT_FLOAT_EQ(c.data()[5], 64.0f);
    EXPECT_FLOAT_EQ(c.data()[6], 139.0f);
    EXPECT_FLOAT_EQ(c.data()[7], 154.0f);
    // block (1,2): a1=[[7,8,9],[10,11,12]] @ b2=[[13,14],[15,16],[17,18]] = [[364,388],[499,532]]
    EXPECT_FLOAT_EQ(c.data()[20], 364.0f);
    EXPECT_FLOAT_EQ(c.data()[21], 388.0f);
    EXPECT_FLOAT_EQ(c.data()[22], 499.0f);
    EXPECT_FLOAT_EQ(c.data()[23], 532.0f);
}

TEST(TensorMatmulBatched, IncompatibleBatchThrows) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2, 1});
    Tensor b({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 1, 2});
    try {
        a.matmul(b);
        FAIL() << "expected ShapeError";
    } catch (const torc::ShapeError& e) {
        std::string msg(e.what());
        EXPECT_TRUE(msg.find("(2, 2, 1)") != std::string::npos);
        EXPECT_TRUE(msg.find("(3, 1, 2)") != std::string::npos);
    }
}

TEST(TensorMatmul, ZeroSizeDimensions) {
    Tensor a({2, 0});
    Tensor b({0, 3});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 3}));
    for (int i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(c.data()[i], 0.0f);

    Tensor p({0, 3});
    Tensor q({3, 4});
    Tensor r = p.matmul(q);
    EXPECT_EQ(r.shape(), std::vector<int>({0, 4}));
    EXPECT_EQ(r.numel(), 0);
}

#ifdef TORC_USE_BLAS
TEST(TensorMatmulBLAS, MatchesNaive2D) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    Tensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Tensor c_blas = a.matmul(b);
    
    // Re-run with naive by temporarily disabling BLAS (not possible at runtime)
    // Instead, compute expected values manually
    Tensor expected({58.0f, 64.0f, 139.0f, 154.0f}, {2, 2});
    EXPECT_TRUE(c_blas == expected);
}

TEST(TensorMatmulBLAS, MatchesNaiveBatched) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor b({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 3, 2});
    Tensor c_blas = a.matmul(b);
    
    Tensor expected({22.0f, 28.0f, 49.0f, 64.0f, 220.0f, 244.0f, 301.0f, 334.0f}, {2, 2, 2});
    EXPECT_TRUE(c_blas == expected);
}

TEST(TensorMatmulBLAS, MatchesNaiveBroadcastBatch) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {2, 2, 3});
    Tensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});
    Tensor c_blas = a.matmul(b);
    
    Tensor expected({58.0f, 64.0f, 139.0f, 154.0f, 58.0f, 64.0f, 139.0f, 154.0f}, {2, 2, 2});
    EXPECT_TRUE(c_blas == expected);
}

TEST(TensorMatmulBLAS, ZeroSizeDimensions) {
    Tensor a({2, 0});
    Tensor b({0, 3});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), std::vector<int>({2, 3}));
    for (int i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(c.data()[i], 0.0f);
}
#endif // TORC_USE_BLAS

