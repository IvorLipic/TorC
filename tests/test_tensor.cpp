#include <gtest/gtest.h>
#include "torc/tensor.hpp"
#include "torc/utils.hpp"
#include <sstream>
#include <string>

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
