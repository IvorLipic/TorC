#include <gtest/gtest.h>
#include "torc/tensor.hpp"

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
    EXPECT_THROW(Tensor({1.0f, 2.0f}, {3}), std::runtime_error);
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

TEST(TensorOps, MulIdenticalShape) {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({4.0f, 5.0f, 6.0f}, {3});
    Tensor c = a.mul(b);
    EXPECT_FLOAT_EQ(c.data()[0], 4.0f);
    EXPECT_FLOAT_EQ(c.data()[1], 10.0f);
    EXPECT_FLOAT_EQ(c.data()[2], 18.0f);
}

TEST(TensorOps, AddShapeMismatchThrows) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.add(b), std::runtime_error);
}

TEST(TensorOps, MulShapeMismatchThrows) {
    Tensor a({1.0f, 2.0f}, {2});
    Tensor b({1.0f, 2.0f, 3.0f}, {3});
    EXPECT_THROW(a.mul(b), std::runtime_error);
}

TEST(TensorOps, MultiDimElementwise) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Tensor b({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2});
    Tensor c = a.add(b);
    EXPECT_FLOAT_EQ(c.data()[0], 11.0f);
    EXPECT_FLOAT_EQ(c.data()[3], 44.0f);
}
