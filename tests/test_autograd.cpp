#include <gtest/gtest.h>
#include "torc/autograd.hpp"
#include <cmath>
#include <string>

using torc::Variable;
using torc::Tensor;
using torc::ShapeError;

static constexpr float EPS = 1e-4f;

static void expect_near(float actual, float expected, float tol = EPS) {
    EXPECT_NEAR(actual, expected, tol)
        << "actual=" << actual << " expected=" << expected;
}

// ============================================================
// Step 1: Variable scaffold tests
// ============================================================

TEST(VariableScaffold, ConstructFromTensor) {
    Tensor t({3.0f}, std::vector<int>{1});
    Variable v(t, true);
    EXPECT_TRUE(v.requires_grad());
    EXPECT_FLOAT_EQ(v.data().data()[0], 3.0f);
}

TEST(VariableScaffold, ConstructFromScalar) {
    Variable v(4.0f, true);
    EXPECT_TRUE(v.requires_grad());
    EXPECT_FLOAT_EQ(v.data().data()[0], 4.0f);
    EXPECT_EQ(v.data().shape(), std::vector<int>({1}));
}

TEST(VariableScaffold, RequiresGradFalse) {
    Tensor t({1.0f}, std::vector<int>{1});
    Variable v(t, false);
    EXPECT_FALSE(v.requires_grad());
}

TEST(VariableScaffold, HasGradInitiallyFalse) {
    Variable v(3.0f, true);
    EXPECT_FALSE(v.has_grad());
}

TEST(VariableScaffold, ZeroGradResetsHasGrad) {
    Variable v(3.0f, true);
    v.backward();
    EXPECT_FALSE(v.has_grad());
    v.zero_grad();
    EXPECT_FALSE(v.has_grad());
}

TEST(VariableScaffold, BackwardNoOpWhenNotRequiresGrad) {
    Variable v(3.0f, false);
    v.backward();
    EXPECT_FALSE(v.has_grad());
}

TEST(VariableScaffold, BackwardNonScalarThrows) {
    Tensor t({1.0f, 2.0f}, std::vector<int>{2});
    Variable v(t, true);
    EXPECT_THROW(v.backward(), ShapeError);
}

TEST(VariableScaffold, BackwardWithGradOutputShapeMismatchThrows) {
    Tensor data({1.0f}, std::vector<int>{1});
    Variable v(data, true);
    Tensor bad_grad({1.0f, 2.0f}, std::vector<int>{2});
    EXPECT_THROW(v.backward(bad_grad), ShapeError);
}

TEST(VariableScaffold, BackwardOnEmptyTapeDoesNothing) {
    Variable v(3.0f, true);
    v.backward();
    EXPECT_FALSE(v.has_grad());
}

// ============================================================
// Step 2: Scalar-only autograd tests
// ============================================================

// z = a + b, dz/da = 1, dz/db = 1
TEST(ScalarAutograd, AddBackward) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable c = torc::add_scalar(a, b);
    EXPECT_FLOAT_EQ(c.data().data()[0], 7.0f);
    c.backward();
    EXPECT_TRUE(a.has_grad());
    EXPECT_TRUE(b.has_grad());
    expect_near(a.grad().data()[0], 1.0f);
    expect_near(b.grad().data()[0], 1.0f);
}

// z = a - b, dz/da = 1, dz/db = -1
TEST(ScalarAutograd, SubBackward) {
    Variable a(5.0f, true);
    Variable b(3.0f, true);
    Variable c = torc::sub_scalar(a, b);
    EXPECT_FLOAT_EQ(c.data().data()[0], 2.0f);
    c.backward();
    expect_near(a.grad().data()[0], 1.0f);
    expect_near(b.grad().data()[0], -1.0f);
}

// z = a * b, dz/da = b, dz/db = a
TEST(ScalarAutograd, MulBackward) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable c = torc::mul_scalar(a, b);
    EXPECT_FLOAT_EQ(c.data().data()[0], 12.0f);
    c.backward();
    expect_near(a.grad().data()[0], 4.0f);
    expect_near(b.grad().data()[0], 3.0f);
}

// z = a / b, dz/da = 1/b, dz/db = -a/b^2
TEST(ScalarAutograd, DivBackward) {
    Variable a(6.0f, true);
    Variable b(2.0f, true);
    Variable c = torc::div_scalar(a, b);
    EXPECT_FLOAT_EQ(c.data().data()[0], 3.0f);
    c.backward();
    expect_near(a.grad().data()[0], 0.5f);
    expect_near(b.grad().data()[0], -1.5f);
}

// z = -a, dz/da = -1
TEST(ScalarAutograd, NegBackward) {
    Variable a(3.0f, true);
    Variable c = torc::neg_scalar(a);
    EXPECT_FLOAT_EQ(c.data().data()[0], -3.0f);
    c.backward();
    expect_near(a.grad().data()[0], -1.0f);
}

// Chain: c = (a + b) * a
// dc/da = (a+b)*1 + 1*a = 2a + b = 10, dc/db = a = 3
TEST(ScalarAutograd, ChainAddThenMul) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable s = torc::add_scalar(a, b);
    Variable c = torc::mul_scalar(s, a);
    EXPECT_FLOAT_EQ(c.data().data()[0], 21.0f);
    c.backward();
    expect_near(a.grad().data()[0], 10.0f);
    expect_near(b.grad().data()[0], 3.0f);
}

// Chain: c = (a * b) + a
// dc/da = b + 1 = 5, dc/db = a = 3
TEST(ScalarAutograd, ChainMulThenAdd) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable p = torc::mul_scalar(a, b);
    Variable c = torc::add_scalar(p, a);
    EXPECT_FLOAT_EQ(c.data().data()[0], 15.0f);
    c.backward();
    expect_near(a.grad().data()[0], 5.0f);
    expect_near(b.grad().data()[0], 3.0f);
}

// Gradient accumulation via += on repeated backward
TEST(ScalarAutograd, GradientAccumulation) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable c = torc::mul_scalar(a, b);
    c.backward();
    expect_near(a.grad().data()[0], 4.0f);
    Variable c2 = torc::mul_scalar(a, b);
    c2.backward();
    expect_near(a.grad().data()[0], 8.0f);
}

// zero_grad() resets accumulation
TEST(ScalarAutograd, ZeroGradResetsAccumulation) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable c = torc::mul_scalar(a, b);
    c.backward();
    expect_near(a.grad().data()[0], 4.0f);
    a.zero_grad();
    EXPECT_FALSE(a.has_grad());
    Variable c2 = torc::mul_scalar(a, b);
    c2.backward();
    expect_near(a.grad().data()[0], 4.0f);
}

// Non-tracked input gets no grad allocated
TEST(ScalarAutograd, NonTrackedInputNoGrad) {
    Variable a(3.0f, true);
    Variable b(4.0f, false);
    Variable c = torc::add_scalar(a, b);
    c.backward();
    EXPECT_TRUE(a.has_grad());
    expect_near(a.grad().data()[0], 1.0f);
    EXPECT_FALSE(b.has_grad());
}

// Both inputs non-tracked: output tape is empty
TEST(ScalarAutograd, BothNonTrackedEmptyTape) {
    Variable a(3.0f, false);
    Variable b(4.0f, false);
    Variable c = torc::add_scalar(a, b);
    EXPECT_FALSE(c.requires_grad());
    EXPECT_TRUE(c.tape_.empty());
}

// Tape consumed after backward
TEST(ScalarAutograd, TapeConsumedAfterBackward) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable c = torc::mul_scalar(a, b);
    EXPECT_FALSE(c.tape_.empty());
    c.backward();
    EXPECT_TRUE(c.tape_.empty());
    expect_near(a.grad().data()[0], 4.0f);
    expect_near(b.grad().data()[0], 3.0f);
    c.backward(); // second call: tape empty, grads unchanged
    expect_near(a.grad().data()[0], 4.0f);
    expect_near(b.grad().data()[0], 3.0f);
}

// Hand-computed: f(x) = 2*x + 1, df/dx = 2
TEST(ScalarAutograd, HandComputedLinear) {
    Variable x(5.0f, true);
    Variable two(2.0f, true);
    Variable one(1.0f, false);
    Variable p = torc::mul_scalar(two, x);
    Variable f = torc::add_scalar(p, one);
    f.backward();
    expect_near(x.grad().data()[0], 2.0f);
    expect_near(two.grad().data()[0], 5.0f);
}

// Hand-computed: f(x,y) = x^2 + y^3 at x=2, y=3
TEST(ScalarAutograd, HandComputedPolynomial) {
    Variable x(2.0f, true);
    Variable y(3.0f, true);
    Variable x2 = torc::mul_scalar(x, x);
    Variable y2 = torc::mul_scalar(y, y);
    Variable y3 = torc::mul_scalar(y2, y);
    Variable f = torc::add_scalar(x2, y3);
    EXPECT_FLOAT_EQ(f.data().data()[0], 31.0f);
    f.backward();
    expect_near(x.grad().data()[0], 4.0f);
    expect_near(y.grad().data()[0], 27.0f);
}
