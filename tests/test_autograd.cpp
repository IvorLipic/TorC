#include <gtest/gtest.h>
#include "torc/autograd.hpp"
#include <cmath>
#include <string>
#include <utility>

using torc::Variable;
using torc::Tensor;
using torc::ShapeError;
using torc::TorcError;

static constexpr float GRAD_ATOL = 1e-2f;

static void expect_near(float actual, float expected, float tol = GRAD_ATOL) {
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
    EXPECT_TRUE(c.tape().empty());
}

// Tape consumed after backward
TEST(ScalarAutograd, TapeConsumedAfterBackward) {
    Variable a(3.0f, true);
    Variable b(4.0f, true);
    Variable c = torc::mul_scalar(a, b);
    EXPECT_FALSE(c.tape().empty());
    c.backward();
    EXPECT_TRUE(c.tape().empty());
    expect_near(a.grad().data()[0], 4.0f);
    expect_near(b.grad().data()[0], 3.0f);
    c.backward(); // second call: tape empty, grads unchanged
    expect_near(a.grad().data()[0], 4.0f);
    expect_near(b.grad().data()[0], 3.0f);
}

TEST(GraphLifetime, ExpiredIntermediateFailsBeforeDereference) {
    Variable x(2.0f, true);
    Variable output(0.0f, true);
    {
        Variable intermediate = torc::mul_scalar(x, Variable(3.0f, false));
        output = torc::add_scalar(intermediate, x);
    }

    EXPECT_THROW(output.backward(), torc::TorcError);
    EXPECT_FALSE(x.has_grad());
}

TEST(GraphLifetime, MoveAssignmentInvalidatesMovedFromVariable) {
    Variable x(2.0f, true);
    Variable intermediate = torc::mul_scalar(x, Variable(3.0f, false));
    Variable output = torc::add_scalar(intermediate, x);

    Variable moved_target(0.0f, false);
    moved_target = std::move(intermediate);

    EXPECT_THROW(output.backward(), torc::TorcError);
    EXPECT_FALSE(x.has_grad());
}

TEST(GraphLifetime, MoveConstructionInvalidatesMovedFromVariable) {
    Variable x(2.0f, true);
    Variable intermediate = torc::mul_scalar(x, Variable(3.0f, false));
    Variable output = torc::add_scalar(intermediate, x);

    Variable moved_target(std::move(intermediate));

    EXPECT_THROW(output.backward(), torc::TorcError);
    EXPECT_FALSE(x.has_grad());
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

// ============================================================
// Step 3: Tensor-tensor elementwise + broadcasting backward
// ============================================================

static Tensor numerical_grad_add(const Tensor& a, const Tensor& b) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = (a_ph.add(b)).sum();
        float f_mh = (a_mh.add(b)).sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_mul(const Tensor& a, const Tensor& b) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = (a_ph.mul(b)).sum();
        float f_mh = (a_mh.mul(b)).sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

TEST(TensorElementwiseAutograd, AddIdenticalShape) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Tensor b_data({4.0f, 5.0f, 6.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::add(a, b);
    EXPECT_TRUE(c.data() == a.data().add(b.data()));
    Tensor ones({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    c.backward(ones);
    EXPECT_TRUE(a.has_grad());
    EXPECT_TRUE(b.has_grad());
    Tensor expected_grad({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected_grad);
    EXPECT_TRUE(b.grad() == expected_grad);
}

TEST(TensorElementwiseAutograd, AddBroadcastDimOne) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Tensor b_data({10.0f, 20.0f}, std::vector<int>{2, 1});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::add(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor expected_a_grad({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    Tensor expected_b_grad({2.0f, 2.0f}, std::vector<int>{2, 1});
    EXPECT_TRUE(a.grad() == expected_a_grad);
    EXPECT_TRUE(b.grad() == expected_b_grad);
}

TEST(TensorElementwiseAutograd, AddBroadcastMultiDim) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Tensor b_data({10.0f, 20.0f}, std::vector<int>{1, 2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::add(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor expected_a_grad({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    Tensor expected_b_grad({2.0f, 2.0f}, std::vector<int>{1, 2});
    EXPECT_TRUE(a.grad() == expected_a_grad);
    EXPECT_TRUE(b.grad() == expected_b_grad);
}

TEST(TensorElementwiseAutograd, MulIdenticalShape) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Tensor b_data({4.0f, 5.0f, 6.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::mul(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    c.backward(ones);
    Tensor expected_a_grad({4.0f, 5.0f, 6.0f}, std::vector<int>{3});
    Tensor expected_b_grad({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected_a_grad);
    EXPECT_TRUE(b.grad() == expected_b_grad);
}

TEST(TensorElementwiseAutograd, MulBroadcast) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Tensor b_data({2.0f, 3.0f}, std::vector<int>{2, 1});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::mul(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor expected_a_grad({2.0f, 2.0f, 3.0f, 3.0f}, std::vector<int>{2, 2});
    Tensor expected_b_grad({3.0f, 7.0f}, std::vector<int>{2, 1});
    EXPECT_TRUE(a.grad() == expected_a_grad);
    EXPECT_TRUE(b.grad() == expected_b_grad);
}

TEST(TensorElementwiseAutograd, SubIdenticalShape) {
    Tensor a_data({10.0f, 20.0f, 30.0f}, std::vector<int>{3});
    Tensor b_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::sub(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    c.backward(ones);
    Tensor expected_a_grad({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    Tensor expected_b_grad({-1.0f, -1.0f, -1.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected_a_grad);
    EXPECT_TRUE(b.grad() == expected_b_grad);
}

TEST(TensorElementwiseAutograd, DivIdenticalShape) {
    Tensor a_data({10.0f, 20.0f, 30.0f}, std::vector<int>{3});
    Tensor b_data({2.0f, 4.0f, 5.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::div(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    c.backward(ones);
    Tensor expected_a_grad({0.5f, 0.25f, 0.2f}, std::vector<int>{3});
    Tensor expected_b_grad({-2.5f, -1.25f, -1.2f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected_a_grad);
    EXPECT_TRUE(b.grad() == expected_b_grad);
}

TEST(TensorElementwiseAutograd, NegTensor) {
    Tensor a_data({1.0f, -2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::neg(a);
    Tensor ones({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    c.backward(ones);
    Tensor expected_grad({-1.0f, -1.0f, -1.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected_grad);
}

TEST(TensorElementwiseAutograd, NonTrackedInputNoGrad) {
    Tensor a_data({1.0f, 2.0f}, std::vector<int>{2});
    Tensor b_data({3.0f, 4.0f}, std::vector<int>{2});
    Variable a(a_data, true);
    Variable b(b_data, false);
    Variable c = torc::add(a, b);
    Tensor ones({1.0f, 1.0f}, std::vector<int>{2});
    c.backward(ones);
    EXPECT_TRUE(a.has_grad());
    Tensor expected_a_grad({1.0f, 1.0f}, std::vector<int>{2});
    EXPECT_TRUE(a.grad() == expected_a_grad);
    EXPECT_FALSE(b.has_grad());
}

TEST(TensorElementwiseAutograd, GradCheckAddBroadcast) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Tensor b_data({10.0f, 20.0f}, std::vector<int>{2, 1});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::add(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_add(a_data, b_data);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(TensorElementwiseAutograd, GradCheckMulBroadcast) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Tensor b_data({2.0f, 3.0f}, std::vector<int>{2, 1});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::mul(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_mul(a_data, b_data);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

// ============================================================
// Unary transcendental ops backward (sqrt, log)
// ============================================================

static Tensor numerical_grad_sqrt(const Tensor& a) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = a_ph.sqrt().sum();
        float f_mh = a_mh.sqrt().sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_log(const Tensor& a) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = a_ph.log().sum();
        float f_mh = a_mh.log().sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

TEST(UnaryAutograd, SqrtBackwardMatchesChainRule) {
    Tensor a_data({1.0f, 4.0f, 9.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::sqrt(a);
    Variable loss = torc::sum(c);
    loss.backward();
    ASSERT_TRUE(a.has_grad());
    expect_near(a.grad().data()[0], 0.5f);
    expect_near(a.grad().data()[1], 0.25f);
    expect_near(a.grad().data()[2], 1.0f / 6.0f);
}

TEST(UnaryAutograd, SqrtGradCheck) {
    Tensor a_data({1.0f, 4.0f, 9.0f, 16.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::sqrt(a);
    Variable loss = torc::sum(c);
    loss.backward();
    Tensor num_grad = numerical_grad_sqrt(a_data);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(UnaryAutograd, LogBackwardMatchesChainRule) {
    Tensor a_data({1.0f, 2.0f, 4.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::log(a);
    Variable loss = torc::sum(c);
    loss.backward();
    ASSERT_TRUE(a.has_grad());
    expect_near(a.grad().data()[0], 1.0f);
    expect_near(a.grad().data()[1], 0.5f);
    expect_near(a.grad().data()[2], 0.25f);
}

TEST(UnaryAutograd, LogGradCheck) {
    Tensor a_data({1.0f, 2.0f, 4.0f, 8.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::log(a);
    Variable loss = torc::sum(c);
    loss.backward();
    Tensor num_grad = numerical_grad_log(a_data);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

// ============================================================
// Step 4: Reduction ops backward (sum / mean)
// ============================================================

static Tensor numerical_grad_sum(const Tensor& a, int axis) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = axis < 0 ? a_ph.sum() : a_ph.sum(axis).sum();
        float f_mh = axis < 0 ? a_mh.sum() : a_mh.sum(axis).sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_mean(const Tensor& a, int axis) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = axis < 0 ? a_ph.mean() : a_ph.mean(axis).sum();
        float f_mh = axis < 0 ? a_mh.mean() : a_mh.mean(axis).sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

TEST(ReductionAutograd, SumWholeTensorBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::sum(a);
    EXPECT_FLOAT_EQ(c.data().data()[0], 6.0f);
    Tensor grad({2.0f}, std::vector<int>{1});
    c.backward(grad);
    Tensor expected({2.0f, 2.0f, 2.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ReductionAutograd, SumAxis0Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::sum(a, 0);
    EXPECT_TRUE(c.data() == Tensor({4.0f, 6.0f}, std::vector<int>{2}));
    Tensor grad({1.0f, 2.0f}, std::vector<int>{2});
    c.backward(grad);
    Tensor expected({1.0f, 2.0f, 1.0f, 2.0f}, std::vector<int>{2, 2});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ReductionAutograd, SumAxis1Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::sum(a, 1);
    EXPECT_TRUE(c.data() == Tensor({3.0f, 7.0f}, std::vector<int>{2}));
    Tensor grad({1.0f, 2.0f}, std::vector<int>{2});
    c.backward(grad);
    Tensor expected({1.0f, 1.0f, 2.0f, 2.0f}, std::vector<int>{2, 2});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ReductionAutograd, SumAxis1Backward3D) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3, 1});
    Variable a(a_data, true);
    Variable c = torc::sum(a, 1);
    EXPECT_TRUE(c.data() == Tensor({6.0f, 15.0f}, std::vector<int>{2, 1}));
    Tensor grad({1.0f, 2.0f}, std::vector<int>{2, 1});
    c.backward(grad);
    Tensor expected({1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}, std::vector<int>{2, 3, 1});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ReductionAutograd, MeanWholeTensorBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::mean(a);
    EXPECT_FLOAT_EQ(c.data().data()[0], 2.0f);
    Tensor grad({2.0f}, std::vector<int>{1});
    c.backward(grad);
    Tensor expected({2.0f/3.0f, 2.0f/3.0f, 2.0f/3.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ReductionAutograd, MeanAxis0Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::mean(a, 0);
    EXPECT_TRUE(c.data() == Tensor({2.0f, 3.0f}, std::vector<int>{2}));
    Tensor grad({1.0f, 2.0f}, std::vector<int>{2});
    c.backward(grad);
    Tensor expected({0.5f, 1.0f, 0.5f, 1.0f}, std::vector<int>{2, 2});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ReductionAutograd, MeanAxis1Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::mean(a, 1);
    EXPECT_TRUE(c.data() == Tensor({1.5f, 3.5f}, std::vector<int>{2}));
    Tensor grad({1.0f, 2.0f}, std::vector<int>{2});
    c.backward(grad);
    Tensor expected({0.5f, 0.5f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ReductionAutograd, SumBackwardNonTrackedNoGrad) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, false);
    Variable c = torc::sum(a);
    EXPECT_FALSE(c.requires_grad());
    EXPECT_TRUE(c.tape().empty());
}

TEST(ReductionAutograd, GradCheckSumWholeTensor) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::sum(a);
    c.backward();
    Tensor num_grad = numerical_grad_sum(a_data, -1);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(ReductionAutograd, GradCheckSumAxis0) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::sum(a, 0);
    Tensor ones({1.0f, 1.0f}, std::vector<int>{2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_sum(a_data, 0);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(ReductionAutograd, GradCheckSumAxis1) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::sum(a, 1);
    Tensor ones({1.0f, 1.0f}, std::vector<int>{2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_sum(a_data, 1);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(ReductionAutograd, GradCheckMeanWholeTensor) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::mean(a);
    c.backward();
    Tensor num_grad = numerical_grad_mean(a_data, -1);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(ReductionAutograd, GradCheckMeanAxis0) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::mean(a, 0);
    Tensor ones({1.0f, 1.0f}, std::vector<int>{2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_mean(a_data, 0);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(ReductionAutograd, GradCheckMeanAxis1) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::mean(a, 1);
    Tensor ones({1.0f, 1.0f}, std::vector<int>{2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_mean(a_data, 1);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

// ============================================================
// Step 9: max/min backward with argmax tracking
// ============================================================

static Tensor numerical_grad_max(const Tensor& a, int axis) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        float f_ph = axis < 0 ? a_ph.max() : a_ph.max(axis).sum();
        float f = axis < 0 ? a.max() : a.max(axis).sum();
        g.data()[i] = (f_ph - f) / h;
    }
    return g;
}

static Tensor numerical_grad_min(const Tensor& a, int axis) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        float f_ph = axis < 0 ? a_ph.min() : a_ph.min(axis).sum();
        float f = axis < 0 ? a.min() : a.min(axis).sum();
        g.data()[i] = (f_ph - f) / h;
    }
    return g;
}

TEST(MaxMinAutograd, MaxWholeTensorBackward) {
    Tensor a_data({1.0f, 3.0f, 2.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::max(a);
    EXPECT_FLOAT_EQ(c.data().data()[0], 3.0f);
    Tensor grad({2.0f}, std::vector<int>{1});
    c.backward(grad);
    Tensor expected({0.0f, 2.0f, 0.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(MaxMinAutograd, MinWholeTensorBackward) {
    Tensor a_data({1.0f, 3.0f, 2.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable c = torc::min(a);
    EXPECT_FLOAT_EQ(c.data().data()[0], 1.0f);
    Tensor grad({2.0f}, std::vector<int>{1});
    c.backward(grad);
    Tensor expected({2.0f, 0.0f, 0.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(MaxMinAutograd, MaxAxis0Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    Variable c = torc::max(a, 0);
    EXPECT_TRUE(c.data() == Tensor({4.0f, 5.0f, 6.0f}, std::vector<int>{3}));
    Tensor grad({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    c.backward(grad);
    Tensor expected({0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f}, std::vector<int>{2, 3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(MaxMinAutograd, MaxAxis1Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    Variable c = torc::max(a, 1);
    EXPECT_TRUE(c.data() == Tensor({3.0f, 6.0f}, std::vector<int>{2}));
    Tensor grad({1.0f, 2.0f}, std::vector<int>{2});
    c.backward(grad);
    Tensor expected({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f}, std::vector<int>{2, 3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(MaxMinAutograd, MinAxis0Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    Variable c = torc::min(a, 0);
    EXPECT_TRUE(c.data() == Tensor({1.0f, 2.0f, 3.0f}, std::vector<int>{3}));
    Tensor grad({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    c.backward(grad);
    Tensor expected({1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f}, std::vector<int>{2, 3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(MaxMinAutograd, MinAxis1Backward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    Variable c = torc::min(a, 1);
    EXPECT_TRUE(c.data() == Tensor({1.0f, 4.0f}, std::vector<int>{2}));
    Tensor grad({1.0f, 2.0f}, std::vector<int>{2});
    c.backward(grad);
    Tensor expected({1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f}, std::vector<int>{2, 3});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(MaxMinAutograd, GradCheckMaxAxis0) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    Variable c = torc::max(a, 0);
    Tensor ones({1.0f, 1.0f, 1.0f}, std::vector<int>{3});
    c.backward(ones);
    Tensor num_grad = numerical_grad_max(a_data, 0);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(MaxMinAutograd, GradCheckMaxAxis1) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    Variable c = torc::max(a, 1);
    Tensor ones({1.0f, 1.0f}, std::vector<int>{2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_max(a_data, 1);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

// ============================================================
// Step 5: matmul backward (2D + batched, batch-broadcast)
// ============================================================

static Tensor numerical_grad_matmul_a(const Tensor& a, const Tensor& b) {
    Tensor g(a.shape());
    float h = 1e-3f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        Tensor c_ph = a_ph.matmul(b);
        Tensor c_mh = a_mh.matmul(b);
        float f_ph = 0, f_mh = 0;
        for (int j = 0; j < c_ph.numel(); ++j) {
            f_ph += c_ph.data()[j];
            f_mh += c_mh.data()[j];
        }
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_matmul_b(const Tensor& a, const Tensor& b) {
    Tensor g(b.shape());
    float h = 1e-3f;
    for (int i = 0; i < b.numel(); ++i) {
        Tensor b_ph = b; b_ph.data()[i] += h;
        Tensor b_mh = b; b_mh.data()[i] -= h;
        Tensor c_ph = a.matmul(b_ph);
        Tensor c_mh = a.matmul(b_mh);
        float f_ph = 0, f_mh = 0;
        for (int j = 0; j < c_ph.numel(); ++j) {
            f_ph += c_ph.data()[j];
            f_mh += c_mh.data()[j];
        }
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_matmul_a_batched(const Tensor& a, const Tensor& b, const Tensor& grad_output) {
    Tensor g(a.shape());
    float h = 1e-2f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        Tensor c_ph = a_ph.matmul(b);
        Tensor c_mh = a_mh.matmul(b);
        float f_ph = 0, f_mh = 0;
        for (int j = 0; j < c_ph.numel(); ++j) {
            f_ph += grad_output.data()[j] * c_ph.data()[j];
            f_mh += grad_output.data()[j] * c_mh.data()[j];
        }
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_matmul_b_batched(const Tensor& a, const Tensor& b, const Tensor& grad_output) {
    Tensor g(b.shape());
    float h = 1e-2f;
    for (int i = 0; i < b.numel(); ++i) {
        Tensor b_ph = b; b_ph.data()[i] += h;
        Tensor b_mh = b; b_mh.data()[i] -= h;
        Tensor c_ph = a.matmul(b_ph);
        Tensor c_mh = a.matmul(b_mh);
        float f_ph = 0, f_mh = 0;
        for (int j = 0; j < c_ph.numel(); ++j) {
            f_ph += grad_output.data()[j] * c_ph.data()[j];
            f_mh += grad_output.data()[j] * c_mh.data()[j];
        }
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

TEST(MatmulAutograd, Matmul2DBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Tensor b_data({5.0f, 6.0f, 7.0f, 8.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::matmul(a, b);
    Tensor expected_c({19.0f, 22.0f, 43.0f, 50.0f}, std::vector<int>{2, 2});
    EXPECT_TRUE(c.data() == expected_c);
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    c.backward(grad);
    Tensor expected_da({17.0f, 23.0f, 39.0f, 53.0f}, std::vector<int>{2, 2});
    Tensor expected_db({10.0f, 14.0f, 14.0f, 20.0f}, std::vector<int>{2, 2});
    EXPECT_TRUE(a.grad() == expected_da);
    EXPECT_TRUE(b.grad() == expected_db);
}

TEST(MatmulAutograd, Matmul2DGradCheck) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Tensor b_data({5.0f, 6.0f, 7.0f, 8.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::matmul(a, b);
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor num_grad_a = numerical_grad_matmul_a(a_data, b_data);
    Tensor num_grad_b = numerical_grad_matmul_b(a_data, b_data);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad_a.data()[i]);
    for (int i = 0; i < b_data.numel(); ++i)
        expect_near(b.grad().data()[i], num_grad_b.data()[i]);
}

TEST(MatmulAutograd, MatmulBatchedBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 1, 2});
    Tensor b_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{1, 2, 2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::matmul(a, b);
    Tensor expected_c({7.0f, 10.0f, 15.0f, 22.0f}, std::vector<int>{2, 1, 2});
    EXPECT_TRUE(c.data() == expected_c);
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 1, 2});
    c.backward(grad);
    Tensor expected_da({5.0f, 11.0f, 11.0f, 25.0f}, std::vector<int>{2, 1, 2});
    Tensor expected_db({10.0f, 14.0f, 14.0f, 20.0f}, std::vector<int>{1, 2, 2});
    EXPECT_TRUE(a.grad() == expected_da);
    EXPECT_TRUE(b.grad() == expected_db);
}

TEST(MatmulAutograd, MatmulBatchBroadcastBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{1, 2, 2});
    Tensor b_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, std::vector<int>{2, 2, 2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::matmul(a, b);
    Tensor expected_c({7.0f, 10.0f, 15.0f, 22.0f, 19.0f, 22.0f, 43.0f, 50.0f}, std::vector<int>{2, 2, 2});
    EXPECT_TRUE(c.data() == expected_c);
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, std::vector<int>{2, 2, 2});
    c.backward(grad);
    Tensor expected_da({66.0f, 94.0f, 94.0f, 138.0f}, std::vector<int>{1, 2, 2});
    Tensor expected_db({10.0f, 14.0f, 14.0f, 20.0f, 26.0f, 30.0f, 38.0f, 44.0f}, std::vector<int>{2, 2, 2});
    EXPECT_TRUE(a.grad() == expected_da);
    EXPECT_TRUE(b.grad() == expected_db);
}

TEST(MatmulAutograd, MatmulBatchedGradCheck) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 1, 2});
    Tensor b_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{1, 2, 2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::matmul(a, b);
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 1, 2});
    c.backward(grad);
    Tensor num_grad_a = numerical_grad_matmul_a_batched(a_data, b_data, grad);
    Tensor num_grad_b = numerical_grad_matmul_b_batched(a_data, b_data, grad);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad_a.data()[i]);
    for (int i = 0; i < b_data.numel(); ++i)
        expect_near(b.grad().data()[i], num_grad_b.data()[i]);
}

// ============================================================
// Step 6: View ops backward (transpose, reshape/view, slice)
// ============================================================

static Tensor numerical_grad_transpose(const Tensor& a, const std::vector<int>& axes) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = a_ph.transpose(axes).sum();
        float f_mh = a_mh.transpose(axes).sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_reshape(const Tensor& a, const std::vector<int>& new_shape) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        float f_ph = a_ph.reshape(new_shape).sum();
        float f_mh = a_mh.reshape(new_shape).sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

static Tensor numerical_grad_slice(const Tensor& a, const std::vector<std::pair<int, int>>& slices) {
    Tensor g(a.shape());
    float h = 1e-4f;
    for (int i = 0; i < a.numel(); ++i) {
        Tensor a_ph = a; a_ph.data()[i] += h;
        Tensor a_mh = a; a_mh.data()[i] -= h;
        std::vector<Tensor::Slice> ts;
        for (auto& p : slices) ts.push_back({p.first, p.second});
        float f_ph = a_ph.slice(ts).sum();
        float f_mh = a_mh.slice(ts).sum();
        g.data()[i] = (f_ph - f_mh) / (2.0f * h);
    }
    return g;
}

TEST(ViewAutograd, TransposeDefaultBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::transpose(a);
    EXPECT_TRUE(c.data() == Tensor({1.0f, 3.0f, 2.0f, 4.0f}, std::vector<int>{2, 2}));
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    c.backward(grad);
    Tensor expected({1.0f, 3.0f, 2.0f, 4.0f}, std::vector<int>{2, 2});
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ViewAutograd, TransposeCustomAxesBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3, 1});
    Variable a(a_data, true);
    std::vector<int> axes = {2, 0, 1};
    Variable c = torc::transpose(a, axes);
    EXPECT_TRUE(c.data() == a.data().transpose(axes));
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{1, 2, 3});
    c.backward(grad);
    std::vector<int> inv(3);
    for (int i = 0; i < 3; ++i) inv[axes[i]] = i;
    Tensor expected = grad.transpose(inv);
    EXPECT_TRUE(a.grad() == expected);
}

TEST(ViewAutograd, Transpose3DBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3, 1});
    Variable a(a_data, true);
    std::vector<int> axes = {1, 2, 0};
    Variable c = torc::transpose(a, axes);
    Tensor expected_c({1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}, std::vector<int>{3, 1, 2});
    EXPECT_TRUE(c.data() == expected_c);
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{3, 1, 2});
    c.backward(grad);
    std::vector<int> inv(3);
    for (int i = 0; i < 3; ++i) inv[axes[i]] = i;
    Tensor expected_grad = grad.transpose(inv);
    EXPECT_TRUE(a.grad() == expected_grad);
}

TEST(ViewAutograd, ReshapeBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    Variable c = torc::reshape(a, std::vector<int>{3, 2});
    Tensor expected_c({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{3, 2});
    EXPECT_TRUE(c.data() == expected_c);
    Tensor grad({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{3, 2});
    c.backward(grad);
    Tensor expected_grad({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    EXPECT_TRUE(a.grad() == expected_grad);
}

TEST(ViewAutograd, SliceBackward) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    std::vector<std::pair<int, int>> slices = {{0, 1}, {1, 3}};
    Variable c = torc::slice(a, slices);
    Tensor expected_c({2.0f, 3.0f}, std::vector<int>{1, 2});
    EXPECT_TRUE(c.data() == expected_c);
    Tensor grad({2.0f, 3.0f}, std::vector<int>{1, 2});
    c.backward(grad);
    Tensor expected_grad({0.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f}, std::vector<int>{2, 3});
    EXPECT_TRUE(a.grad() == expected_grad);
}

TEST(ViewAutograd, GradCheckTransposeDefault) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable a(a_data, true);
    Variable c = torc::transpose(a);
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_transpose(a_data, {1, 0});
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(ViewAutograd, GradCheckReshape) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{4});
    Variable a(a_data, true);
    Variable c = torc::reshape(a, std::vector<int>{2, 2});
    Tensor ones({1.0f, 1.0f, 1.0f, 1.0f}, std::vector<int>{2, 2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_reshape(a_data, std::vector<int>{2, 2});
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

TEST(ViewAutograd, GradCheckSlice) {
    Tensor a_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable a(a_data, true);
    std::vector<std::pair<int, int>> slices = {{0, 1}, {1, 3}};
    Variable c = torc::slice(a, slices);
    Tensor ones({1.0f, 1.0f}, std::vector<int>{1, 2});
    c.backward(ones);
    Tensor num_grad = numerical_grad_slice(a_data, slices);
    for (int i = 0; i < a_data.numel(); ++i)
        expect_near(a.grad().data()[i], num_grad.data()[i]);
}

// ============================================================
// Step 7: detach() and set_grad_enabled(bool)
// ============================================================

TEST(DetachNoGradAutograd, DetachBreaksGraph) {
    Tensor a_data({1.0f, 2.0f}, std::vector<int>{2});
    Tensor b_data({3.0f, 4.0f}, std::vector<int>{2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable c = torc::add(a, b);
    Variable d = c.detach();
    EXPECT_TRUE(d.requires_grad() == false);
    EXPECT_TRUE(d.tape().empty());
    Tensor grad({1.0f, 1.0f}, std::vector<int>{2});
    d.backward(grad);
    EXPECT_FALSE(a.has_grad());
    EXPECT_FALSE(b.has_grad());
}

TEST(DetachNoGradAutograd, DetachPreservesData) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    Variable d = a.detach();
    EXPECT_TRUE(d.data() == a_data);
}

TEST(DetachNoGradAutograd, SetGradEnabledFalseDisablesTape) {
    Tensor a_data({1.0f, 2.0f}, std::vector<int>{2});
    Tensor b_data({3.0f, 4.0f}, std::vector<int>{2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable::set_grad_enabled(false);
    Variable c = torc::add(a, b);
    EXPECT_TRUE(c.requires_grad() == false);
    EXPECT_TRUE(c.tape().empty());
    Variable::set_grad_enabled(true);
}

TEST(DetachNoGradAutograd, SetGradEnabledTrueRestoresTape) {
    Tensor a_data({1.0f, 2.0f}, std::vector<int>{2});
    Variable a(a_data, true);
    Variable::set_grad_enabled(true);
    Variable c = torc::neg(a);
    EXPECT_TRUE(c.requires_grad() == true);
    EXPECT_FALSE(c.tape().empty());
}

TEST(DetachNoGradAutograd, GradEnabledDefaultsToTrue) {
    EXPECT_TRUE(Variable::grad_enabled());
}

TEST(DetachNoGradAutograd, NoGradGuardDisablesAndRestores) {
    EXPECT_TRUE(Variable::grad_enabled());
    {
        Variable::NoGradGuard guard;
        EXPECT_FALSE(Variable::grad_enabled());
    }
    EXPECT_TRUE(Variable::grad_enabled());
}

TEST(DetachNoGradAutograd, NoGradGuardRestoresAfterException) {
    EXPECT_TRUE(Variable::grad_enabled());
    try {
        Variable::NoGradGuard guard;
        EXPECT_FALSE(Variable::grad_enabled());
        throw std::runtime_error("simulated");
    } catch (...) {
    }
    EXPECT_TRUE(Variable::grad_enabled());
}

TEST(DetachNoGradAutograd, NoGradScopePreservesTrackingOnInputs) {
    Tensor a_data({1.0f, 2.0f}, std::vector<int>{2});
    Tensor b_data({3.0f, 4.0f}, std::vector<int>{2});
    Variable a(a_data, true);
    Variable b(b_data, true);
    Variable::set_grad_enabled(false);
    Variable c = torc::add(a, b);
    Variable::set_grad_enabled(true);
    EXPECT_TRUE(a.requires_grad());
    EXPECT_TRUE(b.requires_grad());
    EXPECT_FALSE(c.requires_grad());
}

TEST(DetachNoGradAutograd, NoGradBlocksActivationOps) {
    Tensor a_data({1.0f, -1.0f}, std::vector<int>{2});
    Variable a(a_data, true);
    Variable::set_grad_enabled(false);
    Variable c = torc::relu(a);
    Variable d = torc::exp(c);
    Variable::set_grad_enabled(true);
    EXPECT_FALSE(c.requires_grad());
    EXPECT_FALSE(d.requires_grad());
    EXPECT_TRUE(c.tape().empty());
    EXPECT_TRUE(d.tape().empty());
}

TEST(DetachNoGradAutograd, NoGradBlocksTranscendentalOps) {
    Tensor a_data({1.0f, 2.0f}, std::vector<int>{2});
    Variable a(a_data, true);
    Variable::set_grad_enabled(false);
    Variable c = torc::log(a);
    Variable d = torc::sqrt(c);
    Variable::set_grad_enabled(true);
    EXPECT_FALSE(c.requires_grad());
    EXPECT_FALSE(d.requires_grad());
    EXPECT_TRUE(c.tape().empty());
    EXPECT_TRUE(d.tape().empty());
}

TEST(GradMapAccumulation, DiamondDependencyAccumulates) {
    Variable x(2.0f, true);
    Variable a = torc::add(x, x);
    Variable b = torc::add(a, a);
    Variable c = torc::mul_scalar(a, Variable(3.0f, false));
    Variable d = torc::add(b, c);
    Variable s = torc::sum(d);
    s.backward();
    ASSERT_TRUE(x.has_grad());
    expect_near(x.grad().data()[0], 10.0f);
}

// ============================================================
// Step 8: In-place ops forbidden for requires_grad Variables
// ============================================================

TEST(InPlaceGuardAutograd, FillOnTrackedVariableThrows) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, true);
    EXPECT_THROW(a.fill(0.0f), TorcError);
}

TEST(InPlaceGuardAutograd, FillOnUntrackedVariableSucceeds) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable a(a_data, false);
    EXPECT_NO_THROW(a.fill(0.0f));
    Tensor expected({0.0f, 0.0f, 0.0f}, std::vector<int>{3});
    EXPECT_TRUE(a.data() == expected);
}

TEST(InPlaceGuardAutograd, TensorFillIsAlwaysAllowed) {
    Tensor a_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    a_data.fill(5.0f);
    Tensor expected({5.0f, 5.0f, 5.0f}, std::vector<int>{3});
    EXPECT_TRUE(a_data == expected);
}
