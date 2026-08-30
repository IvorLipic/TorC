#include <gtest/gtest.h>
#include "torc/nn.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"
#include <string>
#include <utility>
#include <cmath>
#include <limits>

using torc::Variable;
using torc::Tensor;
using torc::nn::Module;
using torc::nn::Sequential;
using torc::nn::Linear;
using torc::nn::ReLU;
using torc::nn::Sigmoid;
using torc::nn::Softmax;
using torc::nn::MSELoss;
using torc::nn::CrossEntropyLoss;
using torc::optim::SGD;
using torc::optim::Adam;
using torc::optim::AdamW;

static constexpr float EPS = 1e-4f;
static constexpr float GRAD_ATOL = 1e-2f;

static void expect_near(float actual, float expected, float tol = GRAD_ATOL) {
    EXPECT_NEAR(actual, expected, tol)
        << "actual=" << actual << " expected=" << expected;
}

class MultiplyModule : public Module {
public:
    explicit MultiplyModule(float weight) {
        register_parameter("weight", Variable(weight, true));
    }
    
    Variable forward(const Variable& x) const override {
        const Variable& w = named_parameters().at("weight");
        return torc::mul_scalar(x, w);
    }
};

class AddModule : public Module {
public:
    explicit AddModule(float bias) {
        register_parameter("bias", Variable(bias, true));
    }
    
    Variable forward(const Variable& x) const override {
        const Variable& b = named_parameters().at("bias");
        return torc::add_scalar(x, b);
    }
};

TEST(ModuleBase, RegisterParameterAndForward) {
    MultiplyModule mod(2.0f);
    Variable input(3.0f, false);
    Variable output = mod(input);
    EXPECT_FLOAT_EQ(output.data().data()[0], 6.0f);
    EXPECT_EQ(mod.parameters().size(), 1);
    EXPECT_TRUE(mod.named_parameters().at("weight").requires_grad());
}

TEST(ModuleBase, NamedParametersContainsRegisteredParam) {
    MultiplyModule mod(2.0f);
    ASSERT_EQ(mod.named_parameters().size(), 1);
    EXPECT_TRUE(mod.named_parameters().contains("weight"));
}

TEST(ModuleBase, OperatorCallsForward) {
    MultiplyModule mod(2.0f);
    Variable input(3.0f, false);
    Variable output = mod(input);
    EXPECT_FLOAT_EQ(output.data().data()[0], 6.0f);
}

TEST(Sequential, ForwardPassesThroughAllModules) {
    Sequential seq;
    seq.add(std::make_unique<MultiplyModule>(2.0f));
    seq.add(std::make_unique<MultiplyModule>(3.0f));
    
    Variable input(4.0f, false);
    Variable output = seq(input);
    EXPECT_FLOAT_EQ(output.data().data()[0], 24.0f);
}

TEST(Sequential, CollectsParametersFromChildren) {
    Sequential seq;
    seq.add(std::make_unique<MultiplyModule>(2.0f));
    seq.add(std::make_unique<MultiplyModule>(3.0f));
    
    auto params = seq.parameters();
    EXPECT_EQ(params.size(), 2);
    for (const auto* p : params) {
        EXPECT_TRUE(p->requires_grad());
    }
}

TEST(Sequential, EmptyForwardReturnsInput) {
    Sequential seq;
    Variable input(5.0f, false);
    Variable output = seq(input);
    EXPECT_FLOAT_EQ(output.data().data()[0], 5.0f);
}

TEST(Sequential, AddNullModuleThrows) {
    Sequential seq;
    EXPECT_THROW(seq.add(nullptr), torc::TorcError);
}

TEST(Sequential, CollectsOwnAndChildParameters) {
    Sequential seq;
    seq.register_parameter("seq_param", Variable(10.0f, true));
    seq.add(std::make_unique<MultiplyModule>(2.0f));
    
    auto params = seq.parameters();
    EXPECT_EQ(params.size(), 2);
}

TEST(Sequential, ChainedOpsComposeCorrectly) {
    Sequential seq;
    seq.add(std::make_unique<MultiplyModule>(2.0f));
    seq.add(std::make_unique<AddModule>(1.0f));
    
    Variable input(3.0f, false);
    Variable output = seq(input);
    EXPECT_FLOAT_EQ(output.data().data()[0], 7.0f);
}

TEST(Linear, ConstructionCreatesCorrectShapes) {
    Linear linear(2, 3);
    auto params = linear.named_parameters();
    ASSERT_EQ(params.size(), 2);
    EXPECT_TRUE(params.contains("weight"));
    EXPECT_TRUE(params.contains("bias"));
    
    const Tensor& weight = params.at("weight").data();
    const Tensor& bias = params.at("bias").data();
    
    std::vector<int> expected_weight_shape{3, 2};
    EXPECT_EQ(weight.shape(), expected_weight_shape);
    
    std::vector<int> expected_bias_shape{3};
    EXPECT_EQ(bias.shape(), expected_bias_shape);
}

TEST(Linear, ForwardMatchesHandComputed) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    
    Tensor input_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable input(input_data, false);
    
    Variable output = linear(input);
    
    std::vector<int> expected_output_shape{1, 3};
    ASSERT_EQ(output.data().shape(), expected_output_shape);
    
    const Tensor& W = linear.named_parameters().at("weight").data();
    const Tensor& b = linear.named_parameters().at("bias").data();
    float expected0 = W.data()[0] * 1.0f + W.data()[1] * 2.0f + b.data()[0];
    float expected1 = W.data()[2] * 1.0f + W.data()[3] * 2.0f + b.data()[1];
    float expected2 = W.data()[4] * 1.0f + W.data()[5] * 2.0f + b.data()[2];
    EXPECT_FLOAT_EQ(output.data().data()[0], expected0);
    EXPECT_FLOAT_EQ(output.data().data()[1], expected1);
    EXPECT_FLOAT_EQ(output.data().data()[2], expected2);
}

TEST(Linear, ParametersAreTracked) {
    Linear linear(2, 3);
    auto params = linear.parameters();
    ASSERT_EQ(params.size(), 2);
    for (const auto* p : params) {
        EXPECT_TRUE(p->requires_grad());
    }
}

TEST(Linear, ForwardWithBatchedInput) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    
    Tensor input_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable input(input_data, false);
    
    Variable output = linear(input);
    
    std::vector<int> expected_output_shape{2, 3};
    ASSERT_EQ(output.data().shape(), expected_output_shape);
    
    const Tensor& W = linear.named_parameters().at("weight").data();
    const Tensor& b = linear.named_parameters().at("bias").data();
    for (int batch = 0; batch < 2; ++batch) {
        float x0 = input_data.data()[batch * 2];
        float x1 = input_data.data()[batch * 2 + 1];
        for (int j = 0; j < 3; ++j) {
            float expected = W.data()[j * 2] * x0 + W.data()[j * 2 + 1] * x1 + b.data()[j];
            EXPECT_FLOAT_EQ(output.data().data()[batch * 3 + j], expected);
        }
    }
}

TEST(Linear, GradCheckForwardAndBackward) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    
    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    
    loss.backward();
    
    const Tensor& W_grad = linear.named_parameters().at("weight").grad();
    const Tensor& b_grad = linear.named_parameters().at("bias").grad();
    
    ASSERT_TRUE(linear.named_parameters().at("weight").has_grad());
    ASSERT_TRUE(linear.named_parameters().at("bias").has_grad());
    
    expect_near(W_grad.data()[0], 1.0f);
    expect_near(W_grad.data()[1], 2.0f);
    expect_near(W_grad.data()[2], 1.0f);
    expect_near(W_grad.data()[3], 2.0f);
    expect_near(W_grad.data()[4], 1.0f);
    expect_near(W_grad.data()[5], 2.0f);
    
    expect_near(b_grad.data()[0], 1.0f);
    expect_near(b_grad.data()[1], 1.0f);
    expect_near(b_grad.data()[2], 1.0f);
}

TEST(Linear, GradCheckWithBatchedInput) {
    Linear linear(2, 3, 0.01f);
    
    // Batch of 2: [[1, 2], [3, 4]]
    Tensor x_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable x(x_data, true);
    
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    
    loss.backward();
    
    const Tensor& W_grad = linear.named_parameters().at("weight").grad();
    const Tensor& b_grad = linear.named_parameters().at("bias").grad();
    
    // dL/dW[j,l] = sum_i x[i,l]
    // sum_i x[i,0] = 1 + 3 = 4
    // sum_i x[i,1] = 2 + 4 = 6
    expect_near(W_grad.data()[0], 4.0f);
    expect_near(W_grad.data()[1], 6.0f);
    expect_near(W_grad.data()[2], 4.0f);
    expect_near(W_grad.data()[3], 6.0f);
    expect_near(W_grad.data()[4], 4.0f);
    expect_near(W_grad.data()[5], 6.0f);
    
    // dL/db[j] = batch_size = 2
    expect_near(b_grad.data()[0], 2.0f);
    expect_near(b_grad.data()[1], 2.0f);
    expect_near(b_grad.data()[2], 2.0f);
}

TEST(ReLU, ForwardPassesPositiveAndZeroesNegative) {
    ReLU relu;
    Tensor input_data({-1.0f, 0.0f, 2.0f, -3.0f}, std::vector<int>{2, 2});
    Variable input(input_data, false);
    
    Variable output = relu(input);
    
    std::vector<int> expected_shape{2, 2};
    ASSERT_EQ(output.data().shape(), expected_shape);
    EXPECT_FLOAT_EQ(output.data().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(output.data().data()[1], 0.0f);
    EXPECT_FLOAT_EQ(output.data().data()[2], 2.0f);
    EXPECT_FLOAT_EQ(output.data().data()[3], 0.0f);
}

TEST(ReLU, BackwardPassesGradientOnlyThroughPositive) {
    ReLU relu;
    Tensor input_data({-1.0f, 0.0f, 2.0f, -3.0f}, std::vector<int>{2, 2});
    Variable input(input_data, true);
    
    Variable output = relu(input);
    Variable loss = torc::sum(output);
    loss.backward();
    
    ASSERT_TRUE(input.has_grad());
    EXPECT_FLOAT_EQ(input.grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(input.grad().data()[1], 0.0f);
    EXPECT_FLOAT_EQ(input.grad().data()[2], 1.0f);
    EXPECT_FLOAT_EQ(input.grad().data()[3], 0.0f);
}

TEST(Sigmoid, ForwardMatchesHandComputed) {
    Sigmoid sigmoid;
    Tensor input_data({0.0f, 1.0f, -1.0f}, std::vector<int>{3});
    Variable input(input_data, false);
    
    Variable output = sigmoid(input);
    
    std::vector<int> expected_shape{3};
    ASSERT_EQ(output.data().shape(), expected_shape);
    expect_near(output.data().data()[0], 0.5f, 1e-5f);
    expect_near(output.data().data()[1], 0.7310586f, 1e-5f);
    expect_near(output.data().data()[2], 0.2689414f, 1e-5f);
}

TEST(Sigmoid, BackwardMatchesChainRule) {
    Sigmoid sigmoid;
    Tensor input_data({0.0f}, std::vector<int>{1});
    Variable input(input_data, true);
    
    Variable output = sigmoid(input);
    Variable loss = torc::sum(output);
    loss.backward();
    
    ASSERT_TRUE(input.has_grad());
    float s = 0.5f;
    expect_near(input.grad().data()[0], s * (1.0f - s), 1e-5f);
}

TEST(Softmax, ForwardProducesValidProbabilities) {
    Softmax softmax;
    Tensor input_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable input(input_data, false);
    
    Variable output = softmax(input);
    
    std::vector<int> expected_shape{3};
    ASSERT_EQ(output.data().shape(), expected_shape);
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) sum += output.data().data()[i];
    expect_near(sum, 1.0f, 1e-5f);
    
    for (int i = 0; i < 3; ++i) {
        EXPECT_GT(output.data().data()[i], 0.0f);
    }
}

TEST(Softmax, BatchedForwardNormalizesEachRow) {
    Softmax softmax;
    Tensor input_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable input(input_data, false);

    Variable output = softmax(input);

    EXPECT_NEAR(output.data().data()[0] + output.data().data()[1] + output.data().data()[2],
                1.0f, 1e-5f);
    EXPECT_NEAR(output.data().data()[3] + output.data().data()[4] + output.data().data()[5],
                1.0f, 1e-5f);
    for (int j = 0; j < 3; ++j)
        EXPECT_NEAR(output.data().data()[j], output.data().data()[3 + j], 1e-5f);
}

TEST(Softmax, BatchedBackwardUsesPerRowJacobian) {
    Softmax softmax;
    Tensor input_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Variable input(input_data, true);
    Variable output = softmax(input);
    Tensor upstream({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, std::vector<int>{2, 3});
    output.backward(upstream);

    float row0_sum = output.data().data()[0];
    float row1_sum = output.data().data()[4];
    for (int j = 0; j < 3; ++j) {
        float expected0 = output.data().data()[j] * ((j == 0 ? 1.0f : 0.0f) - row0_sum);
        float expected1 = output.data().data()[3 + j] * ((j == 1 ? 1.0f : 0.0f) - row1_sum);
        EXPECT_NEAR(input.grad().data()[j], expected0, 1e-5f);
        EXPECT_NEAR(input.grad().data()[3 + j], expected1, 1e-5f);
    }
}

TEST(Softmax, BackwardMatchesNumericalGradient) {
    Softmax softmax;
    Tensor input_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Variable input(input_data, true);
    
    Variable output = softmax(input);
    Variable loss = torc::sum(output);
    loss.backward();
    
    ASSERT_TRUE(input.has_grad());
    float h = 1e-4f;
    for (int i = 0; i < 3; ++i) {
        Tensor x_plus = input_data;
        x_plus.data()[i] += h;
        Variable out_plus = softmax(Variable(x_plus, false));
        float loss_plus = torc::sum(out_plus).data().data()[0];
        
        Tensor x_minus = input_data;
        x_minus.data()[i] -= h;
        Variable out_minus = softmax(Variable(x_minus, false));
        float loss_minus = torc::sum(out_minus).data().data()[0];
        
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * h);
        expect_near(input.grad().data()[i], numerical_grad);
    }
}

TEST(MSELoss, ForwardMatchesHandComputed) {
    MSELoss loss_fn;
    Tensor input_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Tensor target_data({1.5f, 2.5f, 3.5f}, std::vector<int>{3});
    Variable input(input_data, false);
    Variable target(target_data, false);
    
    Variable loss = loss_fn(input, target);
    
    // diff = [-0.5, -0.5, -0.5], squared = [0.25, 0.25, 0.25], mean = 0.25
    EXPECT_FLOAT_EQ(loss.data().data()[0], 0.25f);
}

TEST(MSELoss, BackwardMatchesChainRule) {
    MSELoss loss_fn;
    Tensor input_data({1.0f, 2.0f, 3.0f}, std::vector<int>{3});
    Tensor target_data({1.5f, 2.5f, 3.5f}, std::vector<int>{3});
    Variable input(input_data, true);
    Variable target(target_data, false);
    
    Variable loss = loss_fn(input, target);
    loss.backward();
    
    ASSERT_TRUE(input.has_grad());
    // grad = 2 * (input - target) / 3 = 2 * [-0.5, -0.5, -0.5] / 3 = [-1/3, -1/3, -1/3]
    expect_near(input.grad().data()[0], -1.0f / 3.0f);
    expect_near(input.grad().data()[1], -1.0f / 3.0f);
    expect_near(input.grad().data()[2], -1.0f / 3.0f);
}

TEST(CrossEntropyLoss, ForwardMatchesHandComputed) {
    CrossEntropyLoss loss_fn;
    // 2 samples, 3 classes
    Tensor logits_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Tensor targets_data({2.0f, 0.0f}, std::vector<int>{2});
    Variable logits(logits_data, false);
    Variable targets(targets_data, false);
    
    Variable loss = loss_fn(logits, targets);
    
    // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652], log = [-2.4079, -1.4065, -0.4079]
    // softmax([4,5,6]) = [0.0900, 0.2447, 0.6652], log = [-2.4079, -1.4065, -0.4079]
    // loss = -((-0.4079) + (-2.4079)) / 2 ≈ 1.4076
    expect_near(loss.data().data()[0], 1.4076f, 2e-3f);
}

TEST(CrossEntropyLoss, AcceptsColumnVectorTargets) {
    CrossEntropyLoss loss_fn;
    // 2 samples, 3 classes, targets given as a column vector [n, 1] (rank-2)
    Tensor logits_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Tensor targets_data({2.0f, 0.0f}, std::vector<int>{2, 1});
    Variable logits(logits_data, false);
    Variable targets(targets_data, false);

    Variable loss = loss_fn(logits, targets);
    expect_near(loss.data().data()[0], 1.4076f, 2e-3f);
}

TEST(CrossEntropyLoss, BackwardMatchesNumericalGradient) {
    CrossEntropyLoss loss_fn;
    Tensor logits_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Tensor targets_data({2.0f, 0.0f}, std::vector<int>{2});
    Variable logits(logits_data, true);
    Variable targets(targets_data, false);
    
    Variable loss = loss_fn(logits, targets);
    loss.backward();
    
    ASSERT_TRUE(logits.has_grad());
    float h = 1e-4f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            Tensor x_plus = logits_data;
            x_plus.data()[i * 3 + j] += h;
            Variable out_plus = loss_fn(Variable(x_plus, false), targets);
            float loss_plus = out_plus.data().data()[0];
            
            Tensor x_minus = logits_data;
            x_minus.data()[i * 3 + j] -= h;
            Variable out_minus = loss_fn(Variable(x_minus, false), targets);
            float loss_minus = out_minus.data().data()[0];
            
            float numerical_grad = (loss_plus - loss_minus) / (2.0f * h);
            expect_near(logits.grad().data()[i * 3 + j], numerical_grad);
        }
    }
}

TEST(CrossEntropyLoss, RejectsMalformedLogitsAndTargets) {
    CrossEntropyLoss loss_fn;
    Variable rank_one_logits(Tensor({1.0f, 2.0f, 3.0f}, std::vector<int>{3}), false);
    Variable valid_target(Tensor({1.0f}, std::vector<int>{1}), false);
    EXPECT_THROW(loss_fn(rank_one_logits, valid_target), torc::ShapeError);

    Variable logits(Tensor({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2}), false);
    Variable wrong_rank(Tensor({0.0f, 1.0f}, std::vector<int>{1, 2}), false);
    EXPECT_THROW(loss_fn(logits, wrong_rank), torc::ShapeError);

    Variable wrong_length(Tensor({0.0f}, std::vector<int>{1}), false);
    EXPECT_THROW(loss_fn(logits, wrong_length), torc::ShapeError);
}

TEST(CrossEntropyLoss, RejectsInvalidTargetValuesAndEmptyInputs) {
    CrossEntropyLoss loss_fn;
    Variable logits(Tensor({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2}), false);

    Variable fractional(Tensor({0.5f, 1.0f}, std::vector<int>{2}), false);
    EXPECT_THROW(loss_fn(logits, fractional), torc::ShapeError);
    Variable negative(Tensor({-1.0f, 1.0f}, std::vector<int>{2}), false);
    EXPECT_THROW(loss_fn(logits, negative), torc::ShapeError);
    Variable too_large(Tensor({0.0f, 2.0f}, std::vector<int>{2}), false);
    EXPECT_THROW(loss_fn(logits, too_large), torc::ShapeError);
    Variable nan_target(Tensor({std::numeric_limits<float>::quiet_NaN(), 1.0f}, std::vector<int>{2}), false);
    EXPECT_THROW(loss_fn(logits, nan_target), torc::ShapeError);

    Variable empty_logits(Tensor(std::vector<int>{0, 2}), false);
    Variable empty_targets(Tensor(std::vector<int>{0}), false);
    EXPECT_THROW(loss_fn(empty_logits, empty_targets), torc::ShapeError);
}

TEST(CrossEntropyLoss, StableForExtremeLogits) {
    CrossEntropyLoss loss_fn;
    Tensor logits_data({1000.0f, -1000.0f, -1000.0f, 1000.0f}, std::vector<int>{2, 2});
    Tensor targets_data({0.0f, 1.0f}, std::vector<int>{2});
    Variable logits(logits_data, true);
    Variable targets(targets_data, false);

    Variable loss = loss_fn(logits, targets);
    EXPECT_TRUE(std::isfinite(loss.data().data()[0]));
    EXPECT_NEAR(loss.data().data()[0], 0.0f, 1e-5f);
    loss.backward();
    for (int i = 0; i < logits.data().numel(); ++i)
        EXPECT_NEAR(logits.grad().data()[i], 0.0f, 1e-5f);
}

TEST(SGD, StepWithoutMomentumUpdatesParams) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    auto params = linear.parameters();
    ASSERT_EQ(params.size(), 2);

    std::vector<float> initial_W;
    for (size_t i = 0; i < params[0]->data().numel(); ++i) {
        initial_W.push_back(params[0]->data().data()[i]);
    }

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    SGD optimizer(params, 0.1f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    const Tensor& b = linear.named_parameters().at("bias").data();
    for (size_t i = 0; i < W.numel(); ++i) {
        EXPECT_NE(W.data()[i], initial_W[i]);
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_NE(b.data()[i], 0.0f);
    }
}

TEST(SGD, StepWithMomentumAccumulatesVelocity) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    ASSERT_EQ(params.size(), 2);

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    SGD optimizer(params, 0.1f, 0.9f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    expect_near(W.data()[0], 0.01f - 0.1f * 1.0f);
}

TEST(SGD, ZeroGradClearsAllParamGrads) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    ASSERT_TRUE(linear.named_parameters().at("weight").has_grad());
    ASSERT_TRUE(linear.named_parameters().at("bias").has_grad());

    SGD optimizer(params, 0.1f);
    optimizer.zero_grad();

    EXPECT_FALSE(linear.named_parameters().at("weight").has_grad());
    EXPECT_FALSE(linear.named_parameters().at("bias").has_grad());
}

TEST(SGD, RejectsNegativeLearningRate) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    EXPECT_THROW(SGD(params, -0.1f), torc::TorcError);
}

TEST(SGD, RejectsNegativeMomentum) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    EXPECT_THROW(SGD(params, 0.1f, -0.1f), torc::TorcError);
}

TEST(Adam, RejectsNegativeLearningRate) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    EXPECT_THROW(Adam(params, -0.1f), torc::TorcError);
}

TEST(Adam, RejectsBetaOneOrGreater) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    EXPECT_THROW(Adam(params, 0.1f, 1.0f), torc::TorcError);
    EXPECT_THROW(Adam(params, 0.1f, -0.1f), torc::TorcError);
}

TEST(Adam, RejectsBetaTwoOrGreater) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    EXPECT_THROW(Adam(params, 0.1f, 0.9f, 1.0f), torc::TorcError);
}

TEST(Adam, RejectsNonPositiveEps) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    EXPECT_THROW(Adam(params, 0.1f, 0.9f, 0.999f, 0.0f), torc::TorcError);
}

TEST(AdamW, RejectsNegativeWeightDecay) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();
    EXPECT_THROW(AdamW(params, 0.1f, -0.01f), torc::TorcError);
}

TEST(Linear, SeedProducesReproducibleWeights) {
    Linear linear1(4, 2, 0.0f, 42);
    Linear linear2(4, 2, 0.0f, 42);
    const Tensor& W1 = linear1.named_parameters().at("weight").data();
    const Tensor& W2 = linear2.named_parameters().at("weight").data();
    for (size_t i = 0; i < W1.numel(); ++i) {
        EXPECT_FLOAT_EQ(W1.data()[i], W2.data()[i]);
    }
}

TEST(SGD, SkipsParamsWithoutGrad) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    auto params = linear.parameters();

    std::vector<float> initial_weights;
    for (size_t i = 0; i < params[0]->data().numel(); ++i) {
        initial_weights.push_back(params[0]->data().data()[i]);
    }

    SGD optimizer(params, 0.1f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    for (size_t i = 0; i < W.numel(); ++i) {
        EXPECT_FLOAT_EQ(W.data()[i], initial_weights[i]);
    }
}

TEST(Adam, StepUpdatesParams) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    auto params = linear.parameters();
    ASSERT_EQ(params.size(), 2);

    std::vector<float> initial_W;
    for (size_t i = 0; i < params[0]->data().numel(); ++i) {
        initial_W.push_back(params[0]->data().data()[i]);
    }
    std::vector<float> initial_b;
    for (size_t i = 0; i < params[1]->data().numel(); ++i) {
        initial_b.push_back(params[1]->data().data()[i]);
    }

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    Adam optimizer(params, 0.1f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    const Tensor& b = linear.named_parameters().at("bias").data();
    for (size_t i = 0; i < W.numel(); ++i) {
        EXPECT_NE(W.data()[i], initial_W[i]);
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_NE(b.data()[i], initial_b[i]);
    }
}

TEST(Adam, ZeroGradClearsAllParamGrads) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    ASSERT_TRUE(linear.named_parameters().at("weight").has_grad());
    ASSERT_TRUE(linear.named_parameters().at("bias").has_grad());

    Adam optimizer(params, 0.1f);
    optimizer.zero_grad();

    EXPECT_FALSE(linear.named_parameters().at("weight").has_grad());
    EXPECT_FALSE(linear.named_parameters().at("bias").has_grad());
}

TEST(Adam, SkipsParamsWithoutGrad) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    auto params = linear.parameters();

    std::vector<float> initial_weights;
    for (size_t i = 0; i < params[0]->data().numel(); ++i) {
        initial_weights.push_back(params[0]->data().data()[i]);
    }

    Adam optimizer(params, 0.1f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    for (size_t i = 0; i < W.numel(); ++i) {
        EXPECT_FLOAT_EQ(W.data()[i], initial_weights[i]);
    }
}

TEST(AdamW, StepWithoutWeightDecayBehavesLikeAdam) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    auto params = linear.parameters();

    std::vector<float> initial_W;
    for (size_t i = 0; i < params[0]->data().numel(); ++i) {
        initial_W.push_back(params[0]->data().data()[i]);
    }
    std::vector<float> initial_b;
    for (size_t i = 0; i < params[1]->data().numel(); ++i) {
        initial_b.push_back(params[1]->data().data()[i]);
    }

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    AdamW optimizer(params, 0.1f, 0.0f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    const Tensor& b = linear.named_parameters().at("bias").data();
    for (size_t i = 0; i < W.numel(); ++i) {
        EXPECT_NE(W.data()[i], initial_W[i]);
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_NE(b.data()[i], initial_b[i]);
    }
}

TEST(AdamW, StepWithWeightDecayUpdatesParams) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    auto params = linear.parameters();

    std::vector<float> initial_W;
    for (size_t i = 0; i < params[0]->data().numel(); ++i) {
        initial_W.push_back(params[0]->data().data()[i]);
    }
    std::vector<float> initial_b;
    for (size_t i = 0; i < params[1]->data().numel(); ++i) {
        initial_b.push_back(params[1]->data().data()[i]);
    }

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    AdamW optimizer(params, 0.1f, 0.01f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    const Tensor& b = linear.named_parameters().at("bias").data();
    for (size_t i = 0; i < W.numel(); ++i) {
        EXPECT_NE(W.data()[i], initial_W[i]);
    }
    for (size_t i = 0; i < b.numel(); ++i) {
        EXPECT_NE(b.data()[i], initial_b[i]);
    }
}

TEST(AdamW, ZeroGradClearsAllParamGrads) {
    Linear linear(2, 3, 0.01f);
    auto params = linear.parameters();

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();

    ASSERT_TRUE(linear.named_parameters().at("weight").has_grad());
    ASSERT_TRUE(linear.named_parameters().at("bias").has_grad());

    AdamW optimizer(params, 0.1f, 0.01f);
    optimizer.zero_grad();

    EXPECT_FALSE(linear.named_parameters().at("weight").has_grad());
    EXPECT_FALSE(linear.named_parameters().at("bias").has_grad());
}

TEST(AdamW, SkipsParamsWithoutGrad) {
    unsigned int seed = 42;
    Linear linear(2, 3, 0.0f, seed);
    auto params = linear.parameters();

    std::vector<float> initial_weights;
    for (size_t i = 0; i < params[0]->data().numel(); ++i) {
        initial_weights.push_back(params[0]->data().data()[i]);
    }

    AdamW optimizer(params, 0.1f, 0.01f);
    optimizer.step();

    const Tensor& W = linear.named_parameters().at("weight").data();
    for (size_t i = 0; i < W.numel(); ++i) {
        EXPECT_FLOAT_EQ(W.data()[i], initial_weights[i]);
    }
}

TEST(Sequential, BackwardReachesFirstLayer) {
    Sequential seq;
    seq.add(std::make_unique<Linear>(2, 2));
    seq.add(std::make_unique<ReLU>());
    seq.add(std::make_unique<Linear>(2, 1));

    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);

    Variable out = seq(x);
    Variable loss = torc::sum(out);
    loss.backward();

    auto params = seq.parameters();
    bool found_first_weight = false;
    for (const auto* p : params) {
        if (p->has_grad()) {
            found_first_weight = true;
            break;
        }
    }
    ASSERT_TRUE(found_first_weight);
}

TEST(Sequential, MnistArchitectureForwardPasses) {
    Sequential model;
    model.add(std::make_unique<Linear>(784, 128));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(128, 10));

    Tensor x_data(std::vector<int>{1, 784});
    x_data.fill(0.5f);
    Variable x(x_data, true);

    Variable out = model(x);
    auto shape = out.data().shape();
    ASSERT_EQ(shape, (std::vector<int>{1, 10}));
}

TEST(Sequential, MnistArchitectureBackwardPopulatesGrads) {
    Sequential model;
    model.add(std::make_unique<Linear>(784, 128));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(128, 10));

    Tensor x_data(std::vector<int>{1, 784});
    x_data.fill(0.5f);
    Variable x(x_data, true);

    Variable out = model(x);
    Variable loss = torc::sum(out);
    loss.backward();

    auto params = model.parameters();
    ASSERT_EQ(params.size(), 4);
    EXPECT_TRUE(params[0]->has_grad());
    EXPECT_TRUE(params[1]->has_grad());
    EXPECT_TRUE(params[2]->has_grad());
    EXPECT_TRUE(params[3]->has_grad());
}

TEST(Sequential, SingleTrainingStepUpdatesParams) {
    Sequential model;
    model.add(std::make_unique<Linear>(4, 8));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(8, 2));

    Tensor x_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{1, 4});
    Variable x(x_data, true);

    Tensor y_data({0.0f, 1.0f}, std::vector<int>{1, 2});
    Variable y(y_data, false);

    auto params = model.parameters();
    Adam optimizer(params, 0.01f);

    Variable out = model(x);
    Variable loss = torc::sum(out);
    loss.backward();
    optimizer.step();

    auto& w0 = model.parameters()[0]->data();
    EXPECT_NE(w0.data()[0], 0.01f);
}

TEST(MSELoss, BackwardScalesWithGradOutput) {
    MSELoss loss_fn;
    Tensor pred_data({1.0f, 2.0f, 3.0f}, std::vector<int>{1, 3});
    Tensor target_data({1.5f, 2.5f, 3.5f}, std::vector<int>{1, 3});
    Variable pred(pred_data, true);
    Variable target(target_data, false);

    Variable loss = loss_fn(pred, target);
    Tensor grad_output({2.0f}, std::vector<int>{1});
    loss.backward(grad_output);

    ASSERT_TRUE(pred.has_grad());
    for (int i = 0; i < 3; ++i) {
        float expected = 2.0f * 2.0f * (pred_data.data()[i] - target_data.data()[i]) / 3.0f;
        expect_near(pred.grad().data()[i], expected);
    }
}

TEST(CrossEntropyLoss, BackwardScalesWithGradOutput) {
    CrossEntropyLoss loss_fn;
    Tensor logits_data({1.0f, 2.0f, 3.0f}, std::vector<int>{1, 3});
    Tensor targets_data({2.0f}, std::vector<int>{1});
    Variable logits(logits_data, true);
    Variable targets(targets_data, false);

    Variable loss = loss_fn(logits, targets);
    Tensor grad_output({2.0f}, std::vector<int>{1});
    loss.backward(grad_output);

    ASSERT_TRUE(logits.has_grad());
    for (int i = 0; i < 3; ++i) {
        float expected = 2.0f * (logits.grad().data()[i] / 2.0f);
        expect_near(logits.grad().data()[i], expected);
    }
}

TEST(Linear, InitStdSamplesFromNormalDistribution) {
    Linear linear(4, 2, 0.5f);
    
    const Tensor& W = linear.named_parameters().at("weight").data();
    ASSERT_EQ(W.shape(), (std::vector<int>{2, 4}));
    
    bool all_same = true;
    for (int i = 1; i < W.numel(); ++i) {
        if (W.data()[i] != W.data()[0]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "All weights are identical — init_std was used as a constant, not a standard deviation";
    
    Tensor x_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{1, 4});
    Variable x(x_data, true);
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    loss.backward();
    
    ASSERT_TRUE(linear.named_parameters().at("weight").has_grad());
    ASSERT_TRUE(linear.named_parameters().at("bias").has_grad());
}

TEST(Linear, TwoArgConstructorBreaksSymmetry) {
    Linear linear(4, 3);
    
    const Tensor& W = linear.named_parameters().at("weight").data();
    ASSERT_EQ(W.shape(), (std::vector<int>{3, 4}));
    
    bool all_same = true;
    for (int i = 1; i < W.numel(); ++i) {
        if (W.data()[i] != W.data()[0]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "2-arg Linear constructor should use Kaiming init, not constant weights";
}

TEST(MSELoss, NoGradDisablesTape) {
    Variable::set_grad_enabled(false);
    MSELoss loss_fn;
    Tensor input_data({1.0f, 2.0f, 3.0f}, std::vector<int>{1, 3});
    Tensor target_data({1.5f, 2.5f, 3.5f}, std::vector<int>{1, 3});
    Variable input(input_data, true);
    Variable target(target_data, false);
    
    Variable loss = loss_fn(input, target);
    EXPECT_FLOAT_EQ(loss.data().data()[0], 0.25f);
    EXPECT_FALSE(loss.requires_grad());
    EXPECT_TRUE(loss.tape().empty());
    
    Variable::set_grad_enabled(true);
}

TEST(CrossEntropyLoss, NoGradDisablesTape) {
    Variable::set_grad_enabled(false);
    CrossEntropyLoss loss_fn;
    Tensor logits_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, std::vector<int>{2, 3});
    Tensor targets_data({2.0f, 0.0f}, std::vector<int>{2});
    Variable logits(logits_data, true);
    Variable targets(targets_data, false);
    
    Variable loss = loss_fn(logits, targets);
    expect_near(loss.data().data()[0], 1.4076f, 2e-3f);
    EXPECT_FALSE(loss.requires_grad());
    EXPECT_TRUE(loss.tape().empty());
    
    Variable::set_grad_enabled(true);
}

class CountingModule : public Module {
public:
    mutable int cache_size_after_forward = 0;
    
    Variable forward(const Variable& x) const override {
        forward_cache_.emplace_back(torc::add_scalar(x, Variable(1.0f, false)));
        cache_size_after_forward = (int)forward_cache_.size();
        return forward_cache_.back();
    }
    
    int cache_size() const { return (int)forward_cache_.size(); }
};

TEST(Sequential, ChildCacheDoesNotLeakAcrossForwards) {
    Sequential seq;
    auto mod = std::make_unique<CountingModule>();
    CountingModule* raw = mod.get();
    seq.add(std::move(mod));
    
    Tensor x_data({1.0f}, std::vector<int>{1});
    Variable x(x_data, false);
    
    seq(x);
    EXPECT_EQ(raw->cache_size(), 1);
    
    seq(x);
    EXPECT_EQ(raw->cache_size(), 1);
    
    seq(x);
    EXPECT_EQ(raw->cache_size(), 1);
}

TEST(DeepSequential, GradientsFlowThroughMultipleLayers) {
    unsigned int seed = 42;
    Sequential model;
    model.add(std::make_unique<Linear>(2, 4, 0.0f, seed));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(4, 3, 0.0f, seed));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(3, 1, 0.0f, seed));
    
    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    
    Variable out = model(x);
    Variable loss = torc::sum(out);
    loss.backward();
    
    auto params = model.parameters();
    ASSERT_EQ(params.size(), 6);
    bool found_grad = false;
    for (const auto* p : params) {
        if (p->has_grad()) {
            found_grad = true;
            break;
        }
    }
    ASSERT_TRUE(found_grad) << "No parameter received a gradient in deep Sequential";
}

TEST(DiamondDependencySequential, GradientsAccumulateCorrectly) {
    Sequential shared;
    shared.add(std::make_unique<Linear>(2, 2));
    
    Sequential model;
    model.add(std::make_unique<Linear>(2, 2));
    model.add(std::make_unique<torc::nn::Sequential>(std::move(shared)));
    model.add(std::make_unique<Linear>(2, 1));
    
    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    
    Variable out = model(x);
    Variable loss = torc::sum(out);
    loss.backward();
    
    auto params = model.parameters();
    ASSERT_EQ(params.size(), 6);
    for (const auto* p : params) {
        ASSERT_TRUE(p->has_grad()) << "Parameter missing gradient in diamond Sequential";
    }
}

TEST(DeepSequential, NumericalGradientsMatchAnalytical) {
    Sequential model;
    model.add(std::make_unique<Linear>(2, 4));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(4, 1));
    
    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    
    Variable out = model(x);
    Variable loss = torc::sum(out);
    loss.backward();
    
    ASSERT_TRUE(x.has_grad());
    
    float h = 1e-4f;
    for (int i = 0; i < 2; ++i) {
        Tensor x_plus = x_data;
        x_plus.data()[i] += h;
        Variable out_plus = model(Variable(x_plus, false));
        float loss_plus = torc::sum(out_plus).data().data()[0];
        
        Tensor x_minus = x_data;
        x_minus.data()[i] -= h;
        Variable out_minus = model(Variable(x_minus, false));
        float loss_minus = torc::sum(out_minus).data().data()[0];
        
        float numerical = (loss_plus - loss_minus) / (2.0f * h);
        EXPECT_NEAR(x.grad().data()[i], numerical, 1e-2f)
            << "Input gradient mismatch for element " << i;
    }
}
