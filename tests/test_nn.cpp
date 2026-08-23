#include <gtest/gtest.h>
#include "torc/nn.hpp"
#include "torc/nn/linear.hpp"
#include <string>
#include <utility>

using torc::Variable;
using torc::Tensor;
using torc::nn::Module;
using torc::nn::Sequential;
using torc::nn::Linear;

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
    for (const auto& p : params) {
        EXPECT_TRUE(p.requires_grad());
    }
}

TEST(Sequential, EmptyForwardReturnsInput) {
    Sequential seq;
    Variable input(5.0f, false);
    Variable output = seq(input);
    EXPECT_FLOAT_EQ(output.data().data()[0], 5.0f);
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
    Linear linear(2, 3);
    
    // Default weight is 0.01, bias is 0
    // Input x = [1.0f, 2.0f], shape (1, 2)
    // x @ W.T() = [1*0.01 + 2*0.01, 1*0.01 + 2*0.01, 1*0.01 + 2*0.01] = [0.03, 0.03, 0.03]
    // + bias = [0.03, 0.03, 0.03]
    Tensor input_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable input(input_data, false);
    
    Variable output = linear(input);
    
    std::vector<int> expected_output_shape{1, 3};
    ASSERT_EQ(output.data().shape(), expected_output_shape);
    EXPECT_FLOAT_EQ(output.data().data()[0], 0.03f);
    EXPECT_FLOAT_EQ(output.data().data()[1], 0.03f);
    EXPECT_FLOAT_EQ(output.data().data()[2], 0.03f);
}

TEST(Linear, ParametersAreTracked) {
    Linear linear(2, 3);
    auto params = linear.parameters();
    ASSERT_EQ(params.size(), 2);
    for (const auto& p : params) {
        EXPECT_TRUE(p.requires_grad());
    }
}

TEST(Linear, ForwardWithBatchedInput) {
    Linear linear(2, 3);
    
    // Default weight is 0.01, bias is 0
    // Batch of 2 samples: [[1, 2], [3, 4]]
    // [[1,2], [3,4]] @ [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
    // = [[0.03, 0.03, 0.03], [0.07, 0.07, 0.07]]
    Tensor input_data({1.0f, 2.0f, 3.0f, 4.0f}, std::vector<int>{2, 2});
    Variable input(input_data, false);
    
    Variable output = linear(input);
    
    std::vector<int> expected_output_shape{2, 3};
    ASSERT_EQ(output.data().shape(), expected_output_shape);
    EXPECT_FLOAT_EQ(output.data().data()[0], 0.03f);
    EXPECT_FLOAT_EQ(output.data().data()[1], 0.03f);
    EXPECT_FLOAT_EQ(output.data().data()[2], 0.03f);
    EXPECT_FLOAT_EQ(output.data().data()[3], 0.07f);
    EXPECT_FLOAT_EQ(output.data().data()[4], 0.07f);
    EXPECT_FLOAT_EQ(output.data().data()[5], 0.07f);
}

TEST(Linear, GradCheckForwardAndBackward) {
    Linear linear(2, 3);
    
    // x = [1.0f, 2.0f], shape (1, 2)
    Tensor x_data({1.0f, 2.0f}, std::vector<int>{1, 2});
    Variable x(x_data, true);
    
    Variable out = linear(x);
    Variable loss = torc::sum(out);
    
    EXPECT_FLOAT_EQ(loss.data().data()[0], 0.09f);
    
    loss.backward();
    
    const Tensor& W_grad = linear.named_parameters().at("weight").grad();
    const Tensor& b_grad = linear.named_parameters().at("bias").grad();
    
    ASSERT_TRUE(linear.named_parameters().at("weight").has_grad());
    ASSERT_TRUE(linear.named_parameters().at("bias").has_grad());
    
    // dL/dW[j,l] = sum_i x[i,l] = x[0,l] for batch=1
    expect_near(W_grad.data()[0], 1.0f);
    expect_near(W_grad.data()[1], 2.0f);
    expect_near(W_grad.data()[2], 1.0f);
    expect_near(W_grad.data()[3], 2.0f);
    expect_near(W_grad.data()[4], 1.0f);
    expect_near(W_grad.data()[5], 2.0f);
    
    // dL/db[j] = 1 for each j
    expect_near(b_grad.data()[0], 1.0f);
    expect_near(b_grad.data()[1], 1.0f);
    expect_near(b_grad.data()[2], 1.0f);
}

TEST(Linear, GradCheckWithBatchedInput) {
    Linear linear(2, 3);
    
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
