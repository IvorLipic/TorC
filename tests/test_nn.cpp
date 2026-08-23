#include <gtest/gtest.h>
#include "torc/nn.hpp"
#include <string>
#include <utility>

using torc::Variable;
using torc::Tensor;
using torc::nn::Module;
using torc::nn::Sequential;

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
