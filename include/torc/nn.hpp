// nn.hpp
#pragma once
#include "torc/autograd.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>

namespace torc::nn {

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    virtual Variable forward(const Variable& x) const = 0;
    
    Variable operator()(const Variable& x) const {
        return forward(x);
    }
    
    void register_parameter(const std::string& name, Variable param) {
        named_params_.emplace(name, std::move(param));
    }
    
    const std::unordered_map<std::string, Variable>& named_parameters() const {
        return named_params_;
    }
    
    std::unordered_map<std::string, Variable>& named_parameters() {
        return named_params_;
    }
    
    virtual std::vector<Variable> parameters() const {
        std::vector<Variable> result;
        result.reserve(named_params_.size());
        for (const auto& [name, param] : named_params_) {
            result.push_back(param);
        }
        return result;
    }
    
private:
    std::unordered_map<std::string, Variable> named_params_;
};

class Sequential : public Module {
public:
    Sequential() = default;
    
    void add(std::unique_ptr<Module> module) {
        modules_.push_back(std::move(module));
    }
    
    Variable forward(const Variable& x) const override {
        Variable out = x;
        for (const auto& module : modules_) {
            out = module->forward(out);
        }
        return out;
    }
    
    std::vector<Variable> parameters() const override {
        std::vector<Variable> result = Module::parameters();
        for (const auto& module : modules_) {
            auto child_params = module->parameters();
            result.insert(result.end(),
                         std::make_move_iterator(child_params.begin()),
                         std::make_move_iterator(child_params.end()));
        }
        return result;
    }
    
private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace torc::nn
