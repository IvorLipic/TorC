// nn.cpp
#include "torc/nn.hpp"

namespace torc::nn {

void Module::register_parameter(const std::string& name, Variable param) {
    named_params_.emplace(name, std::move(param));
}

const std::unordered_map<std::string, Variable>& Module::named_parameters() const {
    return named_params_;
}

std::unordered_map<std::string, Variable>& Module::named_parameters() {
    return named_params_;
}

std::vector<Variable*> Module::parameters() const {
    std::vector<Variable*> result;
    result.reserve(named_params_.size());
    for (const auto& [name, param] : named_params_) {
        result.push_back(const_cast<Variable*>(&param));
    }
    return result;
}

void Sequential::add(std::unique_ptr<Module> module) {
    modules_.push_back(std::move(module));
}

Variable Sequential::forward(const Variable& x) const {
    Variable out = x;
    for (const auto& module : modules_) {
        out = module->operator()(out);
    }
    return out;
}

std::vector<Variable*> Sequential::parameters() const {
    std::vector<Variable*> result = Module::parameters();
    for (const auto& module : modules_) {
        auto child_params = module->parameters();
        result.insert(result.end(),
                     std::make_move_iterator(child_params.begin()),
                     std::make_move_iterator(child_params.end()));
    }
    return result;
}

} // namespace torc::nn
