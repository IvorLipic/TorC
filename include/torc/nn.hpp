// nn.hpp
#pragma once
#include "torc/autograd.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <list>

namespace torc::nn {

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    virtual Variable forward(const Variable& x) const = 0;
    
    Variable operator()(const Variable& x) const {
        forward_cache_.clear();
        return forward(x);
    }
    
    void register_parameter(const std::string& name, Variable param);
    
    const std::unordered_map<std::string, Variable>& named_parameters() const;
    std::unordered_map<std::string, Variable>& named_parameters();
    
    virtual std::vector<Variable> parameters() const;
    
private:
    std::unordered_map<std::string, Variable> named_params_;
    
protected:
    mutable std::list<Variable> forward_cache_;
};

class Sequential : public Module {
public:
    Sequential() = default;
    
    void add(std::unique_ptr<Module> module);
    
    Variable forward(const Variable& x) const override;
    std::vector<Variable> parameters() const override;
    
private:
    std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace torc::nn
