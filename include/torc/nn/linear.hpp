// linear.hpp
#pragma once
#include "torc/nn.hpp"
#include <vector>

namespace torc::nn {

class Linear : public Module {
public:
    Linear(int in_features, int out_features);
    
    Variable forward(const Variable& x) const override;
    
private:
    int in_features_;
    int out_features_;
};

} // namespace torc::nn
