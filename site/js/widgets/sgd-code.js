// widgets/sgd-code.js — short quotations of TorC's real source for the sgd-step lesson (no DOM).
//
// SOURCES holds the lines EXACTLY as they are in the library at the pinned ref (tools/trace/TORC_REF); `from`/`to` are 1-based file
// line numbers. site/test/sgd-step.test.mjs compares every quoted line with the fetched library source (build/trace/_deps/torc-src/)
// when that directory exists. Never edit a quoted line by hand: re-copy it from the library.

export const SOURCES = {
  "loop": {
    "file": "examples/linear_regression/linear_regression.cpp",
    "from": 66,
    "to": 68,
    "lines": [
      "            loss.backward();",
      "            optimizer.step();",
      "            optimizer.zero_grad();"
    ]
  },
  "step": {
    "file": "src/optim.cpp",
    "from": 17,
    "to": 30,
    "lines": [
      "void SGD::step() {",
      "    for (size_t i = 0; i < params_.size(); ++i) {",
      "        torc::Variable* param = params_[i];",
      "        if (!param->has_grad()) continue;",
      "",
      "        const torc::Tensor& grad = param->grad();",
      "        if (momentum_ != 0.0f) {",
      "            velocities_[i] = velocities_[i].mul(momentum_).add(grad);",
      "            param->data() = param->data().sub(velocities_[i].mul(lr_));",
      "        } else {",
      "            param->data() = param->data().sub(grad.mul(lr_));",
      "        }",
      "    }",
      "}"
    ]
  },
  "accumulate": {
    "file": "include/torc/autograd.hpp",
    "from": 75,
    "to": 85,
    "lines": [
      "    void accumulate_grad(const Tensor& local_grad) {",
      "        if (!has_grad_) {",
      "            grad_ = Tensor(std::vector<int>(local_grad.shape().begin(), local_grad.shape().end()));",
      "            for (int i = 0; i < local_grad.numel(); ++i)",
      "                grad_.data()[i] = local_grad.data()[i];",
      "            has_grad_ = true;",
      "        } else {",
      "            for (int i = 0; i < grad_.numel(); ++i)",
      "                grad_.data()[i] += local_grad.data()[i];",
      "        }",
      "    }"
    ]
  }
};

