// trace_modules.cpp  ->  site/data/modules.json   (lesson `modules`)
//
// 1. Four nn::Sequential models on an input of shape (8, 784): for each module its parameters (name, shape, size) and its
//    output shape, or the error TorC throws; the total from Sequential::parameters().
// 2. The order of calls: forward, loss, backward and a second forward, in three orders, with the result of each (the
//    gradient shape, or the exception). A module clears the values of its last forward pass when it is called again.
// The weights are random (nn::Linear's default), but no weight value is recorded: only shapes, counts and messages.
#include "trace_util.hpp"

#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"

#include <algorithm>
#include <iostream>
#include <memory>

using namespace torc;
using trace::J;

namespace {

struct Spec { std::string kind; int in = 0, out = 0; };   // kind "Linear" or "ReLU"

std::unique_ptr<nn::Module> make(const Spec& s) {
    if (s.kind == "Linear") return std::make_unique<nn::Linear>(s.in, s.out, 0.1f, 1);
    return std::make_unique<nn::ReLU>();
}
std::string label(const Spec& s) { return s.kind == "Linear" ? std::format("Linear({}, {})", s.in, s.out) : "ReLU"; }

J params_json(const nn::Module& m, int& total, int& tensors) {
    std::vector<std::pair<std::string, const Variable*>> ps;
    for (const auto& [name, v] : m.named_parameters()) ps.emplace_back(name, &v);
    std::ranges::sort(ps, {}, &std::pair<std::string, const Variable*>::first);      // by name, so the file does not depend on hashing
    J arr = J::array();
    for (const auto& [name, v] : ps) {
        J p = J::object();
        p.set("name", name).set("shape", trace::jarr(v->data().shape())).set("numel", v->data().numel());
        arr.push(std::move(p));
        total += v->data().numel();
        ++tensors;
    }
    return arr;
}

J model_json(const char* id, const std::vector<Spec>& specs) {
    const Tensor x0(std::vector<int>{8, 784});
    J layers = J::array();
    int total = 0, tensors = 0;
    Variable cur(x0, false);
    bool failed = false;
    std::vector<std::unique_ptr<nn::Module>> keep;                          // keep each module (and its cached forward) alive
    for (const auto& s : specs) {
        keep.push_back(make(s));
        const nn::Module& m = *keep.back();
        J l = J::object();
        l.set("label", label(s)).set("params", params_json(m, total, tensors));
        if (failed) { l.set("out", "not reached"); layers.push(std::move(l)); continue; }
        auto c = trace::catch_call([&] { cur = m(cur); });
        if (c.type == "none") l.set("out_shape", trace::jarr(cur.data().shape()));
        else { l.set("exception", c.type).set("what", c.what); failed = true; }
        layers.push(std::move(l));
    }
    // the same model as one Sequential: its parameters() and its forward
    nn::Sequential seq;
    for (const auto& s : specs) seq.add(make(s));
    int seq_total = 0;
    for (Variable* p : seq.parameters()) seq_total += p->data().numel();
    J o = J::object();
    o.set("id", id).set("input_shape", trace::jarr(x0.shape())).set("layers", std::move(layers)).set("total", total).set("tensors", tensors)
        .set("sequential_parameters_total", seq_total).set("sequential_parameters_count", static_cast<int>(seq.parameters().size()));
    const Variable xin(x0, false);
    auto c = trace::catch_call([&] { (void)seq(xin); });
    if (c.type == "none") { Variable y = seq(xin); o.set("sequential_out_shape", trace::jarr(y.data().shape())); }
    else o.set("sequential_exception", c.type).set("sequential_what", c.what);
    return o;
}

// three orders of calls on the same small model; returns the outcome of the last backward() in each
J orders_json() {
    J arr = J::array();
    auto run = [&](const char* id, const char* code, int order) {
        nn::Sequential model;
        model.add(std::make_unique<nn::Linear>(3, 2, 0.1f, 1));
        model.add(std::make_unique<nn::ReLU>());
        model.add(std::make_unique<nn::Linear>(2, 1, 0.1f, 1));
        nn::MSELoss mse;
        const Variable x1(trace::make_tensor({4, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), false);
        const Variable x2(trace::make_tensor({4, 3}, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}), false);
        const Variable t(trace::make_tensor({4, 1}, {1, 0, 1, 0}), false);
        trace::Caught c;
        if (order == 0) {            // forward, backward, then the next forward and its backward
            Variable y1 = model(x1); Variable L1 = mse(y1, t);
            c = trace::catch_call([&] { L1.backward(); });
            if (c.type == "none") { Variable y2 = model(x2); Variable L2 = mse(y2, t); c = trace::catch_call([&] { L2.backward(); }); }
        } else if (order == 1) {     // a second forward before the first backward, then the first backward
            Variable y1 = model(x1); Variable L1 = mse(y1, t);
            Variable y2 = model(x2); Variable L2 = mse(y2, t);
            c = trace::catch_call([&] { L1.backward(); });
        } else {                     // a second forward, then the backward of the second loss
            Variable y1 = model(x1); Variable L1 = mse(y1, t);
            Variable y2 = model(x2); Variable L2 = mse(y2, t);
            c = trace::catch_call([&] { L2.backward(); });
        }
        J o = J::object();
        o.set("id", id).set("code", code).set("exception", c.type);
        if (c.type != "none") o.set("what", c.what);
        arr.push(std::move(o));
    };
    run("in-order", "Variable y1 = model(x1);\nVariable L1 = mse(y1, t);\nL1.backward();\nVariable y2 = model(x2);\nVariable L2 = mse(y2, t);\nL2.backward();", 0);
    run("early-second-forward", "Variable y1 = model(x1);\nVariable L1 = mse(y1, t);\nVariable y2 = model(x2);\nVariable L2 = mse(y2, t);\nL1.backward();", 1);
    run("latest-backward", "Variable y1 = model(x1);\nVariable L1 = mse(y1, t);\nVariable y2 = model(x2);\nVariable L2 = mse(y2, t);\nL2.backward();", 2);
    return arr;
}

}  // namespace

int main() {
    J models = J::array();
    models.push(model_json("mlp", {{"Linear", 784, 32}, {"ReLU"}, {"Linear", 32, 10}}))
        .push(model_json("mnist", {{"Linear", 784, 32}, {"ReLU"}, {"Linear", 32, 32}, {"ReLU"}, {"Linear", 32, 10}}))
        .push(model_json("no-relu", {{"Linear", 784, 32}, {"Linear", 32, 10}}))
        .push(model_json("mismatch", {{"Linear", 784, 32}, {"ReLU"}, {"Linear", 64, 10}}));
    J doc = trace::envelope("modules", "modules", "tools/trace/trace_modules.cpp");
    doc.set("models", std::move(models)).set("orders", orders_json());
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
