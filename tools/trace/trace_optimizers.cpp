// trace_optimizers.cpp  ->  site/data/optimizers.json   (lesson `optimizers`)
//
// 1. A narrow valley L(w) = 0.5 w1^2 + 50 w2^2 (curvatures 1 and 100), built from TorC ops on one parameter w of shape (2).
//    Three runs of 50 steps from w = (-4, 0.4): optim::SGD (lr 0.019, just below the stable limit 2/100), optim::SGD with
//    momentum 0.9 (same lr) and optim::Adam (lr 0.2). Each step records w and the loss. A grid of losses from the same forward
//    pass draws the contour lines.
// 2. The first step of Adam and of SGD (both lr 0.1) on two parameters whose gradients are 0.001 and 1000.
// 3. Weight decay: Adam and AdamW (lr 0.1, weight_decay 0.1) on L = (w1 - 2)^2 + 0 * w2^2 from w = (0, 3), plus a parameter u = 3
//    that the loss does not use (it has no gradient, so step() skips it). 50 steps.
// Everything is library output, except `derived` (plain arithmetic).
#include "trace_util.hpp"

#include "torc/optim.hpp"

#include <cmath>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

constexpr int STEPS = 50;
const std::vector<float> HALF_CURV = {0.5f, 50.0f};    // L = sum(c * w * w): curvatures 1 and 100

// The ops are written out where they are used: every intermediate Variable must live until backward() (TorC's tape holds raw
// pointers to them), so a helper that returns only the loss would leave the graph with expired inputs.
float valley_at(float w1, float w2) {
    Variable::NoGradGuard ng;
    const Variable w(trace::make_tensor({2}, {w1, w2}), false), c(trace::make_tensor({2}, HALF_CURV), false);
    Variable sq = mul(w, w);
    Variable cs = mul(c, sq);
    return sum(cs).data().data()[0];
}

J point(const Variable& w, float loss) {
    J p = J::object();
    p.set("w1", w.data().data()[0]).set("w2", w.data().data()[1]).set("loss", loss);
    return p;
}

template <class Opt>
J valley_run(const char* id, const char* name, float lr, float momentum, Opt& opt, Variable& w) {
    J path = J::array();
    const Variable c(trace::make_tensor({2}, HALF_CURV), false);
    for (int k = 0; k <= STEPS; ++k) {
        Variable sq = mul(w, w);
        Variable cs = mul(c, sq);
        Variable L = sum(cs);
        path.push(point(w, L.data().data()[0]));
        if (k == STEPS) break;
        L.backward();
        opt.step();
        opt.zero_grad();
    }
    J r = J::object();
    r.set("id", id).set("optimizer", name).set("lr", static_cast<double>(lr)).set("momentum", static_cast<double>(momentum))
        .set("path", std::move(path));
    return r;
}

template <class Opt>
J decay_run(const char* name, Opt& opt, Variable& w, const Variable& u) {
    J steps = J::array();
    // L = (w1 - 2)^2 + 0 * w2^2: w2 is in the graph, so it has a gradient, and the gradient is zero
    const Variable t(trace::make_tensor({2}, {2.0f, 0.0f}), false), c(trace::make_tensor({2}, {1.0f, 0.0f}), false);
    // steps[k] = the values before step k + 1, and what backward() gave just before that step (zero_grad() at the end of each
    // step clears has_grad, so the gradient fields are read between backward() and step()). The last entry has no step after it.
    for (int k = 0; k <= STEPS; ++k) {
        J s = J::object();
        s.set("step", k).set("w1", w.data().data()[0]).set("w2", w.data().data()[1]).set("u", u.data().data()[0]);
        if (k == STEPS) { steps.push(std::move(s)); break; }
        Variable d = sub(w, t);
        Variable sq = mul(d, d);
        Variable cs = mul(c, sq);
        Variable L = sum(cs);
        L.backward();
        s.set("w_has_grad", w.has_grad()).set("w2_grad", w.grad().data()[1]).set("u_has_grad", u.has_grad());
        steps.push(std::move(s));
        opt.step();
        opt.zero_grad();
    }
    J r = J::object();
    r.set("optimizer", name).set("steps", std::move(steps));
    return r;
}

}  // namespace

int main() {
    // ---- 1. the valley ------------------------------------------------------------------------------------------------------
    const std::vector<float> start = {-4.0f, 0.4f};
    J runs = J::array();
    {
        Variable w(trace::make_tensor({2}, start), true);
        optim::SGD opt({&w}, 0.019f);
        runs.push(valley_run("sgd", "optim::SGD", 0.019f, 0.0f, opt, w));
    }
    {
        Variable w(trace::make_tensor({2}, start), true);
        optim::SGD opt({&w}, 0.019f, 0.9f);
        runs.push(valley_run("momentum", "optim::SGD", 0.019f, 0.9f, opt, w));
    }
    {
        Variable w(trace::make_tensor({2}, start), true);
        optim::Adam opt({&w}, 0.2f);
        runs.push(valley_run("adam", "optim::Adam", 0.2f, 0.0f, opt, w));
    }
    // the loss on a grid, built from integers: w1 = -4.5 .. 1.5 (step 0.1), w2 = -0.6 .. 0.6 (step 0.025)
    J g1 = J::array(), g2 = J::array(), grid = J::array();
    std::vector<float> w1s, w2s;
    for (int i = 0; i <= 60; ++i) w1s.push_back(static_cast<float>(i - 45) / 10.0f);
    for (int j = 0; j <= 48; ++j) w2s.push_back(static_cast<float>(j - 24) / 40.0f);
    for (float v : w1s) g1.push(v);
    for (float v : w2s) g2.push(v);
    for (float b : w2s) {
        J row = J::array();
        for (float a : w1s) row.push(valley_at(a, b));
        grid.push(std::move(row));
    }
    J surface = J::object();
    surface.set("w1", std::move(g1)).set("w2", std::move(g2)).set("order", "loss[j][i] at (w1[i], w2[j])").set("loss", std::move(grid));
    J valley = J::object();
    valley.set("curvature", trace::jarr(std::vector<float>{2 * HALF_CURV[0], 2 * HALF_CURV[1]})).set("start", trace::jarr(start))
        .set("steps", STEPS).set("runs", std::move(runs)).set("surface", std::move(surface));

    // ---- 2. the first step ------------------------------------------------------------------------------------------------------
    J first = J::array();
    auto first_step = [&](const char* name, auto make) {
        Variable w(trace::make_tensor({2}, {1.0f, 1.0f}), true);
        auto opt = make(w);
        const Variable c(trace::make_tensor({2}, {0.001f, 1000.0f}), false);
        Variable cw = mul(c, w);
        Variable L = sum(cw);                                            // gradient = c
        L.backward();
        const std::vector<float> g = trace::values_of(w.grad());
        opt.step();
        J e = J::object();
        e.set("optimizer", name).set("lr", 0.1).set("grad", trace::jarr(g))
            .set("change", trace::jarr(std::vector<float>{w.data().data()[0] - 1.0f, w.data().data()[1] - 1.0f}));
        first.push(std::move(e));
    };
    first_step("optim::SGD", [](Variable& w) { return optim::SGD({&w}, 0.1f); });
    first_step("optim::Adam", [](Variable& w) { return optim::Adam({&w}, 0.1f); });

    // ---- 3. weight decay ----------------------------------------------------------------------------------------------------------
    J decay = J::array();
    {
        Variable w(trace::make_tensor({2}, {0.0f, 3.0f}), true), u(trace::make_tensor({1}, {3.0f}), true);
        optim::Adam opt({&w, &u}, 0.1f);
        decay.push(decay_run("optim::Adam", opt, w, u));
    }
    {
        Variable w(trace::make_tensor({2}, {0.0f, 3.0f}), true), u(trace::make_tensor({1}, {3.0f}), true);
        optim::AdamW opt({&w, &u}, 0.1f, 0.1f);
        decay.push(decay_run("optim::AdamW", opt, w, u));
    }
    J decay_setup = J::object();
    decay_setup.set("lr", 0.1).set("weight_decay", 0.1).set("target", 2).set("start", trace::jarr(std::vector<float>{0.0f, 3.0f, 3.0f}));

    J derived = J::object();
    derived.set("sgd_lr_limit", 2.0 / 100.0).set("decay_factor", 1.0 - 0.1 * 0.1)
        .set("w2_after_50_adamw", 3.0 * std::pow(1.0 - 0.1 * 0.1, 50));

    J doc = trace::envelope("optimizers", "optimizers", "tools/trace/trace_optimizers.cpp");
    doc.set("valley", std::move(valley)).set("first_step", std::move(first)).set("decay_setup", std::move(decay_setup))
        .set("decay", std::move(decay)).set("derived", std::move(derived));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
