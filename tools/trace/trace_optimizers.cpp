// trace_optimizers.cpp  ->  site/data/optimizers.json   (lesson `optimizers`)
//
// 1. A narrow valley L(w) = 0.5 w1^2 + 50 w2^2 (curvatures 1 and 100), built from TorC ops on one parameter w of shape (2).
//    Three runs of 50 steps from w = (-4, 0.4): optim::SGD (lr 0.019, just below the stable limit 2/100), optim::SGD with
//    momentum 0.9 (same lr) and optim::Adam (lr 0.2). Each step records w and the loss. A grid of losses from the same forward
//    pass draws the contour lines.
// 2. The first step of Adam and of SGD (both lr 0.1) on two parameters whose gradients are 0.001 and 1000.
// 3. Weight decay: Adam and AdamW (lr 0.1, weight_decay 0.1) on L = (w1 - 2)^2 + 0 * w2^2 from w = (0, 3), plus a parameter u = 3
//    that the loss does not use (it has no gradient, so step() skips it). 50 steps.
// 4. (v3, brief C4) `sweep`: on the same valley, start and 50 steps, a learning-rate sweep for each optimizer (optim::SGD,
//    optim::SGD with momentum 0.9, optim::Adam, optim::AdamW with weight_decay 0.01, TorC's default), lr = 0.001 ... 1. Each run
//    records its final loss and whether it diverged (final loss above the start loss, or the run stopped because w overflowed and
//    TorC's sum() threw NumericalError; final_loss is then null and the exception is recorded), and `best` is the lr with the
//    lowest final loss among the runs that did not diverge.
//    `race`: the four optimizers, each at its own best lr, with the full path in the format of valley.runs.
// Everything is library output, except `derived` (plain arithmetic) and the choice of `best` (a comparison of recorded losses).
#include "trace_util.hpp"

#include "torc/optim.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>

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


// ---- 4. sweep ---------------------------------------------------------------------------------------------------------------
// A type-erased optimizer, so the four kinds share one loop.
struct AnyOpt {
    std::function<void()> step, zero_grad;
};

struct Kind {
    const char* id;
    const char* optimizer;
    float momentum, weight_decay;
    std::function<std::shared_ptr<void>(Variable&, float, AnyOpt&)> make;  // builds the optimizer on w; the handle keeps it alive
};

template <class O>
std::shared_ptr<void> hold(std::shared_ptr<O> o, AnyOpt& a) {
    a.step = [o] { o->step(); };
    a.zero_grad = [o] { o->zero_grad(); };
    return o;
}

std::vector<Kind> kinds() {
    return {
        {"sgd", "optim::SGD", 0.0f, 0.0f, [](Variable& w, float lr, AnyOpt& a) { return hold(std::make_shared<optim::SGD>(std::vector<Variable*>{&w}, lr), a); }},
        {"momentum", "optim::SGD", 0.9f, 0.0f, [](Variable& w, float lr, AnyOpt& a) { return hold(std::make_shared<optim::SGD>(std::vector<Variable*>{&w}, lr, 0.9f), a); }},
        {"adam", "optim::Adam", 0.0f, 0.0f, [](Variable& w, float lr, AnyOpt& a) { return hold(std::make_shared<optim::Adam>(std::vector<Variable*>{&w}, lr), a); }},
        {"adamw", "optim::AdamW", 0.0f, 0.01f, [](Variable& w, float lr, AnyOpt& a) { return hold(std::make_shared<optim::AdamW>(std::vector<Variable*>{&w}, lr, 0.01f), a); }},
    };
}

// 50 steps on the valley from `start`: the loss before each step and after the last one (51 values when the run completes)
// and the matching w. When w overflows, TorC's sum() throws NumericalError ("sum requires finite input values"); the run then
// stops, and `error` holds the type and what() text and `losses` ends at the last finite loss.
// The ops are written out here so that every Variable of the graph lives until backward().
struct SweepRun {
    std::vector<float> losses;
    std::vector<std::pair<float, float>> ws;
    trace::Caught error{"none", ""};
};

SweepRun sweep_run(const Kind& k, float lr, const std::vector<float>& start) {
    Variable w(trace::make_tensor({2}, start), true);
    AnyOpt opt;
    auto keep = k.make(w, lr, opt);
    const Variable c(trace::make_tensor({2}, HALF_CURV), false);
    SweepRun r;
    try {
        for (int s = 0; s <= STEPS; ++s) {
            Variable sq = mul(w, w);
            Variable cs = mul(c, sq);
            Variable L = sum(cs);
            r.losses.push_back(L.data().data()[0]);
            r.ws.emplace_back(w.data().data()[0], w.data().data()[1]);
            if (s == STEPS) break;
            L.backward();
            opt.step();
            opt.zero_grad();
        }
    } catch (const torc::NumericalError& e) {  // any other exception is not expected and ends the program
        r.error = {"NumericalError", e.what()};
    }
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

    // ---- 4. learning-rate sweep and a race at each optimizer's best lr (C4) -----------------------------------------------------
    const std::vector<double> SWEEP_LRS = {0.001, 0.003, 0.01, 0.019, 0.03, 0.1, 0.2, 0.3, 1.0, 3.0};
    J sweep = J::object();
    J race = J::array();
    {
        J per = J::array();
        for (const Kind& k : kinds()) {
            J runs_j = J::array();
            double best_lr = -1.0;
            float best_loss = std::numeric_limits<float>::infinity();
            for (double lrd : SWEEP_LRS) {
                const SweepRun R = sweep_run(k, static_cast<float>(lrd), start);
                const std::vector<float>& L = R.losses;
                const bool completed = R.error.type == "none" && L.size() == static_cast<size_t>(STEPS + 1);
                const float fin = L.back();
                const bool finite = completed && std::isfinite(fin);
                const bool diverged = !finite || fin > L.front();
                float lowest = L.front();
                for (float v : L) if (std::isfinite(v)) lowest = std::fmin(lowest, v);
                J r = J::object();
                r.set("lr", lrd).set("final_loss", finite ? J(fin) : J()).set("finite", finite).set("diverged", diverged).set("lowest_loss", lowest)
                    .set("steps_completed", static_cast<int>(L.size()) - 1).set("exception", R.error.type).set("what", R.error.what);
                runs_j.push(std::move(r));
                if (!diverged && fin < best_loss) { best_loss = fin; best_lr = lrd; }
            }
            J o = J::object();
            o.set("id", k.id).set("optimizer", k.optimizer).set("momentum", static_cast<double>(k.momentum))
                .set("weight_decay", static_cast<double>(k.weight_decay)).set("runs", std::move(runs_j))
                .set("best", [&] { J b = J::object(); b.set("lr", best_lr).set("final_loss", best_loss); return b; }());
            per.push(std::move(o));
            // the race: the same optimizer at its best lr, the full path
            const SweepRun R = sweep_run(k, static_cast<float>(best_lr), start);
            const std::vector<float>& L = R.losses;
            const auto& ws = R.ws;
            J path = J::array();
            for (size_t i = 0; i < L.size(); ++i) {
                J p = J::object();
                p.set("w1", ws[i].first).set("w2", ws[i].second).set("loss", L[i]);
                path.push(std::move(p));
            }
            J rr = J::object();
            rr.set("id", k.id).set("optimizer", k.optimizer).set("lr", best_lr).set("momentum", static_cast<double>(k.momentum))
                .set("weight_decay", static_cast<double>(k.weight_decay)).set("final_loss", L.back()).set("path", std::move(path));
            race.push(std::move(rr));
        }
        sweep.set("lrs", trace::jarr(SWEEP_LRS)).set("start", trace::jarr(start)).set("steps", STEPS)
            .set("start_loss", valley_at(start[0], start[1]))
            .set("diverged_rule", "final loss above the start loss, or the run did not finish its 50 steps (TorC threw; final_loss is then null)")
            .set("best_rule", "the lr with the lowest final loss among the runs that did not diverge")
            .set("optimizers", std::move(per));
    }

    J derived = J::object();
    derived.set("sgd_lr_limit", 2.0 / 100.0).set("decay_factor", 1.0 - 0.1 * 0.1)
        .set("w2_after_50_adamw", 3.0 * std::pow(1.0 - 0.1 * 0.1, 50));

    J doc = trace::envelope("optimizers", "optimizers", "tools/trace/trace_optimizers.cpp");
    doc.set("valley", std::move(valley)).set("first_step", std::move(first)).set("decay_setup", std::move(decay_setup))
        .set("decay", std::move(decay)).set("derived", std::move(derived))
        .set("sweep", std::move(sweep)).set("race", std::move(race));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
