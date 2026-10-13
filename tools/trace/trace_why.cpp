// trace_why.cpp  ->  site/data/why-gradients.json   (lesson `why-gradients`, docs/content-plan.md 3.1)
//
// One weight, one datum:  p = w*x,  e = p - t,  L = e*e   with x = 2, t = 6  (so L(w) = (2w-6)^2 = 4(w-3)^2).
// Everything is computed by the real library: the forward values with torc::mul/sub, the slopes with Variable::backward,
// the weight updates with torc::optim::SGD on a one-element Variable.
//
// Quantities that are NOT library output are marked "derived" in the JSON (plain double arithmetic on recorded numbers,
// e.g. the factor 1 - 8*lr or the tangent-line prediction); they are there so lessons do not have to type them by hand.
#include "trace_util.hpp"

#include "torc/optim.hpp"

#include <cstdlib>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

constexpr float X = 2.0f;
constexpr float T = 6.0f;

struct Point {
    float w, p, e, L, grad;
};

// Real forward + real backward at weight w.
Point eval_why(float w) {
    Variable wv(Tensor({w}, {1}), true);
    Variable xv(Tensor({X}, {1}), false);
    Variable tv(Tensor({T}, {1}), false);
    Variable p = mul(wv, xv);
    Variable e = sub(p, tv);
    Variable L = mul(e, e);
    L.backward();
    return {w, p.data().data()[0], e.data().data()[0], L.data().data()[0], wv.grad().data()[0]};
}

// Real forward only (no tape).
Point eval_forward(float w) {
    Variable::NoGradGuard ng;
    Variable wv(Tensor({w}, {1}), true);
    Variable xv(Tensor({X}, {1}), false);
    Variable tv(Tensor({T}, {1}), false);
    Variable p = mul(wv, xv);
    Variable e = sub(p, tv);
    Variable L = mul(e, e);
    return {w, p.data().data()[0], e.data().data()[0], L.data().data()[0], 0.0f};
}

J point_json(const Point& q) {
    J o = J::object();
    o.set("w", q.w).set("p", q.p).set("e", q.e).set("L", q.L).set("grad", q.grad);
    return o;
}

}  // namespace

int main() {
    bool all_ok = true;

    // ---- Scene `why` in the node schema --------------------------------------------------------------------------
    trace::Scene s;
    s.id = "why";
    s.title = "One weight, one datum";
    s.expression = "L = (w*x - t)^2";
    s.note = "w is the only tunable number; x and t are fixed data. backward() returns dL/dw, the loss change per unit change of w.";
    s.leaves = {{"w", Tensor({0.0f}, {1})}, {"x", Tensor({X}, {1})}, {"t", Tensor({T}, {1})}};
    s.constants = {"x", "t"};
    s.build = [](trace::Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
        Variable& p = g.op("p", "mul", mul(*L["w"], *L["x"]));
        Variable& e = g.op("e", "sub", sub(p, *L["t"]));
        return g.op("L", "mul", mul(e, e));
    };
    trace::SceneResult scene = trace::run_scene(s);

    // ---- grid: w in [-1, 7] step 0.25, real forward + backward ---------------------------------------------------
    J grid = J::array();
    double max_formula_diff = 0.0;
    for (int i = 0; i <= 32; ++i) {
        float w = -1.0f + 0.25f * static_cast<float>(i);  // exactly representable in float32
        Point q = eval_why(w);
        grid.push(point_json(q));
        max_formula_diff = std::fmax(max_formula_diff, std::fabs(static_cast<double>(q.grad) - 8.0 * (static_cast<double>(w) - 3.0)));
        max_formula_diff = std::fmax(max_formula_diff, std::fabs(static_cast<double>(q.L) - 4.0 * (static_cast<double>(w) - 3.0) * (static_cast<double>(w) - 3.0)));
    }

    // ---- probes: the weights the lesson talks about --------------------------------------------------------------
    J probes = J::array();
    for (float w : {0.0f, 0.1f, 1.0f, 1.2f, 3.0f, 6.0f, 7.2f}) probes.push(point_json(eval_why(w)));

    // ---- secants at w = 0: (L(h) - L(0)) / h, with L from the real forward pass ----------------------------------
    const Point p0 = eval_why(0.0f);
    J secants = J::array();
    for (double hd : {1.0, 0.5, 0.1, 0.01, 0.001}) {
        float h = static_cast<float>(hd);
        Point ph = eval_forward(h);
        float dL = ph.L - p0.L;  // float32 difference, as a learner's calculator-in-float32 would see it
        J o = J::object();
        o.set("h", hd).set("w0", p0.w).set("L0", p0.L).set("L_h", ph.L).set("dL", dL).set("slope", dL / h)
            .set("expected_slope", 4.0 * hd - 24.0)  // derived: (4h^2 - 24h)/h
            .set("backward_slope", p0.grad);
        secants.push(std::move(o));
    }

    // ---- nudge: change of every stage when w moves by delta = 0.01 (chain rule as multiplied local slopes) --------
    J nudge = J::object();
    {
        const float delta = 0.01f;
        Point a = p0, b = eval_forward(delta);
        auto g_of = [&](const char* name) { return scene.grad.at(name).data()[0]; };
        const float g_w = g_of("w"), g_p = g_of("p"), g_e = g_of("e");
        J stages = J::array();
        // local slope of a stage = (gradient at its input) / (gradient at its output), read from the real backward pass.
        auto stage = [&](const char* name, const char* from, float before, float after, float local) {
            J o = J::object();
            o.set("name", name).set("from", from).set("before", before).set("after", after).set("change", after - before).set("local_slope", local);
            stages.push(std::move(o));
        };
        stage("p", "w", a.p, b.p, g_w / g_p);
        stage("e", "p", a.e, b.e, g_p / g_e);
        stage("L", "e", a.L, b.L, g_e / 1.0f);  // the root's own seed gradient is 1 (root.grad is not stored)
        nudge.set("w0", a.w).set("delta", 0.01).set("w1", b.w).set("stages", std::move(stages));
        nudge.set("local_slope_product", g_w / g_p * (g_p / g_e) * g_e);  // derived: product of the three local slopes
        nudge.set("backward_slope", g_w);
        nudge.set("predicted_dL", g_w * delta);                          // derived: slope * delta
        nudge.set("actual_dL", b.L - a.L);
        nudge.set("second_order_term", (b.L - a.L) - g_w * delta);      // derived: what the straight-line promise misses
    }

    // ---- trajectories: real optim::SGD on a one-element Variable -------------------------------------------------
    J trajectories = J::array();
    bool all_match = true;
    for (double lrd : {0.05, 0.1, 0.125, 0.25, 0.3}) {
        const float lr = static_cast<float>(lrd);
        Variable w(Tensor({0.0f}, {1}), true);
        optim::SGD opt({&w}, lr);
        J states = J::array();
        bool match = true;
        float max_diff = 0.0f;
        for (int k = 0; k <= 8; ++k) {
            Variable xv(Tensor({X}, {1}), false);
            Variable tv(Tensor({T}, {1}), false);
            Variable p = mul(w, xv);
            Variable e = sub(p, tv);
            Variable L = mul(e, e);
            L.backward();
            const float wk = w.data().data()[0], Lk = L.data().data()[0], gk = w.grad().data()[0];
            const float manual_next = wk - lr * gk;  // the textbook rule, in float32
            J st = J::object();
            st.set("k", k).set("w", wk).set("p", p.data().data()[0]).set("e", e.data().data()[0]).set("L", Lk).set("grad", gk);
            st.set("w_next", manual_next);  // where one more step would land
            // derived: the straight-line (tangent) promise for L at w_next, and what the library then actually gives
            st.set("tangent_L_next", static_cast<float>(static_cast<double>(Lk) + static_cast<double>(gk) * (static_cast<double>(manual_next) - static_cast<double>(wk))));
            states.push(std::move(st));
            opt.step();  // real SGD::step: reads w.grad(), rewrites w.data()
            opt.zero_grad();
            const float after = w.data().data()[0];
            max_diff = std::fmax(max_diff, std::fabs(after - manual_next));
            if (after != manual_next) match = false;
        }
        all_match = all_match && match;
        const double factor = std::round((1.0 - 8.0 * lrd) * 1e12) / 1e12;  // derived: distance to w = 3 is multiplied by this per step
        J tr = J::object();
        tr.set("lr", lrd).set("momentum", 0).set("factor", factor).set("stable_predicted", std::fabs(factor) < 1.0)
            .set("matches_manual", match).set("max_abs_diff_vs_manual", max_diff).set("states", std::move(states));
        trajectories.push(std::move(tr));
    }
    all_ok = all_ok && all_match;

    // ---- assemble ------------------------------------------------------------------------------------------------
    J setup = J::object();
    setup.set("x", X).set("t", T).set("w_min", 3).set("loss_formula", "L(w) = (x*w - t)^2 = 4(w-3)^2")
        .set("grad_formula", "dL/dw = 2*x*(x*w - t) = 8(w-3)").set("critical_lr", 0.25);  // derived (Expected mathematics)
    J checks = J::object();
    checks.set("grid_matches_formula_max_abs_diff", max_formula_diff).set("grid_matches_formula", max_formula_diff < 1e-4)
        .set("sgd_matches_manual", all_match);
    all_ok = all_ok && max_formula_diff < 1e-4;

    J doc = trace::envelope("why", "why-gradients", "tools/trace/trace_why.cpp");
    J scenes = J::array();
    scenes.push(scene.json);
    doc.set("scenes", std::move(scenes)).set("setup", std::move(setup)).set("grid", std::move(grid)).set("probes", std::move(probes))
        .set("secants", std::move(secants)).set("nudge", std::move(nudge)).set("trajectories", std::move(trajectories))
        .set("checks", std::move(checks));
    std::fputs(doc.dump().c_str(), stdout);

    if (!all_ok) {
        std::cerr << "trace_why: a self-check failed (SGD != manual update, or grid != 8(w-3))\n";
        return 1;
    }
    return 0;
}
