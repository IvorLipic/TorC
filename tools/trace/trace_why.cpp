// trace_why.cpp  ->  site/data/why-gradients.json   (lesson `why-gradients`, docs/content-plan.md 3.1)
//
// One weight, one datum:  p = w*x,  e = p - t,  L = e*e   with x = 2, t = 6  (so L(w) = (2w-6)^2 = 4(w-3)^2).
// Everything is computed by the real library: the forward values with torc::mul/sub, the slopes with Variable::backward,
// the weight updates with torc::optim::SGD on a one-element Variable.
//
// Quantities that are NOT library output are marked "derived" in the JSON (plain double arithmetic on recorded numbers,
// e.g. the factor 1 - 8*lr or the tangent-line prediction); they are there so lessons do not have to type them by hand.
//
// v3 additions (brief C3 and Appendix B; all appended after the existing keys, nothing renamed or reordered):
//   - every trajectory state also records `step` (w_next - w), `distance` |w - 3|, `distance_next` |w_next - 3|, `kept`
//     (distance_next / distance, the share of the distance one step keeps; null when the distance is 0) and `crossed` (the step
//     lands on the other side of w = 3). Each trajectory records `kept` and `kept_percent` of its first step. All of these are
//     double arithmetic on the recorded float32 values (derived), so a lesson never has to compute them.
//   - `nudge` also records `miss_before`, `miss_after` (|e| before and after), `fall` (-actual_dL) and `ratio` (actual_dL / delta).
//   - `steep`: the same experiment for four notebooks that cost 12 euros, L = (4w - 12)^2 = 16(w-3)^2 (same minimum w = 3,
//     gradient 32(w-3), so -96 at w = 0). It has `setup`, `grid`, `probes`, `secants` and `trajectories` in the same format, built
//     from the same real torc::mul/sub, Variable::backward and optim::SGD::step calls, plus `compare` (steep vs. the bowl above at
//     the same weights) and `checks`.
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

// Real forward + real backward at weight w (x, t default to the two-notebook datum).
Point eval_why(float w, float x = X, float t = T) {
    Variable wv(Tensor({w}, {1}), true);
    Variable xv(Tensor({x}, {1}), false);
    Variable tv(Tensor({t}, {1}), false);
    Variable p = mul(wv, xv);
    Variable e = sub(p, tv);
    Variable L = mul(e, e);
    L.backward();
    return {w, p.data().data()[0], e.data().data()[0], L.data().data()[0], wv.grad().data()[0]};
}

// Real forward only (no tape).
Point eval_forward(float w, float x = X, float t = T) {
    Variable::NoGradGuard ng;
    Variable wv(Tensor({w}, {1}), true);
    Variable xv(Tensor({x}, {1}), false);
    Variable tv(Tensor({t}, {1}), false);
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


constexpr double W_MIN = 3.0;  // the minimum of both bowls (t / x = 6/2 = 12/4)

// Derived from two recorded weights (double arithmetic on float32 values): how far the step went, the distance to the minimum
// before and after, the share of the distance the step keeps, and whether it lands on the other side of the minimum.
void add_distance_fields(J& st, float wk, float w_next) {
    const double a = static_cast<double>(wk), b = static_cast<double>(w_next);
    const double d0 = std::fabs(a - W_MIN), d1 = std::fabs(b - W_MIN);
    st.set("step", b - a).set("distance", d0).set("distance_next", d1);
    st.set("kept", d0 > 0.0 ? J(d1 / d0) : J());
    st.set("crossed", (a - W_MIN) * (b - W_MIN) < 0.0);
}

struct Traj {
    J json;
    bool match;
};

// Real optim::SGD on a one-element Variable from w = 0, nine states (k = 0..8), for the datum (x, t).
// `factor_scale` is the curvature 2x^2 (8 or 32): the distance to w = 3 is multiplied by 1 - factor_scale*lr per step (derived).
Traj trajectory(double lrd, float x, float t, double factor_scale, bool extra_fields) {
    const float lr = static_cast<float>(lrd);
    Variable w(Tensor({0.0f}, {1}), true);
    optim::SGD opt({&w}, lr);
    J states = J::array();
    bool match = true;
    float max_diff = 0.0f;
    double first_kept = -1.0;
    for (int k = 0; k <= 8; ++k) {
        Variable xv(Tensor({x}, {1}), false);
        Variable tv(Tensor({t}, {1}), false);
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
        if (extra_fields) add_distance_fields(st, wk, manual_next);
        if (k == 0) first_kept = std::fabs(static_cast<double>(manual_next) - W_MIN) / std::fabs(static_cast<double>(wk) - W_MIN);
        states.push(std::move(st));
        opt.step();  // real SGD::step: reads w.grad(), rewrites w.data()
        opt.zero_grad();
        const float after = w.data().data()[0];
        max_diff = std::fmax(max_diff, std::fabs(after - manual_next));
        if (after != manual_next) match = false;
    }
    const double factor = std::round((1.0 - factor_scale * lrd) * 1e12) / 1e12;  // derived: distance to w = 3 is multiplied by this per step
    J tr = J::object();
    tr.set("lr", lrd).set("momentum", 0).set("factor", factor).set("stable_predicted", std::fabs(factor) < 1.0)
        .set("matches_manual", match).set("max_abs_diff_vs_manual", max_diff).set("states", std::move(states));
    if (extra_fields) tr.set("kept", first_kept).set("kept_percent", 100.0 * first_kept);  // derived: share of the distance after step 1
    return {std::move(tr), match};
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
        // derived (v3): the size of the miss before and after the nudge, how much the loss fell, and the loss change per unit of w
        nudge.set("miss_before", std::fabs(a.e)).set("miss_after", std::fabs(b.e)).set("fall", -(b.L - a.L)).set("ratio", (b.L - a.L) / delta);
    }

    // ---- trajectories: real optim::SGD on a one-element Variable -------------------------------------------------
    J trajectories = J::array();
    bool all_match = true;
    for (double lrd : {0.05, 0.1, 0.125, 0.25, 0.3}) {
        Traj tr = trajectory(lrd, X, T, 8.0, true);
        all_match = all_match && tr.match;
        trajectories.push(std::move(tr.json));
    }
    all_ok = all_ok && all_match;

    // ---- steep: four notebooks cost 12 euros, L = (4w - 12)^2 = 16(w-3)^2 (brief C3) --------------------------------
    // Same minimum and the same start, so the same distance; the gradient at every weight is four times larger.
    constexpr float SX = 4.0f, ST = 12.0f;
    J steep = J::object();
    bool steep_ok = true;
    {
        J ssetup = J::object();
        ssetup.set("x", SX).set("t", ST).set("w_min", 3).set("loss_formula", "L(w) = (x*w - t)^2 = 16(w-3)^2")
            .set("grad_formula", "dL/dw = 2*x*(x*w - t) = 32(w-3)").set("critical_lr", 0.0625);  // derived (Expected mathematics)
        J sgrid = J::array();
        double smax = 0.0;
        for (int i = 0; i <= 32; ++i) {
            float w = -1.0f + 0.25f * static_cast<float>(i);
            Point q = eval_why(w, SX, ST);
            sgrid.push(point_json(q));
            const double d = static_cast<double>(w) - 3.0;
            smax = std::fmax(smax, std::fabs(static_cast<double>(q.grad) - 32.0 * d));
            smax = std::fmax(smax, std::fabs(static_cast<double>(q.L) - 16.0 * d * d));
        }
        J sprobes = J::array();
        for (float w : {0.0f, 0.1f, 1.0f, 1.2f, 3.0f, 6.0f, 7.2f}) sprobes.push(point_json(eval_why(w, SX, ST)));
        const Point s0 = eval_why(0.0f, SX, ST);
        J ssec = J::array();
        for (double hd : {1.0, 0.5, 0.1, 0.01, 0.001}) {
            float h = static_cast<float>(hd);
            Point ph = eval_forward(h, SX, ST);
            float dL = ph.L - s0.L;
            J o = J::object();
            o.set("h", hd).set("w0", s0.w).set("L0", s0.L).set("L_h", ph.L).set("dL", dL).set("slope", dL / h)
                .set("expected_slope", 16.0 * hd - 96.0)  // derived: (16h^2 - 96h)/h
                .set("backward_slope", s0.grad);
            ssec.push(std::move(o));
        }
        // lr 0.05 ... 0.3: the learning rates of the two-notebook bowl (index i here = index i there).
        // lr 0.0125 ... 0.075: a quarter of those, so each keeps the same share of the distance as trajectories[i - 5] above.
        J straj = J::array();
        bool smatch = true;
        for (double lrd : {0.05, 0.1, 0.125, 0.25, 0.3, 0.0125, 0.025, 0.03125, 0.0625, 0.075}) {
            Traj tr = trajectory(lrd, SX, ST, 32.0, true);
            smatch = smatch && tr.match;
            tr.json.set("same_factor_as_lr", 4.0 * lrd);  // derived: the two-notebook learning rate with the same factor
            straj.push(std::move(tr.json));
        }
        // compare: both bowls at the same weights (the probes), from the real backward passes above
        J cmp = J::array();
        for (float w : {0.0f, 1.0f, 1.2f, 6.0f}) {
            const Point a = eval_why(w), b = eval_why(w, SX, ST);
            J o = J::object();
            o.set("w", w).set("distance", std::fabs(static_cast<double>(w) - W_MIN)).set("grad", a.grad).set("steep_grad", b.grad)
                .set("L", a.L).set("steep_L", b.L).set("grad_ratio", static_cast<double>(b.grad) / static_cast<double>(a.grad));  // derived
            cmp.push(std::move(o));
        }
        J schecks = J::object();
        schecks.set("grid_matches_formula_max_abs_diff", smax).set("grid_matches_formula", smax < 1e-3).set("sgd_matches_manual", smatch);
        steep_ok = smax < 1e-3 && smatch;
        steep.set("setup", std::move(ssetup)).set("grid", std::move(sgrid)).set("probes", std::move(sprobes)).set("secants", std::move(ssec))
            .set("trajectories", std::move(straj)).set("compare", std::move(cmp)).set("checks", std::move(schecks));
    }
    all_ok = all_ok && steep_ok;

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
        .set("checks", std::move(checks)).set("steep", std::move(steep));
    std::fputs(doc.dump().c_str(), stdout);

    if (!all_ok) {
        std::cerr << "trace_why: a self-check failed (SGD != manual update, or grid != 8(w-3) or 32(w-3))\n";
        return 1;
    }
    return 0;
}
