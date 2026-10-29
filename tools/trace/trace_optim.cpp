// trace_optim.cpp  ->  site/data/optim.json   (lesson `sgd-step`, docs/content-plan.md 3.2)
//
// The line y = 2x + 1 fitted to three points, x = [1,2,3], t = [3,5,7], by nn::Linear(1,1) (w = b = 0 at the start),
// nn::MSELoss and optim::SGD. Everything below is produced by running the real library; quantities that are plain
// mathematics computed here in double (the Hessian, its eigenvalues, convergence factors) sit under keys named
// "hessian", "analysis" or "derived" so a lesson can tell them apart from library output.
//
// v3 additions (brief §6 `sgd-step`, §7 c-zero and D13; appended after the existing keys, nothing renamed or reordered):
//   - `data.points[]`: the three data points listed one by one, {x, t}.
//   - `start`: the state at w = b = 0 from one forward + backward pass of nn::Linear + nn::MSELoss: {w, b, loss, gw, gb}. `gw`
//     is the one-pass gradient the c-zero check needs.
//   - `nudge_start`: the gradient found by nudging one parameter at a time from the start. For `w` alone (b stays 0) and for `b`
//     alone (w stays 0), and h = 1, 0.5, 0.1, 0.01, 0.001: {h, L_h, dL, slope, expected_slope, backward_slope}, with L from the
//     real forward pass in float32 and slope = (L_h - L0)/h; `expected_slope` is derived (-68/3 + 14h/3 for w, -10 + h for b).
//     `gradient` = [gw, gb] from backward(), the list of the two slopes; `secant_gradient` = the two slopes at h = 0.001.
//   - `after_ten_steps`: one quantity for "the loss after ten steps" (D13): lr 0.05, the state after the 10th update, with
//     zero_grad() (equal to runs[0].final) and without it (the no_zero_grad training loop), plus `no_zero_grad.training_loop.final`.
#include "trace_util.hpp"

#include "torc/nn.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

#include <complex>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

const std::vector<float> XS = {1.0f, 2.0f, 3.0f};
const std::vector<float> TS = {3.0f, 5.0f, 7.0f};

Tensor col(const std::vector<float>& v) { return trace::make_tensor({static_cast<int>(v.size()), 1}, v); }

struct State {
    float w, b, loss, gw, gb;
};

J state_json(const State& s) {
    J o = J::object();
    o.set("w", s.w).set("b", s.b).set("loss", s.loss).set("gw", s.gw).set("gb", s.gb);
    return o;
}

// nn::Linear(1,1) with chosen starting parameters + the data above + the real loss module.
struct Setup {
    nn::Linear lin;
    Variable* W;
    Variable* B;
    Variable x, t;
    nn::MSELoss mse;

    Setup(float w0 = 0.0f, float b0 = 0.0f) : lin(1, 1, 0.0f, 1u), x(col(XS), false), t(col(TS), false) {
        W = &lin.named_parameters().at("weight");
        B = &lin.named_parameters().at("bias");
        W->data().data()[0] = w0;
        B->data().data()[0] = b0;
    }
    State read(float loss) const {
        return {W->data().data()[0], B->data().data()[0], loss, W->has_grad() ? W->grad().data()[0] : 0.0f,
                B->has_grad() ? B->grad().data()[0] : 0.0f};
    }
    // forward + backward (gradients ACCUMULATE into W and B, as in the library)
    State fwd_bwd() {
        Variable pred = lin(x);
        Variable loss = mse(pred, t);
        loss.backward();
        return read(loss.data().data()[0]);
    }
    float loss_only() {
        Variable::NoGradGuard ng;
        Variable pred = lin(x);
        Variable loss = mse(pred, t);
        return loss.data().data()[0];
    }
};

// ---- derived mathematics (double) ---------------------------------------------------------------------------------
struct Hessian {
    double a, c, d;                 // [[a, c], [c, d]]
    double l_max, l_min;
    double v_max[2], v_min[2];      // unit eigenvectors
};

Hessian make_hessian() {
    double n = static_cast<double>(XS.size()), sx = 0, sxx = 0;
    for (float x : XS) { sx += x; sxx += static_cast<double>(x) * x; }
    Hessian h{};
    h.a = 2.0 / n * sxx;  // d2L/dw2
    h.c = 2.0 / n * sx;   // d2L/dwdb
    h.d = 2.0 / n * n;    // d2L/db2
    double tr = h.a + h.d, det = h.a * h.d - h.c * h.c, disc = std::sqrt(tr * tr / 4 - det);
    h.l_max = tr / 2 + disc;
    h.l_min = tr / 2 - disc;
    auto vec = [&](double l, double* v) {
        double x = l - h.d, y = h.c, nrm = std::hypot(x, y);
        v[0] = x / nrm; v[1] = y / nrm;
        if (v[0] < 0) { v[0] = -v[0]; v[1] = -v[1]; }
    };
    vec(h.l_max, h.v_max);
    vec(h.l_min, h.v_min);
    return h;
}

// Spectral radius of the optimiser's linear iteration on the quadratic loss, per eigen-direction lambda:
//   plain SGD:        e' = (1 - lr*lambda) e
//   with momentum:    roots of r^2 - (1 + mu - lr*lambda) r + mu = 0     (v = mu v + g; p -= lr v)
double spectral_radius(double lr, double mu, const Hessian& h) {
    double rho = 0;
    for (double l : {h.l_max, h.l_min}) {
        if (mu == 0.0) { rho = std::fmax(rho, std::fabs(1.0 - lr * l)); continue; }
        double tr = 1.0 + mu - lr * l;
        std::complex<double> disc = std::sqrt(std::complex<double>(tr * tr - 4.0 * mu, 0.0));
        rho = std::fmax(rho, std::fmax(std::abs((tr + disc) / 2.0), std::abs((tr - disc) / 2.0)));
    }
    return rho;
}

J run_sgd(const std::string& id, double lr, double mu, int n_steps, const Hessian& h) {
    Setup S;
    optim::SGD opt(S.lin.parameters(), static_cast<float>(lr), static_cast<float>(mu));
    J steps = J::array();
    float first_loss = 0.0f;
    for (int i = 0; i < n_steps; ++i) {
        State st = S.fwd_bwd();
        if (i == 0) first_loss = st.loss;
        steps.push(state_json(st));  // state BEFORE update i+1: params, loss, and the gradients step() will read
        opt.step();
        opt.zero_grad();
    }
    State fin = S.fwd_bwd();  // state after the last update
    S.W->zero_grad();
    S.B->zero_grad();

    const double rho = spectral_radius(lr, mu, h);
    J analysis = J::object();
    analysis.set("critical_lr", 2.0 * (1.0 + mu) / h.l_max).set("spectral_radius", rho).set("stable_predicted", rho < 1.0);
    J o = J::object();
    o.set("id", id).set("optimizer", "SGD").set("lr", lr).set("momentum", mu).set("steps", std::move(steps))
        .set("final", state_json(fin)).set("observed_diverged", fin.loss > first_loss).set("analysis", std::move(analysis));
    return o;
}

J pair_json(const char* n1, float v1, const char* n2, float v2) {
    J o = J::object();
    o.set(n1, v1).set(n2, v2);
    return o;
}

}  // namespace

int main() {
    bool ok = true;
    const Hessian H = make_hessian();

    // ---- scene `sgd`: the Linear computation spelled out with real ops ------------------------------------------
    trace::Scene s;
    s.id = "sgd";
    s.title = "One SGD step on a line fit";
    s.expression = "L = mse( x @ W^T + b, t )";
    s.note = "Three points on y = 2x + 1. backward() fills W.grad and b.grad; SGD::step() reads them and rewrites W and b.";
    s.leaves = {{"x", col(XS)}, {"W", trace::make_tensor({1, 1}, {0.0f})}, {"b", trace::make_tensor({1}, {0.0f})}, {"t", col(TS)}};
    s.constants = {"x", "t"};
    s.build = [](trace::Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
        Variable& wt = g.op("Wt", "transpose", transpose(*L["W"]));
        Variable& xw = g.op("xW", "matmul", matmul(*L["x"], wt));
        Variable& y = g.op("y", "add", add(xw, *L["b"]));
        return g.op("L", "mse", nn::MSELoss{}(y, *L["t"]));
    };
    trace::SceneResult scene = trace::run_scene(s);

    // parity: the real module computes exactly what the hand-built graph computes
    {
        Setup S;
        State st = S.fwd_bwd();
        Variable pred = S.lin(S.x);  // second forward only to read the output tensor
        const bool out_eq = trace::tensors_identical(pred.data(), scene.value.at("y"));
        const bool gw_eq = trace::tensors_identical(S.W->grad(), scene.grad.at("W"));
        const bool gb_eq = trace::tensors_identical(S.B->grad(), scene.grad.at("b"));
        const bool loss_eq = st.loss == scene.value.at("L").data()[0];
        float md = std::max({trace::max_abs_diff(pred.data(), scene.value.at("y")), trace::max_abs_diff(S.W->grad(), scene.grad.at("W")),
                             trace::max_abs_diff(S.B->grad(), scene.grad.at("b")), std::fabs(st.loss - scene.value.at("L").data()[0])});
        J par = J::object();
        par.set("against", "nn::Linear(1,1) + nn::MSELoss").set("output_equal", out_eq).set("grads_equal", gw_eq && gb_eq)
            .set("loss_equal", loss_eq).set("max_abs_diff", md);
        scene.json.set("parity", std::move(par));
        ok = ok && out_eq && gw_eq && gb_eq && loss_eq;
    }

    // ---- runs ------------------------------------------------------------------------------------------------------
    J runs = J::array();
    for (double lr : {0.05, 0.1, 0.15, 0.2}) runs.push(run_sgd("sgd-lr" + J::fmt_double(lr), lr, 0.0, 10, H));
    for (double mu : {0.5, 0.9}) runs.push(run_sgd("sgd-lr0.05-mu" + J::fmt_double(mu), 0.05, mu, 10, H));

    // ---- no zero_grad: gradients accumulate ---------------------------------------------------------------------
    J no_zero = J::object();
    {
        // (a) three forward+backward passes at the SAME parameters, no step, no zero_grad
        Setup S;
        J its = J::array();
        float prev_gw = 0, prev_gb = 0;
        for (int i = 1; i <= 3; ++i) {
            State st = S.fwd_bwd();
            J o = state_json(st);
            o.set("gw_this_pass", st.gw - prev_gw).set("gb_this_pass", st.gb - prev_gb).set("iteration", i);
            prev_gw = st.gw; prev_gb = st.gb;
            its.push(std::move(o));
        }
        no_zero.set("accumulate_only", std::move(its));
        // (b) a training loop that calls step() every iteration but never zero_grad()
        Setup S2;
        optim::SGD opt(S2.lin.parameters(), 0.05f);
        J loop = J::array();
        prev_gw = 0; prev_gb = 0;
        for (int i = 1; i <= 10; ++i) {
            State st = S2.fwd_bwd();
            J o = state_json(st);
            o.set("gw_this_pass", st.gw - prev_gw).set("gb_this_pass", st.gb - prev_gb).set("iteration", i);
            prev_gw = st.gw; prev_gb = st.gb;
            loop.push(std::move(o));
            opt.step();  // no zero_grad: the next backward() adds to what is already there
        }
        J b = J::object();
        b.set("lr", 0.05).set("momentum", 0).set("iterations", std::move(loop));
        J fin = J::object();  // v3: after the 10th step (no forward/backward after it, so no gradient fields)
        fin.set("w", S2.W->data().data()[0]).set("b", S2.B->data().data()[0]).set("loss", S2.loss_only());
        b.set("final", std::move(fin));
        no_zero.set("training_loop", std::move(b));
    }

    // ---- skipped step: step() with no gradient changes nothing and raises nothing --------------------------------
    J skipped = J::array();
    {
        auto case_json = [&](const char* name, const char* what, float w0, float b0, float w1, float b1, const trace::Caught& c) {
            J o = J::object();
            o.set("name", name).set("what", what).set("before", pair_json("w", w0, "b", b0)).set("after", pair_json("w", w1, "b", b1))
                .set("changed", w0 != w1 || b0 != b1).set("threw", c.type != "none").set("exception", c.type);
            skipped.push(std::move(o));
        };
        {
            Setup S;
            optim::SGD opt(S.lin.parameters(), 0.05f);
            float w0 = S.W->data().data()[0], b0 = S.B->data().data()[0];
            auto c = trace::catch_call([&] { opt.step(); });
            case_json("step_without_backward", "step() on parameters that never had backward() called", w0, b0, S.W->data().data()[0], S.B->data().data()[0], c);
        }
        {
            Setup S;
            optim::SGD opt(S.lin.parameters(), 0.05f);
            (void)S.fwd_bwd();
            opt.zero_grad();
            float w0 = S.W->data().data()[0], b0 = S.B->data().data()[0];
            auto c = trace::catch_call([&] { opt.step(); });
            case_json("step_after_zero_grad", "backward(), then zero_grad(), then step()", w0, b0, S.W->data().data()[0], S.B->data().data()[0], c);
        }
        {
            Setup S;
            optim::SGD opt(S.lin.parameters(), 0.05f, 0.9f);
            float w0 = S.W->data().data()[0], b0 = S.B->data().data()[0];
            auto c = trace::catch_call([&] { opt.step(); opt.step(); });
            case_json("momentum_step_without_backward", "two step() calls with momentum 0.9 and no gradient", w0, b0, S.W->data().data()[0], S.B->data().data()[0], c);
        }
        {
            Setup S;
            optim::SGD opt(S.lin.parameters(), 0.05f);
            (void)S.fwd_bwd();
            float w0 = S.W->data().data()[0], b0 = S.B->data().data()[0];
            auto c = trace::catch_call([&] { opt.step(); });
            case_json("step_after_backward", "control: backward(), then step()", w0, b0, S.W->data().data()[0], S.B->data().data()[0], c);
        }
    }

    // ---- constructor errors (exact what() text) -------------------------------------------------------------------
    J ctor_errors = J::array();
    {
        Setup S;
        auto add_err = [&](const char* call, const trace::Caught& c) {
            J o = J::object();
            o.set("call", call).set("exception", c.type).set("what", c.what);
            ctor_errors.push(std::move(o));
        };
        add_err("SGD(params, lr = 0)", trace::catch_call([&] { optim::SGD o(S.lin.parameters(), 0.0f); }));
        add_err("SGD(params, lr = -0.1)", trace::catch_call([&] { optim::SGD o(S.lin.parameters(), -0.1f); }));
        add_err("SGD(params, lr = 0.1, momentum = -0.5)", trace::catch_call([&] { optim::SGD o(S.lin.parameters(), 0.1f, -0.5f); }));
    }

    // ---- surface: loss on a (w, b) grid from the real forward pass --------------------------------------------------
    J surface = J::object();
    {
        std::vector<float> ws = {0, 1, 2, 3, 4}, bs = {-1, 0, 1, 2, 3};
        J pts = J::array();
        for (float w : ws)
            for (float b : bs) {
                Setup S(w, b);
                J o = J::object();
                o.set("w", w).set("b", b).set("loss", S.loss_only());
                pts.push(std::move(o));
            }
        surface.set("w", trace::jarr(ws)).set("b", trace::jarr(bs)).set("order", "w-major: for w in w: for b in b").set("points", std::move(pts));
    }

    // ---- v3: the start point, and the gradient found by nudging one parameter at a time -----------------------------
    J start = J::object();
    bool start_ok = true;
    J nudge_start = J::object();
    {
        Setup S;
        const State st = S.fwd_bwd();
        start.set("w", st.w).set("b", st.b).set("loss", st.loss).set("gw", st.gw).set("gb", st.gb);
        const float L0 = Setup().loss_only();
        auto nudges = [&](bool on_w) {
            J arr = J::array();
            for (double hd : {1.0, 0.5, 0.1, 0.01, 0.001}) {
                const float h = static_cast<float>(hd);
                const float Lh = on_w ? Setup(h, 0.0f).loss_only() : Setup(0.0f, h).loss_only();
                const float dL = Lh - L0;  // float32 difference, as in why-gradients.json secants
                J o = J::object();
                o.set("h", hd).set("L_h", Lh).set("dL", dL).set("slope", dL / h)
                    .set("expected_slope", on_w ? -68.0 / 3.0 + 14.0 / 3.0 * hd : -10.0 + hd)  // derived: (L(h) - L(0))/h of the exact loss
                    .set("backward_slope", on_w ? st.gw : st.gb);
                arr.push(std::move(o));
            }
            return arr;
        };
        J nw = nudges(true), nb = nudges(false);
        const float sw = Setup(0.001f, 0.0f).loss_only(), sb = Setup(0.0f, 0.001f).loss_only();
        nudge_start.set("w0", 0).set("b0", 0).set("L0", L0).set("w", std::move(nw)).set("b", std::move(nb))
            .set("gradient", trace::jarr(std::vector<float>{st.gw, st.gb}))
            .set("secant_gradient", trace::jarr(std::vector<float>{(sw - L0) / 0.001f, (sb - L0) / 0.001f}));
        start_ok = L0 == st.loss;  // the no-gradient forward pass gives the loss of the traced pass
    }

    // ---- v3: one "loss after ten steps" (D13) -----------------------------------------------------------------------
    J after_ten = J::object();
    {
        auto ten = [](bool zero) {
            Setup S;
            optim::SGD opt(S.lin.parameters(), 0.05f);
            for (int i = 0; i < 10; ++i) {
                (void)S.fwd_bwd();
                opt.step();
                if (zero) opt.zero_grad();
            }
            J o = J::object();
            o.set("w", S.W->data().data()[0]).set("b", S.B->data().data()[0]).set("loss", S.loss_only());
            return o;
        };
        after_ten.set("lr", 0.05).set("updates", 10).set("with_zero_grad", ten(true)).set("without_zero_grad", ten(false))
            .set("note", "Parameters and loss after the 10th step() call. with_zero_grad equals runs[0].final; without_zero_grad is the no_zero_grad training loop after its 10th step.");
    }

    // ---- assemble ---------------------------------------------------------------------------------------------------
    J data = J::object();
    data.set("x", trace::jarr(XS)).set("t", trace::jarr(TS)).set("true_w", 2).set("true_b", 1).set("n", 3)
        .set("optimum", [] { J o = J::object(); o.set("w", 2).set("b", 1).set("loss", 0); return o; }());  // derived: t = 2x + 1 exactly
    {
        J pts = J::array();
        for (size_t i = 0; i < XS.size(); ++i) { J o = J::object(); o.set("x", XS[i]).set("t", TS[i]); pts.push(std::move(o)); }
        data.set("points", std::move(pts));
    }

    J hess = J::object();
    {
        J m = J::array();
        m.push(trace::jarr(std::vector<double>{H.a, H.c})).push(trace::jarr(std::vector<double>{H.c, H.d}));
        hess.set("derived", true)
            .set("note", "Expected mathematics, computed in double from x: H = (2/N) [[sum x^2, sum x],[sum x, N]]. Not library output.")
            .set("matrix", std::move(m))
            .set("eigenvalues", trace::jarr(std::vector<double>{H.l_max, H.l_min}))
            .set("eigenvectors", [&] {
                J v = J::array();
                v.push(trace::jarr(std::vector<double>{H.v_max[0], H.v_max[1]})).push(trace::jarr(std::vector<double>{H.v_min[0], H.v_min[1]}));
                return v;
            }())
            .set("critical_lr", 2.0 / H.l_max);
    }

    J checks = J::object();
    checks.set("parity_output_and_grads_equal", ok);

    J doc = trace::envelope("optim", "sgd-step", "tools/trace/trace_optim.cpp");
    J scenes = J::array();
    scenes.push(scene.json);
    doc.set("scenes", std::move(scenes)).set("data", std::move(data)).set("hessian", std::move(hess)).set("runs", std::move(runs))
        .set("no_zero_grad", std::move(no_zero)).set("skipped_step", std::move(skipped)).set("ctor_errors", std::move(ctor_errors))
        .set("surface", std::move(surface)).set("checks", std::move(checks)).set("start", std::move(start))
        .set("nudge_start", std::move(nudge_start)).set("after_ten_steps", std::move(after_ten));
    std::fputs(doc.dump().c_str(), stdout);

    if (!ok) {
        std::cerr << "trace_optim: parity check failed (nn::Linear+MSELoss differs from the hand-built graph)\n";
        return 1;
    }
    if (!start_ok) {
        std::cerr << "trace_optim: the start loss differs between the traced and the no-gradient forward pass\n";
        return 1;
    }
    return 0;
}
