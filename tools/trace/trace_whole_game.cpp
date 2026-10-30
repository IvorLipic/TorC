// trace_whole_game.cpp  ->  site/data/whole-game.json   (lesson `what-is-learning`, brief §6 Part 0, R4)
//
// The whole game on one small, honestly synthetic data set: hours of practice in a week (x) and the score in a test, out of 10
// (y), for 15 students. The data come from a known line y = 1.5x + 2 plus noise, drawn here (not by TorC) with std::mt19937
// (seed 2026) and a Box-Muller transform in double; hours and scores are rounded to one decimal, as a teacher would write them
// down. std::mt19937 is fully specified by the C++ standard, so the data do not depend on the standard library. The first 10
// students are the training points, the last 5 are the hidden points the model never trains on.
//
// 1. `data`: the points, the true line and how they were made.
// 2. `runs[]`: nn::Linear(1, 1) with w and b set by hand (runs[0] from w = b = 0, runs[1] from w = 3, b = 0), nn::MSELoss on
//    shapes (10, 1) vs (10, 1) and optim::SGD (full batch of the 10 training points). Every step records w, b, the training loss
//    (the loss backward() was called on) and the loss on the hidden points (no gradient). A run stops when one step changes
//    neither w nor b by more than 1e-5 (it has settled), or after 1000 steps.
// 3. `probes[]`: MSELoss of a few fixed lines on the training and on the hidden points, with every prediction and residual,
//    so the JS that scores a hand-placed line can be tested against TorC. `train_square` / `hidden_square` (each residual
//    squared in float32, derived) give the lesson's worked example its numbers.
// 4. `true_line`: MSELoss of the true line itself (the noise no line can remove).
// Everything is library output, except `data` (made here, see above) and `derived` (least squares in double).
#include "trace_util.hpp"

#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

#include <cmath>
#include <iostream>
#include <numbers>
#include <random>

using namespace torc;
using trace::J;

namespace {

constexpr int N_ALL = 15, N_TRAIN = 10;
constexpr double TRUE_W = 1.5, TRUE_B = 2.0, NOISE_STD = 0.6;
constexpr unsigned SEED = 2026;
constexpr double LR = 0.1;
constexpr int MAX_STEPS = 1000;
constexpr float SETTLE = 1e-5f;

double round1(double v) { return std::round(v * 10.0) / 10.0; }

struct Data {
    std::vector<float> x, y;  // all 15 points; [0, 10) training, [10, 15) hidden
};

Data make_data() {
    std::mt19937 rng(SEED);
    auto u = [&] { return (static_cast<double>(rng()) + 0.5) / 4294967296.0; };  // in (0, 1), from the raw 32-bit output
    Data d;
    for (int i = 0; i < N_ALL; ++i) {
        const double hours = round1(0.5 + 3.5 * u());                            // 0.5 .. 4.0 hours
        const double n = std::sqrt(-2.0 * std::log(u())) * std::cos(2.0 * std::numbers::pi * u());
        const double score = round1(TRUE_W * hours + TRUE_B + NOISE_STD * n);
        d.x.push_back(static_cast<float>(hours));
        d.y.push_back(static_cast<float>(score));
    }
    return d;
}

Tensor col(const std::vector<float>& v, int from, int to) {
    return trace::make_tensor({to - from, 1}, std::vector<float>(v.begin() + from, v.begin() + to));
}

void set_line(nn::Linear& m, float w, float b) {
    m.named_parameters().at("weight").data().data()[0] = w;
    m.named_parameters().at("bias").data().data()[0] = b;
}
float weight(nn::Linear& m) { return m.named_parameters().at("weight").data().data()[0]; }
float bias(nn::Linear& m) { return m.named_parameters().at("bias").data().data()[0]; }

// MSELoss of the model on a set of points, with no gradient; optionally the predictions.
float loss_on(nn::Linear& m, const Tensor& xs, const Tensor& ys, std::vector<float>* pred = nullptr) {
    Variable::NoGradGuard ng;
    const Variable x(xs, false), y(ys, false);
    nn::MSELoss mse;
    Variable out = m(x);
    Variable L = mse(out, y);
    if (pred) *pred = trace::values_of(out.data());
    return L.data().data()[0];
}

J step_row(int k, float w, float b, float train, float hidden) {
    J o = J::object();
    o.set("step", k).set("w", w).set("b", b).set("train_loss", train).set("hidden_loss", hidden);
    return o;
}

}  // namespace

int main() {
    bool ok = true;
    const Data d = make_data();
    const Tensor xtr = col(d.x, 0, N_TRAIN), ytr = col(d.y, 0, N_TRAIN);
    const Tensor xhd = col(d.x, N_TRAIN, N_ALL), yhd = col(d.y, N_TRAIN, N_ALL);

    // ---- 1. data -------------------------------------------------------------------------------------------------------------
    J pts = J::array();
    for (int i = 0; i < N_ALL; ++i) {
        J p = J::object();
        p.set("i", i).set("x", d.x[i]).set("y", d.y[i]).set("set", i < N_TRAIN ? "train" : "hidden");
        pts.push(std::move(p));
    }
    J data = J::object();
    data.set("story", "hours of practice in a week (x) and the score in a test, out of 10 (y)")
        .set("x_name", "hours of practice").set("y_name", "score out of 10")
        .set("true_line", [] { J o = J::object(); o.set("w", TRUE_W).set("b", TRUE_B); return o; }())
        .set("noise_std", NOISE_STD).set("seed", static_cast<int>(SEED))
        .set("generator", "std::mt19937(seed), hours = round1(0.5 + 3.5u), noise = Box-Muller N(0,1) in double, score = round1(1.5 hours + 2 + 0.6 noise); round1 rounds to one decimal")
        .set("n_train", N_TRAIN).set("n_hidden", N_ALL - N_TRAIN)
        .set("train", [&] { J o = J::object(); o.set("x", trace::jarr(trace::values_of(xtr))).set("y", trace::jarr(trace::values_of(ytr))); return o; }())
        .set("hidden", [&] { J o = J::object(); o.set("x", trace::jarr(trace::values_of(xhd))).set("y", trace::jarr(trace::values_of(yhd))); return o; }())
        .set("points", std::move(pts));

    // ---- 2. training runs ----------------------------------------------------------------------------------------------------
    J runs = J::array();
    struct Start { const char* id; float w, b; };
    for (const Start st : {Start{"from-zero", 0.0f, 0.0f}, Start{"from-w3", 3.0f, 0.0f}}) {
        nn::Linear model(1, 1, 0.0f, 1u);
        set_line(model, st.w, st.b);
        optim::SGD opt(model.parameters(), static_cast<float>(LR));
        nn::MSELoss mse;
        const Variable x(xtr, false), y(ytr, false);
        J steps = J::array();
        int k = 0;
        bool settled = false;
        float last_train = 0.0f, last_hidden = 0.0f;
        for (;; ++k) {
            const float w0 = weight(model), b0 = bias(model);
            const float hidden = loss_on(model, xhd, yhd);
            const float train_ng = loss_on(model, xtr, ytr);  // before the traced forward: a forward clears the module's cache
            Variable out = model(x);
            Variable L = mse(out, y);
            const float train = L.data().data()[0];
            ok = ok && train == train_ng;  // the traced loss equals the no-gradient one
            steps.push(step_row(k, w0, b0, train, hidden));
            last_train = train;
            last_hidden = hidden;
            if (settled || k == MAX_STEPS) break;
            L.backward();
            opt.step();
            opt.zero_grad();
            // settled: this step changed neither parameter by more than SETTLE; the state after it is recorded, then the run stops
            settled = std::fabs(weight(model) - w0) <= SETTLE && std::fabs(bias(model) - b0) <= SETTLE;
        }
        J fin = J::object();
        fin.set("steps", k).set("w", weight(model)).set("b", bias(model)).set("train_loss", last_train).set("hidden_loss", last_hidden);
        J r = J::object();
        r.set("id", st.id).set("start", [&] { J o = J::object(); o.set("w", st.w).set("b", st.b); return o; }())
            .set("lr", LR).set("optimizer", "optim::SGD").set("batch", "all 10 training points").set("settled", settled)
            .set("settle_rule", "stop after the first step that changes neither w nor b by more than 1e-5 (at most 1000 steps)")
            .set("steps", std::move(steps)).set("final", std::move(fin));
        runs.push(std::move(r));
    }

    // ---- 3. probe lines ------------------------------------------------------------------------------------------------------
    J probes = J::array();
    struct Line { const char* id; float w, b; };
    for (const Line ln : {Line{"zero", 0.0f, 0.0f}, Line{"flat-5", 0.0f, 5.0f}, Line{"true", 1.5f, 2.0f}, Line{"w1-b3", 1.0f, 3.0f},
                          Line{"w2-b1", 2.0f, 1.0f}, Line{"w3-b0", 3.0f, 0.0f}, Line{"w-0.5-b6", -0.5f, 6.0f}}) {
        nn::Linear model(1, 1, 0.0f, 1u);
        set_line(model, ln.w, ln.b);
        std::vector<float> ptr, phd;
        const float ltr = loss_on(model, xtr, ytr, &ptr), lhd = loss_on(model, xhd, yhd, &phd);
        std::vector<float> rtr, rhd;  // residual = prediction - target, in float32 (what MSELoss squares)
        for (int i = 0; i < N_TRAIN; ++i) rtr.push_back(ptr[i] - d.y[i]);
        for (int i = 0; i < N_ALL - N_TRAIN; ++i) rhd.push_back(phd[i] - d.y[N_TRAIN + i]);
        std::vector<float> str, shd;  // the square of each residual, in float32 (the terms whose mean MSELoss takes; derived)
        for (const float r : rtr) str.push_back(r * r);
        for (const float r : rhd) shd.push_back(r * r);
        J o = J::object();
        o.set("id", ln.id).set("w", ln.w).set("b", ln.b).set("train_loss", ltr).set("hidden_loss", lhd)
            .set("train_pred", trace::jarr(ptr)).set("train_residual", trace::jarr(rtr))
            .set("hidden_pred", trace::jarr(phd)).set("hidden_residual", trace::jarr(rhd))
            .set("train_square", trace::jarr(str)).set("hidden_square", trace::jarr(shd));
        probes.push(std::move(o));
    }

    // ---- 4. the true line ----------------------------------------------------------------------------------------------------
    J true_line = J::object();
    {
        nn::Linear model(1, 1, 0.0f, 1u);
        set_line(model, static_cast<float>(TRUE_W), static_cast<float>(TRUE_B));
        true_line.set("w", TRUE_W).set("b", TRUE_B).set("train_loss", loss_on(model, xtr, ytr)).set("hidden_loss", loss_on(model, xhd, yhd));
    }

    // ---- derived: the least-squares line on the training points (double; what gradient descent should approach) ---------------
    J derived = J::object();
    {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < N_TRAIN; ++i) {
            const double x = d.x[i], y = d.y[i];
            sx += x; sy += y; sxx += x * x; sxy += x * y;
        }
        const double n = N_TRAIN, w = (n * sxy - sx * sy) / (n * sxx - sx * sx), b = (sy - w * sx) / n;
        double sse = 0;
        for (int i = 0; i < N_TRAIN; ++i) { const double r = w * d.x[i] + b - d.y[i]; sse += r * r; }
        const double h11 = 2.0 / n * sxx, h12 = 2.0 / n * sx, h22 = 2.0, tr = h11 + h22, det = h11 * h22 - h12 * h12;
        const double lmax = tr / 2 + std::sqrt(tr * tr / 4 - det);
        J ls = J::object();
        ls.set("w", w).set("b", b).set("train_loss", sse / n);
        derived.set("least_squares", std::move(ls)).set("hessian_lambda_max", lmax).set("critical_lr", 2.0 / lmax);
        ok = ok && LR < 2.0 / lmax;
    }

    J doc = trace::envelope("whole-game", "what-is-learning", "tools/trace/trace_whole_game.cpp");
    doc.set("data", std::move(data)).set("runs", std::move(runs)).set("probes", std::move(probes)).set("true_line", std::move(true_line))
        .set("derived", std::move(derived));
    std::fputs(doc.dump().c_str(), stdout);
    if (!ok) { std::cerr << "trace_whole_game: a self-check failed (traced loss != no-gradient loss, or lr above 2/lambda_max)\n"; return 1; }
    return 0;
}
