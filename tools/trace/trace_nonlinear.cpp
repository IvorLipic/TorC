// trace_nonlinear.cpp  ->  site/data/nonlinearity.json   (lesson `nonlinearity`)
//
// 1. relu and sigmoid on a grid of 81 inputs, values and derivatives from the real backward pass.
// 2. softmax along axis 1 for a few rows (sums to one, a constant shift changes nothing, large inputs do not overflow).
// 3. Two small models fitted to t = x^2 with the real nn::Linear, relu, nn::MSELoss and optim::Adam:
//      linear:  Linear(1, 8) -> Linear(8, 1)          (no activation)
//      relu:    Linear(1, 8) -> relu -> Linear(8, 1)
//    Both start from the same weights, set by hand (no random generator), so the run is the same on every machine. The start
//    is not a flat line, so the linear model visibly turns flat while it trains.
//    Predictions on a grid are recorded at a few steps. For the linear model the two layers are also multiplied out to one
//    line (w_eff, b_eff), and the trace checks that the model's predictions lie on that line.
// Derived values (plain arithmetic, not library output) are under the key "derived". The input grids are built from integers
// (k / 10, not -4 + 0.1 k) so that no float rounding or contraction can move a grid point off its exact value.
#include "trace_util.hpp"

#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

#include <cmath>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

constexpr int H = 8;                 // hidden width
constexpr int STEPS = 400;
constexpr float LR = 0.05f;
const std::vector<int> CHECKPOINTS = {0, 25, 100, 400};

// the same start for both models: hidden unit k has weight +1 or -1, a bias spread over [-1.75, 1.75], and an output weight
// 0.2 or 0.1 (so the multiplied-out line of the linear model starts with a non-zero slope)
void init(nn::Linear& l1, nn::Linear& l2) {
    Tensor& w1 = l1.named_parameters().at("weight").data();
    Tensor& b1 = l1.named_parameters().at("bias").data();
    Tensor& w2 = l2.named_parameters().at("weight").data();
    Tensor& b2 = l2.named_parameters().at("bias").data();
    for (int k = 0; k < H; ++k) {
        w1.data()[k] = (k % 2 == 0) ? 1.0f : -1.0f;
        b1.data()[k] = -1.75f + 0.5f * static_cast<float>(k);
        w2.data()[k] = (k % 2 == 0) ? 0.2f : 0.1f;
    }
    b2.data()[0] = 0.0f;
}

J activation(const char* name, const Tensor& z, bool use_relu) {
    Variable zv(z, true);
    Variable f = use_relu ? relu(zv) : sigmoid(zv);
    Variable s = sum(f);
    s.backward();                                                       // d sum(f) / dz_i = f'(z_i)
    J o = J::object();
    o.set("name", name).set("value", trace::jarr(trace::values_of(f.data()))).set("grad", trace::jarr(trace::values_of(zv.grad())));
    return o;
}

J fit(const char* name, bool use_relu, const Tensor& x, const Tensor& t, const Tensor& grid, bool& ok) {
    nn::Linear l1(1, H, 0.1f, 1), l2(H, 1, 0.1f, 1);                   // the drawn weights are overwritten by init()
    init(l1, l2);
    std::vector<Variable*> params = l1.parameters();
    for (Variable* p : l2.parameters()) params.push_back(p);
    optim::Adam opt(params, LR);
    nn::MSELoss mse;
    const Variable xv(x, false), tv(t, false), gv(grid, false);

    auto predict = [&](const Variable& in) {
        Variable::NoGradGuard ng;
        Variable h = l1(in);
        Variable r = use_relu ? relu(h) : h;
        Variable y = l2(r);
        return trace::values_of(y.data());
    };
    auto loss_now = [&] {
        Variable::NoGradGuard ng;
        Variable h = l1(xv);
        Variable r = use_relu ? relu(h) : h;
        Variable y = l2(r);
        Variable L = mse(y, tv);
        return L.data().data()[0];
    };

    J checkpoints = J::array();
    auto record = [&](int step) {
        J c = J::object();
        c.set("step", step).set("loss", loss_now()).set("prediction", trace::jarr(predict(gv)));
        checkpoints.push(std::move(c));
    };
    record(0);
    for (int step = 1; step <= STEPS; ++step) {
        Variable h = l1(xv);
        Variable r = use_relu ? relu(h) : h;
        Variable y = l2(r);
        Variable L = mse(y, tv);
        L.backward();
        opt.step();
        opt.zero_grad();
        for (int c : CHECKPOINTS) if (c == step) record(step);
    }
    J o = J::object();
    o.set("name", name).set("checkpoints", std::move(checkpoints));
    if (!use_relu) {
        // multiply the two layers out: y = w2 . (w1 x + b1) + b2 = w_eff x + b_eff  (derived, then checked against the model)
        const Tensor& w1 = l1.named_parameters().at("weight").data();
        const Tensor& b1 = l1.named_parameters().at("bias").data();
        const Tensor& w2 = l2.named_parameters().at("weight").data();
        const Tensor& b2 = l2.named_parameters().at("bias").data();
        double w_eff = 0.0, b_eff = b2.data()[0];
        for (int k = 0; k < H; ++k) { w_eff += double(w2.data()[k]) * w1.data()[k]; b_eff += double(w2.data()[k]) * b1.data()[k]; }
        const std::vector<float> pred = predict(gv);
        double max_diff = 0.0;
        for (int i = 0; i < grid.numel(); ++i) max_diff = std::fmax(max_diff, std::fabs(pred[i] - (w_eff * grid.data()[i] + b_eff)));
        ok = ok && max_diff < 1e-4;
        J d = J::object();
        d.set("w_eff", w_eff).set("b_eff", b_eff).set("max_abs_diff_line_vs_model", max_diff);
        o.set("derived", std::move(d));
    }
    return o;
}

}  // namespace

int main() {
    bool ok = true;

    // ---- 1. activation curves --------------------------------------------------------------------------------------
    std::vector<float> zs;
    for (int k = 0; k <= 80; ++k) zs.push_back(static_cast<float>(k - 40) / 10.0f);
    const Tensor z = trace::make_tensor({81}, zs);
    J acts = J::array();
    acts.push(activation("relu", z, true)).push(activation("sigmoid", z, false));

    // ---- 2. softmax along axis 1 -------------------------------------------------------------------------------------
    struct Row { const char* id; std::vector<float> v; };
    const std::vector<Row> rows = {{"small", {1, 2, 3}}, {"shifted", {11, 12, 13}}, {"large", {1001, 1002, 1003}}, {"equal", {5, 5, 5}}};
    J softmax_rows = J::array();
    for (const auto& r : rows) {
        const Tensor in = trace::make_tensor({1, 3}, r.v);
        const Tensor out = in.softmax(1);
        J e = J::object();
        e.set("id", r.id).set("input", trace::jarr(r.v)).set("output", trace::jarr(trace::values_of(out)));
        softmax_rows.push(std::move(e));
    }

    // ---- 3. fitting t = x^2 ------------------------------------------------------------------------------------------
    std::vector<float> xs, ts, gs;
    for (int k = 0; k <= 20; ++k) { const float v = static_cast<float>(k - 10) / 5.0f; xs.push_back(v); ts.push_back(v * v); }
    for (int k = 0; k <= 44; ++k) gs.push_back(static_cast<float>(k - 22) / 10.0f);
    const Tensor x = trace::make_tensor({21, 1}, xs), t = trace::make_tensor({21, 1}, ts), grid = trace::make_tensor({45, 1}, gs);
    J models = J::array();
    models.push(fit("linear", false, x, t, grid, ok)).push(fit("relu", true, x, t, grid, ok));
    double mean_t = 0.0;
    for (float v : ts) mean_t += v;
    mean_t /= ts.size();

    J data = J::object();
    data.set("x", trace::jarr(xs)).set("t", trace::jarr(ts)).set("grid", trace::jarr(gs));
    J setup = J::object();
    setup.set("points", static_cast<int>(xs.size())).set("hidden", H).set("steps", STEPS).set("lr", static_cast<double>(LR)).set("optimizer", "optim::Adam").set("loss", "nn::MSELoss");
    J derived = J::object();
    derived.set("best_line_slope", 0).set("best_line_intercept", mean_t);  // least squares line for t = x^2 on symmetric x

    J doc = trace::envelope("nonlinearity", "nonlinearity", "tools/trace/trace_nonlinear.cpp");
    doc.set("z", trace::jarr(zs)).set("activations", std::move(acts)).set("softmax", std::move(softmax_rows))
        .set("data", std::move(data)).set("setup", std::move(setup)).set("models", std::move(models)).set("derived", std::move(derived));
    std::fputs(doc.dump().c_str(), stdout);
    if (!ok) { std::cerr << "trace_nonlinear: the linear model's predictions are not on its multiplied-out line\n"; return 1; }
    return 0;
}
