// trace_deep.cpp  ->  site/data/deep.json   (lesson `deep-nets`)
//
// Three small experiments on why deep networks are hard to start, with the real nn::Sequential, nn::Linear, nn::ReLU,
// nn::Sigmoid, nn::MSELoss and optim::SGD:
// 1. Vanishing gradients. Two stacks of 8 hidden layers of width 32, Linear(32, 32) followed by sigmoid in one stack and by
//    ReLU in the other, then a head Linear(32, 1). Both stacks get the same start weights (TorC's default Kaiming init with the
//    same seeds). One forward pass and one backward() of nn::MSELoss on a fixed batch of 64 inputs. Recorded per layer: the
//    mean absolute gradient of the layer's weight, and the spread (std) of the signal after its activation.
// 2. Initialization. The ReLU stack of 1 with three starts: TorC's default (init_std 0, which is Kaiming normal with std
//    kaiming_normal_std(32) = sqrt(2 / 32)), a std that is too small (0.05) and one that is too large (0.5). Recorded per layer:
//    the std, the root mean square and the share of zeros of the signal after ReLU (layer 0 is the input), and the mean
//    absolute weight gradient after one backward().
// 3. Twins. Linear(2, 2) -> ReLU -> Linear(2, 1) on the four XOR points, nn::MSELoss, optim::SGD. Once with the two hidden
//    neurons started with identical weights (same row of W1, same bias, same output weight), once with different weights.
//    Every step records both neurons' parameters and the gradients that step() reads, and whether the two neurons are still
//    bit-for-bit equal. A longer run records the loss at a few steps, so the lesson can show that the twins never split.
//
// The inputs and targets are made here from std::mt19937 (whose output the C++ standard fixes) and plain arithmetic, so they
// are the same with every standard library. The start weights of 1 and 2 come from nn::Linear's own init, so they depend on
// libstdc++'s std::normal_distribution (on one toolchain the output is the same on every run). The weights of 3 are set by
// hand. The per-layer statistics (mean absolute value, std, rms, share of zeros) are computed by this program in double from
// the library's tensors; `derived` holds ratios of recorded numbers.
#include "trace_util.hpp"

#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"
#include "torc/utils.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

using namespace torc;
using trace::J;

namespace {

constexpr int DEPTH = 8, WIDTH = 32, BATCH = 64, DATA_SEED = 11;
constexpr unsigned WEIGHT_SEED = 300;      // layer l (1..8) uses seed 300 + l, the head 300 + 9

double uniform(std::mt19937& rng) { return static_cast<double>(rng()) / 4294967296.0; }   // [0, 1)

struct Stats { double mean_abs, std, rms, frac_zero; };
Stats stats_of(const Tensor& t) {
    const int n = t.numel();
    double s = 0, s2 = 0, a = 0;
    int zeros = 0;
    for (int i = 0; i < n; ++i) {
        const double v = t.data()[i];
        s += v; s2 += v * v; a += std::fabs(v);
        zeros += v == 0.0 ? 1 : 0;
    }
    const double mean = s / n;
    return {a / n, std::sqrt(std::fmax(0.0, s2 / n - mean * mean)), std::sqrt(s2 / n), static_cast<double>(zeros) / n};
}

// A stack of DEPTH x (Linear(WIDTH, WIDTH) -> activation) and a head Linear(WIDTH, 1). The Linear layers are kept as raw
// pointers (the Sequential owns them) to read their weights and gradients.
struct Stack {
    nn::Sequential seq;
    std::vector<nn::Linear*> layers;   // DEPTH hidden layers, then the head
};

std::unique_ptr<Stack> make_stack(bool use_sigmoid, float init_std) {
    auto s = std::make_unique<Stack>();
    for (int l = 1; l <= DEPTH + 1; ++l) {
        const int out = l <= DEPTH ? WIDTH : 1;
        auto lin = std::make_unique<nn::Linear>(WIDTH, out, init_std, WEIGHT_SEED + static_cast<unsigned>(l));
        s->layers.push_back(lin.get());
        s->seq.add(std::move(lin));
        if (l <= DEPTH) {
            if (use_sigmoid) s->seq.add(std::make_unique<nn::Sigmoid>());
            else s->seq.add(std::make_unique<nn::ReLU>());
        }
    }
    return s;
}

// The signal after each layer's activation, with the same modules the Sequential holds, applied one by one (no gradient).
std::vector<Tensor> forward_signal(Stack& s, const Tensor& x, bool use_sigmoid) {
    Variable::NoGradGuard ng;
    std::vector<Tensor> out{x};
    Variable cur(x, false);
    for (int l = 0; l < DEPTH; ++l) {
        Variable z = (*s.layers[l])(cur);
        Variable a = use_sigmoid ? sigmoid(z) : relu(z);
        out.push_back(a.data());
        cur = a;
    }
    return out;
}

// One forward pass of the whole Sequential and one backward() of the MSE loss; returns the loss.
float one_backward(Stack& s, const Tensor& x, const Tensor& t) {
    const Variable xv(x, false), tv(t, false);
    Variable y = s.seq(xv);
    nn::MSELoss mse;
    Variable L = mse(y, tv);
    L.backward();
    return L.data().data()[0];
}

J vanishing(const Tensor& x, const Tensor& t, bool& ok) {
    J runs = J::array();
    for (bool sig : {true, false}) {
        auto s = make_stack(sig, 0.0f);
        const std::vector<Tensor> signal = forward_signal(*s, x, sig);
        const float loss = one_backward(*s, x, t);
        J layers = J::array();
        std::vector<double> g;
        for (int l = 0; l <= DEPTH; ++l) {
            const Variable& w = s->layers[l]->named_parameters().at("weight");
            if (!w.has_grad()) { ok = false; continue; }
            g.push_back(stats_of(w.grad()).mean_abs);
            J e = J::object();
            e.set("layer", l + 1).set("name", l < DEPTH ? "hidden " + std::to_string(l + 1) : std::string("head"))
                .set("mean_abs_grad", g.back());
            if (l < DEPTH) e.set("signal_std", stats_of(signal[l + 1]).std);
            layers.push(std::move(e));
        }
        J d = J::object();
        d.set("first_over_last_hidden", g[0] / g[DEPTH - 1]);
        J r = J::object();
        r.set("activation", sig ? "sigmoid" : "relu").set("loss", loss).set("input_std", stats_of(signal[0]).std)
            .set("layers", std::move(layers)).set("derived", std::move(d));
        runs.push(std::move(r));
    }
    return runs;
}

J initialization(const Tensor& x, const Tensor& t, bool& ok) {
    struct Init { const char* name; float std; };
    const std::vector<Init> inits = {{"kaiming", 0.0f}, {"too_small", 0.05f}, {"too_large", 0.5f}};
    J runs = J::array();
    for (const Init& in : inits) {
        auto s = make_stack(false, in.std);
        const std::vector<Tensor> signal = forward_signal(*s, x, false);
        const float loss = one_backward(*s, x, t);
        J layers = J::array();
        for (int l = 0; l <= DEPTH; ++l) {
            const Stats st = stats_of(signal[l]);
            J e = J::object();
            e.set("layer", l).set("std", st.std).set("rms", st.rms).set("frac_zero", st.frac_zero);
            if (l > 0) {
                const Variable& w = s->layers[l - 1]->named_parameters().at("weight");
                if (!w.has_grad()) ok = false;
                else e.set("mean_abs_grad", stats_of(w.grad()).mean_abs);
            }
            layers.push(std::move(e));
        }
        const Stats w1 = stats_of(s->layers[0]->named_parameters().at("weight").data());
        J r = J::object();
        r.set("name", in.name).set("init_std", static_cast<double>(in.std))
            .set("weight_std", in.std > 0.0f ? static_cast<double>(in.std) : static_cast<double>(kaiming_normal_std(WIDTH)))
            .set("layer1_weight_std_observed", w1.std).set("loss", loss).set("layers", std::move(layers));
        runs.push(std::move(r));
    }
    return runs;
}

// ---- twins ------------------------------------------------------------------------------------------------------------------
struct Tiny {
    nn::Sequential seq;
    nn::Linear* l1 = nullptr;
    nn::Linear* l2 = nullptr;
};

std::unique_ptr<Tiny> make_tiny(const std::vector<float>& w1, const std::vector<float>& b1, const std::vector<float>& w2) {
    auto m = std::make_unique<Tiny>();
    auto a = std::make_unique<nn::Linear>(2, 2, 0.1f, 1), b = std::make_unique<nn::Linear>(2, 1, 0.1f, 1);  // overwritten below
    m->l1 = a.get();
    m->l2 = b.get();
    m->seq.add(std::move(a));
    m->seq.add(std::make_unique<nn::ReLU>());
    m->seq.add(std::move(b));
    auto set = [](nn::Linear* l, const char* name, const std::vector<float>& v) {
        Tensor& d = l->named_parameters().at(name).data();
        for (int i = 0; i < d.numel(); ++i) d.data()[i] = v[static_cast<size_t>(i)];
    };
    set(m->l1, "weight", w1);
    set(m->l1, "bias", b1);
    set(m->l2, "weight", w2);
    set(m->l2, "bias", {0.0f});
    return m;
}

constexpr float TWIN_LR = 0.2f;
constexpr int TWIN_STEPS = 10;
const std::vector<int> LONG_STEPS = {0, 10, 100, 500, 2000};

// neuron k: its row of W1 (2 weights), its bias, its output weight
J neuron_json(const Tensor& w1, const Tensor& b1, const Tensor& w2, int k) {
    J n = J::object();
    n.set("w", trace::jarr(std::vector<float>{w1.data()[2 * k], w1.data()[2 * k + 1]})).set("b", b1.data()[k]).set("v", w2.data()[k]);
    return n;
}

bool neurons_equal(const Tensor& w1, const Tensor& b1, const Tensor& w2) {
    return w1.data()[0] == w1.data()[2] && w1.data()[1] == w1.data()[3] && b1.data()[0] == b1.data()[1] && w2.data()[0] == w2.data()[1];
}

J twins_run(const char* name, const std::vector<float>& w1, const std::vector<float>& b1, const std::vector<float>& w2,
            const Tensor& x, const Tensor& t, bool expect_equal, bool& ok) {
    auto m = make_tiny(w1, b1, w2);
    optim::SGD opt(m->seq.parameters(), TWIN_LR);
    nn::MSELoss mse;
    const Variable xv(x, false), tv(t, false);
    auto P = [&](nn::Linear* l, const char* p) -> Variable& { return l->named_parameters().at(p); };
    J steps = J::array(), long_run = J::array();
    bool equal_all = true;
    for (int step = 0; step <= LONG_STEPS.back(); ++step) {
        Variable y = m->seq(xv);
        Variable L = mse(y, tv);
        L.backward();
        const bool params_eq = neurons_equal(P(m->l1, "weight").data(), P(m->l1, "bias").data(), P(m->l2, "weight").data());
        const bool grads_eq = neurons_equal(P(m->l1, "weight").grad(), P(m->l1, "bias").grad(), P(m->l2, "weight").grad());
        if (step <= TWIN_STEPS) {
            J e = J::object();
            e.set("step", step).set("loss", L.data().data()[0])
                .set("neuron0", neuron_json(P(m->l1, "weight").data(), P(m->l1, "bias").data(), P(m->l2, "weight").data(), 0))
                .set("neuron1", neuron_json(P(m->l1, "weight").data(), P(m->l1, "bias").data(), P(m->l2, "weight").data(), 1))
                .set("grad0", neuron_json(P(m->l1, "weight").grad(), P(m->l1, "bias").grad(), P(m->l2, "weight").grad(), 0))
                .set("grad1", neuron_json(P(m->l1, "weight").grad(), P(m->l1, "bias").grad(), P(m->l2, "weight").grad(), 1))
                .set("output_bias", P(m->l2, "bias").data().data()[0]).set("neurons_equal", params_eq && grads_eq);
            steps.push(std::move(e));
        }
        equal_all = equal_all && params_eq && grads_eq;
        for (int s : LONG_STEPS)
            if (s == step) {
                J e = J::object();
                std::vector<float> pred = trace::values_of(y.data());
                e.set("step", step).set("loss", L.data().data()[0]).set("prediction", trace::jarr(pred)).set("neurons_equal", params_eq);
                long_run.push(std::move(e));
            }
        if (step == LONG_STEPS.back()) break;
        opt.step();
        opt.zero_grad();
    }
    if (equal_all != expect_equal) ok = false;
    J r = J::object();
    r.set("id", name).set("start", J::object()
                                       .set("W1", trace::jarr(w1)).set("b1", trace::jarr(b1)).set("W2", trace::jarr(w2)).set("b2", 0.0f))
        .set("steps", std::move(steps)).set("equal_every_step", equal_all).set("long_run", std::move(long_run));
    return r;
}

}  // namespace

int main() {
    bool ok = true;

    // ---- fixed batch for 1 and 2: inputs uniform in [-sqrt 3, sqrt 3) (std 1), targets uniform in [-1, 1) -------------------
    std::mt19937 rng(DATA_SEED);
    std::vector<float> xs, ts;
    for (int i = 0; i < BATCH * WIDTH; ++i) xs.push_back(static_cast<float>(std::sqrt(3.0) * (2.0 * uniform(rng) - 1.0)));
    for (int i = 0; i < BATCH; ++i) ts.push_back(static_cast<float>(2.0 * uniform(rng) - 1.0));
    const Tensor x = trace::make_tensor({BATCH, WIDTH}, xs), t = trace::make_tensor({BATCH, 1}, ts);

    J van = vanishing(x, t, ok);
    J init = initialization(x, t, ok);

    // ---- twins on XOR ---------------------------------------------------------------------------------------------------------
    const Tensor xx = trace::make_tensor({4, 2}, {0, 0, 0, 1, 1, 0, 1, 1}), tx = trace::make_tensor({4, 1}, {0, 1, 1, 0});
    J twins = J::array();
    twins.push(twins_run("twins", {0.5f, -0.4f, 0.5f, -0.4f}, {0.1f, 0.1f}, {0.3f, 0.3f}, xx, tx, true, ok));
    twins.push(twins_run("different", {0.5f, -0.4f, -0.3f, 0.6f}, {0.1f, 0.1f}, {0.3f, 0.3f}, xx, tx, false, ok));

    J setup = J::object();
    setup.set("depth", DEPTH).set("width", WIDTH).set("batch", BATCH).set("data_seed", DATA_SEED)
        .set("inputs", "uniform in [-sqrt(3), sqrt(3)), std 1").set("targets", "uniform in [-1, 1)")
        .set("weight_seeds", "layer l uses 300 + l (head: 309)").set("loss", "nn::MSELoss")
        .set("kaiming_std", kaiming_normal_std(WIDTH));
    J tsetup = J::object();
    tsetup.set("model", "Linear(2, 2) -> ReLU -> Linear(2, 1)").set("x", trace::tensor_json(xx)).set("t", trace::tensor_json(tx))
        .set("optimizer", "optim::SGD").set("lr", static_cast<double>(TWIN_LR)).set("momentum", 0).set("loss", "nn::MSELoss")
        .set("recorded_steps", TWIN_STEPS).set("long_steps", trace::jarr(LONG_STEPS))
        .set("note", "grad0/grad1 are read after backward(), before step(); step k is the state before update k + 1");

    J doc = trace::envelope("deep", "deep-nets", "tools/trace/trace_deep.cpp");
    doc.set("setup", std::move(setup)).set("vanishing", std::move(van)).set("init", std::move(init))
        .set("twins_setup", std::move(tsetup)).set("twins", std::move(twins));
    std::fputs(doc.dump().c_str(), stdout);
    if (!ok) { std::cerr << "trace_deep: a self-check failed (a missing gradient, or the twins did not behave as expected)\n"; return 1; }
    return 0;
}
