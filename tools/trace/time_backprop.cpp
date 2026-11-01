// time_backprop.cpp  ->  site/data/backprop-timing.json   (lesson `backprop`)
//
// How long does one backward pass take, measured in forward passes? Three MLPs built like the digit network (784 inputs, two
// hidden layers of `h` neurons with ReLU, 10 outputs, nn::MSELoss), with h = 32 (the digit network, 26,506 parameters), 128 and
// 512, one sample. For each network the program times, best of 5 rounds of `reps` repetitions:
//   forward_us          one forward pass and the loss, with the gradient record on (the forward pass that backprop needs);
//   forward_nograd_us   the same under Variable::NoGradGuard (the forward pass that a nudge needs: no record);
//   forward_backward_us one forward pass, the loss, L.backward() and zero_grad() of every parameter.
// A timing depends on the computer and on its load, so this is NOT a trace program: tools/trace/run.sh does not run it, and its
// --check does not compare this file (it only knows that the file exists; docs/trace-schema.md "backprop-timing.json"). Regenerate
// it by hand on a quiet machine:
//   cmake --build build/trace --target time_backprop && build/trace/time_backprop > site/data/backprop-timing.json
// The lesson binds only ratios rounded to one significant figure, and site/test/backprop.test.mjs checks the file's structure and
// that the ratios agree with its own times.
#include "trace_util.hpp"

#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"

#include <algorithm>
#include <chrono>
#include <memory>

using namespace torc;
using trace::J;

namespace {

constexpr int ROUNDS = 5;
constexpr double ROUND_SECONDS = 0.25;   // each round runs about this long

template <class F>
double best_us(F&& f) {   // microseconds per call, best of ROUNDS rounds
    using clock = std::chrono::steady_clock;
    int reps = 1;
    for (;;) {   // find a repetition count that fills one round
        const auto t0 = clock::now();
        for (int i = 0; i < reps; ++i) f();
        const double s = std::chrono::duration<double>(clock::now() - t0).count();
        if (s >= ROUND_SECONDS / 4 || reps >= (1 << 20)) break;
        reps *= 2;
    }
    reps = std::max(1, reps * 4);
    double best = 1e300;
    for (int r = 0; r < ROUNDS; ++r) {
        const auto t0 = clock::now();
        for (int i = 0; i < reps; ++i) f();
        best = std::min(best, std::chrono::duration<double>(clock::now() - t0).count() / reps);
    }
    return best * 1e6;
}

}  // namespace

int main() {
    constexpr int IN = 784, OUT = 10;
    J runs = J::array();
    double lo = 1e300, hi = 0, bn_lo = 1e300, bn_hi = 0;
    long digit_params = 0, max_params = 0;
    for (int hid : {32, 128, 512}) {
        nn::Sequential m;
        m.add(std::make_unique<nn::Linear>(IN, hid, 0.0f, 1)); m.add(std::make_unique<nn::ReLU>());
        m.add(std::make_unique<nn::Linear>(hid, hid, 0.0f, 2)); m.add(std::make_unique<nn::ReLU>());
        m.add(std::make_unique<nn::Linear>(hid, OUT, 0.0f, 3));
        long params = 0;
        for (auto* p : m.parameters()) params += p->data().numel();
        if (hid == 32) digit_params = params;
        max_params = std::max(max_params, params);
        Tensor xt(std::vector<int>{1, IN}), tt(std::vector<int>{1, OUT});
        for (int i = 0; i < xt.numel(); ++i) xt.data()[i] = 0.5f;
        for (int i = 0; i < tt.numel(); ++i) tt.data()[i] = 0.1f;
        const Variable x(xt, false), t(tt, false);
        nn::MSELoss mse;
        const double f = best_us([&] { Variable y = m(x); Variable L = mse(y, t); });
        const double fng = best_us([&] { Variable::NoGradGuard ng; Variable y = m(x); Variable L = mse(y, t); });
        const double fb = best_us([&] { Variable y = m(x); Variable L = mse(y, t); L.backward(); for (auto* p : m.parameters()) p->zero_grad(); });
        const double b_over_fng = (fb - f) / fng;
        lo = std::min(lo, b_over_fng); hi = std::max(hi, b_over_fng);
        const double bn = fb / fng;   // one forward + one backward pass, counted in nudges (no-record forward passes)
        bn_lo = std::min(bn_lo, bn); bn_hi = std::max(bn_hi, bn);
        J r = J::object();
        r.set("hidden", hid).set("parameters", static_cast<double>(params)).set("forward_us", f).set("forward_nograd_us", fng)
            .set("forward_backward_us", fb).set("backward_over_forward", (fb - f) / f).set("backward_over_forward_nograd", b_over_fng)
            .set("backprop_over_nudge", bn).set("nudging_over_backprop", static_cast<double>(params + 1) * fng / fb);
        runs.push(std::move(r));
    }
    J setup = J::object();
    setup.set("model", "Linear(784, h) -> ReLU -> Linear(h, h) -> ReLU -> Linear(h, 10), nn::MSELoss, one sample")
        .set("rounds", ROUNDS).set("round_seconds", ROUND_SECONDS)
        .set("note", "machine-dependent timing, not drift-checked; backward = forward_backward - forward (with the record)");
    J derived = J::object();
    derived.set("backward_over_forward_nograd_min", lo).set("backward_over_forward_nograd_max", hi)
        .set("backprop_over_nudge_min", bn_lo).set("backprop_over_nudge_max", bn_hi)
        .set("largest_over_digit", static_cast<double>(max_params) / static_cast<double>(digit_params));
    J doc = trace::envelope("timing", "backprop", "tools/trace/time_backprop.cpp");
    doc.set("setup", std::move(setup)).set("runs", std::move(runs)).set("derived", std::move(derived));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
