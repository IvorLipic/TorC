// trace_linear.cpp  ->  site/data/linear.json   (lesson `linear-layer`, docs/content-plan.md 3.3)
//
// Runs the real nn::Linear / nn::MSELoss / nn::ReLU / nn::Conv2d / nn::Sequential and records:
//   scenes[]        `linear` (x@W^T + b with MSE, hand-built from real ops) and `linear-dx` (x trainable, loss = sum(y)),
//                   each with a parity check against the real nn::Linear module
//   shape_errors[]  exact what() text of wrong-shape calls
//   init_stats[]    empirical mean/std of the weights nn::Linear really produced for (in, out, seed), with a histogram
//                   (and the weights themselves when there are at most 1024)
//   signal[]        std of activations through six Linear(256,256)+ReLU layers for three initialisations
//   param_counts[]  parameter counts of Linear(784,32) and of the MLP/CNN of the library's examples (built for real)
//
// std::normal_distribution is implementation-defined: init_stats and signal depend on the C++ standard library that
// produced them (libstdc++ here). The JSON therefore SHIPS the numbers; a consumer must never try to reproduce the RNG.
#include "trace_util.hpp"

#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/conv.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"

#include <charconv>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>

using namespace torc;
using trace::J;

namespace {

Tensor mat(int r, int c, const std::vector<float>& v) { return trace::make_tensor({r, c}, v); }

struct Stats {
    double mean = 0, std = 0, min = 0, max = 0, rms = 0, frac_zero = 0;
};

// Population statistics in double over a float32 buffer.
Stats stats_of(const float* p, size_t n) {
    Stats s;
    double sum = 0, sq = 0;
    size_t zeros = 0;
    s.min = p[0];
    s.max = p[0];
    for (size_t i = 0; i < n; ++i) {
        double v = p[i];
        sum += v;
        sq += v * v;
        s.min = std::fmin(s.min, v);
        s.max = std::fmax(s.max, v);
        if (p[i] == 0.0f) ++zeros;
    }
    s.mean = sum / static_cast<double>(n);
    s.std = std::sqrt(std::fmax(0.0, sq / static_cast<double>(n) - s.mean * s.mean));
    s.rms = std::sqrt(sq / static_cast<double>(n));
    s.frac_zero = static_cast<double>(zeros) / static_cast<double>(n);
    return s;
}

// The decimal the author typed for a float parameter (0.01f -> 0.01, not 0.00999999977...).
double typed(float f) {
    char b[48];
    auto r = std::to_chars(b, b + sizeof b, f);
    return std::stod(std::string(b, r.ptr));
}

// ---- parity helper: run the real nn::Linear on the same numbers and compare with the hand-built scene ----------------
struct Parity {
    bool output_equal, grads_equal, loss_equal;
    float max_abs_diff;
};

J parity_json(const Parity& p, const char* against) {
    J o = J::object();
    o.set("against", against).set("output_equal", p.output_equal).set("grads_equal", p.grads_equal).set("loss_equal", p.loss_equal)
        .set("max_abs_diff", p.max_abs_diff);
    return o;
}

nn::Linear make_example_linear(const Tensor& W, const Tensor& b) {
    nn::Linear lin(3, 2, 0.0f, 1u);
    lin.named_parameters().at("weight").data() = W;
    lin.named_parameters().at("bias").data() = b;
    return lin;
}

J shape_error(const std::string& id, const std::string& call, int in, int out, const std::vector<int>& x_shape, const trace::Caught& c) {
    J o = J::object();
    o.set("id", id).set("call", call);
    if (in > 0) o.set("in", in).set("out", out);
    if (!x_shape.empty()) o.set("x_shape", trace::jarr(x_shape));
    o.set("exception", c.type).set("what", c.what);
    return o;
}

}  // namespace

int main() {
    bool ok = true;

    // =================================================================================================================
    // Scene `linear`: y = x W^T + b, L = mse(y, t)
    // =================================================================================================================
    const Tensor X = mat(2, 3, {1, 2, 3, -1, 0, 2});
    const Tensor W = mat(2, 3, {0.5f, -1, 2, 1, 0, -0.5f});
    const Tensor B = trace::make_tensor({2}, {0.1f, -0.2f});
    const Tensor T = mat(2, 2, {4, 0, 3, -2});

    trace::Scene s1;
    s1.id = "linear";
    s1.title = "A Linear layer, spelled out";
    s1.expression = "L = mse( x @ W^T + b, t )";
    s1.note = "Linear(3,2) does exactly this: transpose the {out,in} weight, matmul with the batch, add the bias row to every sample.";
    s1.leaves = {{"x", X}, {"W", W}, {"b", B}, {"t", T}};
    s1.constants = {"x", "t"};
    s1.build = [](trace::Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
        Variable& wt = g.op("Wt", "transpose", transpose(*L["W"]));
        Variable& xwt = g.op("xWt", "matmul", matmul(*L["x"], wt));
        Variable& y = g.op("y", "add", add(xwt, *L["b"]));
        return g.op("L", "mse", nn::MSELoss{}(y, *L["t"]));
    };
    trace::SceneResult r1 = trace::run_scene(s1);
    {
        nn::Linear lin = make_example_linear(W, B);
        Variable x(X, false), t(T, false);
        Variable y = lin(x);
        Variable L = nn::MSELoss{}(y, t);
        L.backward();
        Parity p;
        p.output_equal = trace::tensors_identical(y.data(), r1.value.at("y"));
        p.loss_equal = L.data() == r1.value.at("L");
        p.grads_equal = trace::tensors_identical(lin.named_parameters().at("weight").grad(), r1.grad.at("W")) &&
                        trace::tensors_identical(lin.named_parameters().at("bias").grad(), r1.grad.at("b"));
        p.max_abs_diff = std::max({trace::max_abs_diff(y.data(), r1.value.at("y")), trace::max_abs_diff(L.data(), r1.value.at("L")),
                                   trace::max_abs_diff(lin.named_parameters().at("weight").grad(), r1.grad.at("W")),
                                   trace::max_abs_diff(lin.named_parameters().at("bias").grad(), r1.grad.at("b"))});
        r1.json.set("parity", parity_json(p, "nn::Linear(3,2) + nn::MSELoss"));
        ok = ok && p.output_equal && p.grads_equal && p.loss_equal;
    }

    // Hand-check numbers (derived in double from the recorded tensors, not library output):
    //   errors = y - t,  dY = errors/2 (= 2*errors/N with N = 4),  gW = dY^T x,  gb = column sums of dY
    J example = J::object();
    {
        const Tensor& y = r1.value.at("y");
        std::vector<float> err(4), dY(4);
        for (int i = 0; i < 4; ++i) {
            err[i] = y.data()[i] - T.data()[i];
            dY[i] = err[i] * 0.5f;
        }
        double gw_diff = 0, gb_diff = 0;
        const Tensor& gW = r1.grad.at("W");
        const Tensor& gb = r1.grad.at("b");
        for (int k = 0; k < 2; ++k) {
            for (int j = 0; j < 3; ++j) {
                double acc = 0;
                for (int i = 0; i < 2; ++i) acc += static_cast<double>(dY[i * 2 + k]) * X.data()[i * 3 + j];
                gw_diff = std::fmax(gw_diff, std::fabs(acc - gW.data()[k * 3 + j]));
            }
            double acc = 0;
            for (int i = 0; i < 2; ++i) acc += dY[i * 2 + k];
            gb_diff = std::fmax(gb_diff, std::fabs(acc - gb.data()[k]));
        }
        example.set("derived", true)
            .set("note", "errors = y - t and dY = errors/2 are plain arithmetic on recorded tensors; the checks compare dY^T x and column sums of dY with the library's W.grad and b.grad")
            .set("errors", trace::tensor_json(trace::make_tensor({2, 2}, err)))
            .set("dY", trace::tensor_json(trace::make_tensor({2, 2}, dY)))
            .set("gW_equals_dYT_x_max_abs_diff", gw_diff)
            .set("gb_equals_colsum_dY_max_abs_diff", gb_diff);
        ok = ok && gw_diff < 1e-5 && gb_diff < 1e-5;
    }

    // =================================================================================================================
    // Scene `linear-dx`: the input is trainable, loss = sum(y); dX = column sums of W for every row
    // =================================================================================================================
    trace::Scene s2;
    s2.id = "linear-dx";
    s2.title = "Gradient with respect to the input";
    s2.expression = "S = sum( x @ W^T + b )";
    s2.note = "With a sum loss every output has gradient 1, so each row of dX is the column sums of W.";
    s2.leaves = {{"x", X}, {"W", W}, {"b", B}};
    s2.build = [](trace::Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
        Variable& wt = g.op("Wt", "transpose", transpose(*L["W"]));
        Variable& xwt = g.op("xWt", "matmul", matmul(*L["x"], wt));
        Variable& y = g.op("y", "add", add(xwt, *L["b"]));
        return g.op("S", "sum", sum(y));
    };
    trace::SceneResult r2 = trace::run_scene(s2);
    {
        nn::Linear lin = make_example_linear(W, B);
        Variable x(X, true);
        Variable y = lin(x);
        Variable S = sum(y);
        S.backward();
        Parity p;
        p.output_equal = trace::tensors_identical(y.data(), r2.value.at("y"));
        p.loss_equal = S.data() == r2.value.at("S");
        p.grads_equal = trace::tensors_identical(x.grad(), r2.grad.at("x")) &&
                        trace::tensors_identical(lin.named_parameters().at("weight").grad(), r2.grad.at("W")) &&
                        trace::tensors_identical(lin.named_parameters().at("bias").grad(), r2.grad.at("b"));
        p.max_abs_diff = std::max({trace::max_abs_diff(y.data(), r2.value.at("y")), trace::max_abs_diff(x.grad(), r2.grad.at("x")),
                                   trace::max_abs_diff(lin.named_parameters().at("weight").grad(), r2.grad.at("W")),
                                   trace::max_abs_diff(lin.named_parameters().at("bias").grad(), r2.grad.at("b"))});
        r2.json.set("parity", parity_json(p, "nn::Linear(3,2) with a trainable input, loss = sum"));
        ok = ok && p.output_equal && p.grads_equal && p.loss_equal;
        // dX rows == column sums of W
        double d = 0;
        for (int j = 0; j < 3; ++j) {
            double cs = static_cast<double>(W.data()[j]) + W.data()[3 + j];
            for (int i = 0; i < 2; ++i) d = std::fmax(d, std::fabs(cs - r2.grad.at("x").data()[i * 3 + j]));
        }
        example.set("dX_equals_colsum_W_max_abs_diff", d);
        ok = ok && d < 1e-6;
    }

    // =================================================================================================================
    // shape_errors: exact what() text of real wrong-shape calls
    // =================================================================================================================
    J shape_errors = J::array();
    {
        auto lin_call = [&](int in, int out, std::vector<int> xs) {
            nn::Linear lin(in, out, 0.0f, 1u);
            Variable x{Tensor(xs), false};
            return trace::catch_call([&] { Variable y = lin(x); });
        };
        auto call_str = [](int in, int out, const std::vector<int>& xs) {
            return "Linear(" + std::to_string(in) + ", " + std::to_string(out) + ") applied to x of shape " + shape_to_string(xs);
        };
        for (auto [in, out, xs] : std::vector<std::tuple<int, int, std::vector<int>>>{
                 {3, 2, {2, 2}}, {784, 32, {8, 10}}, {3, 2, {4, 4}}, {3, 2, {3}}}) {
            shape_errors.push(shape_error("linear-" + std::to_string(in) + "x" + std::to_string(out) + "-" + shape_to_string(xs), call_str(in, out, xs), in, out, xs, lin_call(in, out, xs)));
        }
        shape_errors.push(shape_error("linear-zero-features", "Linear(0, 2)", 0, 0, {}, trace::catch_call([] { nn::Linear l(0, 2); })));
        shape_errors.push(shape_error("mse-shapes", "MSELoss()(y of shape (2, 2), t of shape (3, 2))", 0, 0, {},
                                      trace::catch_call([] {
                                          Variable a{Tensor(std::vector<int>{2, 2}), true}, b{Tensor(std::vector<int>{3, 2}), false};
                                          Variable l = nn::MSELoss{}(a, b);
                                      })));
        // the successful call, for contrast
        {
            nn::Linear lin(3, 2, 0.0f, 1u);
            Variable x{Tensor(std::vector<int>{4, 3}), false};
            J o = J::object();
            Variable y = lin(x);
            o.set("id", "linear-3x2-(4, 3)").set("call", "Linear(3, 2) applied to x of shape (4, 3)").set("in", 3).set("out", 2)
                .set("x_shape", trace::jarr(x.data().shape())).set("exception", "none").set("what", "").set("y_shape", trace::jarr(y.data().shape()));
            shape_errors.push(std::move(o));
        }
    }

    // =================================================================================================================
    // init_stats: what nn::Linear(in, out, init_std, seed) really produces
    // =================================================================================================================
    J init_stats = J::array();
    {
        struct Cfg { int in, out; float init_std; unsigned seed; };
        for (Cfg c : {Cfg{3, 2, 0.0f, 7}, Cfg{32, 32, 0.0f, 1}, Cfg{784, 32, 0.0f, 1}, Cfg{256, 256, 0.0f, 1}, Cfg{784, 32, 0.01f, 1}, Cfg{784, 32, 1.0f, 1}}) {
            nn::Linear lin(c.in, c.out, c.init_std, c.seed);
            const Tensor& w = lin.named_parameters().at("weight").data();
            const Tensor& bias = lin.named_parameters().at("bias").data();
            const size_t n = static_cast<size_t>(w.numel());
            Stats st = stats_of(w.data(), n);
            const double expected = c.init_std > 0.0f ? typed(c.init_std) : static_cast<double>(kaiming_normal_std(c.in));
            // histogram over [-4 sigma, 4 sigma], 40 bins, sigma = the std the initialiser was asked for
            const int bins = 40;
            std::vector<int> counts(bins, 0);
            int under = 0, over = 0;
            const double lo = -4.0 * expected, hi = 4.0 * expected;
            for (size_t i = 0; i < n; ++i) {
                double v = w.data()[i];
                if (v < lo) ++under;
                else if (v >= hi) ++over;
                else ++counts[static_cast<size_t>((v - lo) / (hi - lo) * bins)];
            }
            J h = J::object();
            h.set("lo", lo).set("hi", hi).set("bins", bins).set("counts", trace::jarr(counts)).set("under", under).set("over", over);
            J o = J::object();
            o.set("in", c.in).set("out", c.out).set("seed", c.seed).set("init_std", typed(c.init_std))
                .set("kaiming_default", c.init_std <= 0.0f).set("weight_shape", trace::jarr(w.shape())).set("numel", static_cast<int>(n))
                .set("expected_std", c.init_std > 0.0f ? J(expected) : J(kaiming_normal_std(c.in))).set("mean", st.mean).set("std", st.std).set("min", st.min).set("max", st.max)
                .set("std_over_expected", st.std / expected).set("histogram", std::move(h))
                .set("bias_all_zero", stats_of(bias.data(), static_cast<size_t>(bias.numel())).rms == 0.0);
            if (n <= 1024) o.set("weights", trace::tensor_json(w));
            init_stats.push(std::move(o));
            if (n >= 1000) ok = ok && std::fabs(st.std / expected - 1.0) < 0.05;
        }
    }

    // =================================================================================================================
    // signal: std of activations through 6 x (Linear(256,256) + ReLU)
    // =================================================================================================================
    J signal = J::array();
    {
        const int batch = 64, width = 256, layers = 6, extra_seeds = 10;
        std::mt19937 rng(100u);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        Tensor input(std::vector<int>{batch, width});
        for (int i = 0; i < input.numel(); ++i) input.data()[i] = nd(rng);
        const Stats in_stats = stats_of(input.data(), static_cast<size_t>(input.numel()));

        // One pass through the six layers; weights drawn by nn::Linear with seeds `seed_base + layer`.
        struct Pass { std::vector<Stats> pre, post; };
        auto run_pass = [&](float init_std, unsigned seed_base) {
            Variable::NoGradGuard ng;
            Pass p;
            Variable h(input, false);
            nn::ReLU relu;
            for (int l = 1; l <= layers; ++l) {
                nn::Linear lin(width, width, init_std, seed_base + static_cast<unsigned>(l));
                Variable pre = lin(h);
                Variable post = relu(pre);
                p.pre.push_back(stats_of(pre.data().data(), static_cast<size_t>(pre.data().numel())));
                p.post.push_back(stats_of(post.data().data(), static_cast<size_t>(post.data().numel())));
                h = post;
            }
            return p;
        };

        struct Init { const char* name; float init_std; };
        for (Init in : {Init{"kaiming", 0.0f}, Init{"std_0.01", 0.01f}, Init{"std_1.0", 1.0f}}) {
            Pass p = run_pass(in.init_std, 200u);
            J ls = J::array();
            {
                J l0 = J::object();  // layer 0 = the input itself (no layer applied): every statistic is the input's
                l0.set("layer", 0).set("pre_std", in_stats.std).set("post_std", in_stats.std).set("post_mean", in_stats.mean)
                    .set("post_rms", in_stats.rms).set("frac_zero", in_stats.frac_zero);
                ls.push(std::move(l0));
            }
            for (int l = 0; l < layers; ++l) {
                J lj = J::object();
                lj.set("layer", l + 1).set("pre_std", p.pre[l].std).set("post_std", p.post[l].std).set("post_mean", p.post[l].mean)
                    .set("post_rms", p.post[l].rms).set("frac_zero", p.post[l].frac_zero);
                ls.push(std::move(lj));
            }
            // The same experiment with `extra_seeds` other weight seeds, so a lesson can say how typical the primary run is.
            std::vector<std::vector<double>> per_layer(layers);
            for (int sd = 0; sd < extra_seeds; ++sd) {
                Pass q = run_pass(in.init_std, 1000u + 10u * static_cast<unsigned>(sd));
                for (int l = 0; l < layers; ++l) per_layer[l].push_back(q.post[l].std);
            }
            J across = J::array();
            for (int l = 0; l < layers; ++l) {
                double mn = per_layer[l][0], mx = per_layer[l][0], sm = 0;
                for (double v : per_layer[l]) { mn = std::fmin(mn, v); mx = std::fmax(mx, v); sm += v; }
                J a = J::object();
                a.set("layer", l + 1).set("post_std_mean", sm / extra_seeds).set("post_std_min", mn).set("post_std_max", mx);
                across.push(std::move(a));
            }
            J o = J::object();
            o.set("name", in.name).set("init_std", typed(in.init_std)).set("kaiming_default", in.init_std <= 0.0f)
                .set("width", width).set("batch", batch).set("input_seed", 100).set("weight_seeds", "200 + layer")
                .set("layers", std::move(ls)).set("across_seeds", std::move(across)).set("n_extra_seeds", extra_seeds);
            signal.push(std::move(o));
        }
    }

    // =================================================================================================================
    // param_counts: built for real; counted through Module::parameters() / named_parameters()
    // =================================================================================================================
    using Layers = std::vector<std::pair<std::string, std::unique_ptr<nn::Module>>>;
    auto mlp = [] {
        Layers v;
        v.emplace_back("Linear(784, 32)", std::make_unique<nn::Linear>(784, 32, 0.0f, 1u));
        v.emplace_back("ReLU", std::make_unique<nn::ReLU>());
        v.emplace_back("Linear(32, 32)", std::make_unique<nn::Linear>(32, 32, 0.0f, 2u));
        v.emplace_back("ReLU", std::make_unique<nn::ReLU>());
        v.emplace_back("Linear(32, 10)", std::make_unique<nn::Linear>(32, 10, 0.0f, 3u));
        return v;
    };
    auto cnn = [] {
        Layers v;
        v.emplace_back("Conv2d(1, 8, kernel 3, stride 2, padding 1)", std::make_unique<nn::Conv2d>(1, 8, 3, 2, 1, 0.0f, 1u));
        v.emplace_back("ReLU", std::make_unique<nn::ReLU>());
        v.emplace_back("Conv2d(8, 16, kernel 3, stride 2, padding 1)", std::make_unique<nn::Conv2d>(8, 16, 3, 2, 1, 0.0f, 2u));
        v.emplace_back("ReLU", std::make_unique<nn::ReLU>());
        v.emplace_back("Flatten", std::make_unique<nn::Flatten>());
        v.emplace_back("Linear(784, 10)", std::make_unique<nn::Linear>(16 * 7 * 7, 10, 0.0f, 3u));
        return v;
    };
    J param_counts = J::array();
    auto describe = [&](const std::string& name, const std::string& source, Layers layers, std::vector<int> input_shape) {
        Variable::NoGradGuard ng;
        J lj = J::array();
        Variable cur{Tensor(input_shape), false};
        long layer_total = 0;
        for (auto& [label, mod] : layers) {
            std::vector<std::pair<std::string, Variable*>> ps;
            for (auto& [pn, pv] : mod->named_parameters()) ps.emplace_back(pn, const_cast<Variable*>(&pv));
            std::sort(ps.begin(), ps.end(), [](auto& a, auto& b) { return a.first < b.first; });
            J pj = J::array();
            long cnt = 0;
            for (auto& [pn, pv] : ps) {
                J q = J::object();
                q.set("name", pn).set("shape", trace::jarr(pv->data().shape())).set("numel", pv->data().numel());
                pj.push(std::move(q));
                cnt += pv->data().numel();
            }
            cur = (*mod)(cur);
            J l = J::object();
            l.set("layer", label).set("params", std::move(pj)).set("count", cnt).set("output_shape", trace::jarr(cur.data().shape()));
            lj.push(std::move(l));
            layer_total += cnt;
        }
        // the same layers through the real Sequential::parameters()
        nn::Sequential seq;
        for (auto& [label, mod] : layers) seq.add(std::move(mod));
        long total = 0, tensors = 0;
        for (Variable* p : seq.parameters()) { total += p->data().numel(); ++tensors; }
        J o = J::object();
        o.set("name", name).set("source", source).set("input_shape", trace::jarr(input_shape)).set("layers", std::move(lj))
            .set("total", total).set("tensors", tensors).set("sum_of_layers", layer_total);
        param_counts.push(std::move(o));
        ok = ok && total == layer_total;
        return total;
    };
    {
        Layers single;
        single.emplace_back("Linear(784, 32)", std::make_unique<nn::Linear>(784, 32, 0.0f, 1u));
        long t1 = describe("Linear(784, 32)", "single layer", std::move(single), {1, 784});
        long t2 = describe("MNIST MLP", "examples/mnist_mlp/mnist_mlp.cpp", mlp(), {1, 784});
        long t3 = describe("MNIST CNN", "examples/mnist_cnn/mnist_cnn.cpp", cnn(), {1, 1, 28, 28});
        // formulas the lesson teaches: Linear = in*out + out; Conv2d = out*in*k*k + out
        ok = ok && t1 == 784 * 32 + 32 && t2 == (784 * 32 + 32) + (32 * 32 + 32) + (32 * 10 + 10) &&
             t3 == (8 * 1 * 9 + 8) + (16 * 8 * 9 + 16) + (784 * 10 + 10);
    }

    // =================================================================================================================
    J checks = J::object();
    checks.set("all_internal_checks_passed", ok);

    J doc = trace::envelope("linear", "linear-layer", "tools/trace/trace_linear.cpp");
    J scenes = J::array();
    scenes.push(r1.json);
    scenes.push(r2.json);
    doc.set("scenes", std::move(scenes)).set("example", std::move(example)).set("shape_errors", std::move(shape_errors))
        .set("init_stats", std::move(init_stats)).set("signal", std::move(signal)).set("param_counts", std::move(param_counts))
        .set("checks", std::move(checks));
    std::fputs(doc.dump().c_str(), stdout);

    if (!ok) {
        std::cerr << "trace_linear: an internal check failed (parity, hand formulas, parameter counts or init std)\n";
        return 1;
    }
    return 0;
}
