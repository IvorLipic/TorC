// trace_classification.cpp  ->  site/data/classification.json   (lesson `classification`)
//
// Two classes of points in the plane that no straight line separates: class 1 fills a disc in the middle, class 0 is a ring
// around it. Four models are trained on the same 200 training points with the real nn::Sequential, nn::Linear, nn::ReLU,
// nn::CrossEntropyLoss (on two scores per point; TorC has no binary cross-entropy) and optim::Adam, full batch:
//      hidden 0:  Linear(2, 2)                              (the two scores are straight-line functions of the point)
//      hidden H:  Linear(2, H) -> ReLU -> Linear(H, 2)      for H = 2, 4, 8
// At a few checkpoints each model records: the training and test loss (nn::CrossEntropyLoss with no gradient), the number
// and share of training and test points it classifies correctly (class 1 when score 1 > score 0), the probability of
// class 1 (softmax of the two scores along axis 1, the real torc::softmax on the real forward pass) on a 41 x 41 grid over
// the plotted square, every parameter tensor, and each hidden unit's line w1*x + w2*y + b = 0 (row k of the first layer's
// weight and entry k of its bias) with the number of training points on its active side (w1*x + w2*y + b > 0, where relu
// passes the value on; 0 means the unit is dead and gets no gradient). For the model without a hidden layer the decision line (score 1 = score 0) is derived:
// the difference of the two weight rows and of the two biases.
// `sigmoid_identity` shows, for a few pairs of scores, that softmax of the two scores (torc::softmax) gives class 1 the same
// probability as the sigmoid of their difference (torc::sub, then torc::sigmoid). The lesson introduces sigmoid that way.
//
// The points are made here, not by TorC: radius and angle from std::mt19937 (whose output the C++ standard fixes) and plain
// arithmetic, then std::cos/std::sin from the C library (last-bit differences between C libraries are inside the drift check's
// tolerance). The start weights are TorC's default for nn::Linear (Kaiming normal, std sqrt(2 / in)) with fixed seeds, so they
// depend on libstdc++'s std::normal_distribution; the training also depends on the C library's exp and log. On one
// toolchain the output is the same on every run. Everything under "models" and "sigmoid_identity" is library output except
// `derived`, the counts and shares (counted by this program from the library's scores), and `abs_diff`.
#include "trace_util.hpp"

#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

using namespace torc;
using trace::J;

namespace {

constexpr int N_TRAIN = 200, N_TEST = 200, DATA_SEED = 7, STEPS = 1000, GRID_N = 41;
constexpr float LR = 0.05f;
constexpr double R_INNER = 0.9, R_RING_LO = 1.1, R_RING_HI = 1.8, GRID_LO = -2.0, GRID_HI = 2.0;
const std::vector<int> HIDDEN = {0, 2, 4, 8};
const std::vector<int> CHECKPOINTS = {0, 25, 100, STEPS};
constexpr double PI = 3.14159265358979323846;

double uniform(std::mt19937& rng) { return static_cast<double>(rng()) / 4294967296.0; }   // [0, 1)

// Point i has class i % 2 (so each set is half and half). Class 1: uniform over the disc of radius R_INNER.
// Class 0: radius uniform in [R_RING_LO, R_RING_HI), angle uniform.
struct Points { std::vector<float> xy, label; };
Points make_points(std::mt19937& rng, int n) {
    Points p;
    for (int i = 0; i < n; ++i) {
        const int c = i % 2;
        const double u = uniform(rng), a = 2.0 * PI * uniform(rng);
        const double r = c == 1 ? R_INNER * std::sqrt(u) : R_RING_LO + (R_RING_HI - R_RING_LO) * u;
        p.xy.push_back(static_cast<float>(r * std::cos(a)));
        p.xy.push_back(static_cast<float>(r * std::sin(a)));
        p.label.push_back(static_cast<float>(c));
    }
    return p;
}

struct Model {
    int hidden;
    unsigned seed_first, seed_out;
    nn::Sequential seq;
    nn::Linear* first = nullptr;   // the layer whose rows are the hidden units' lines (the only layer when hidden == 0)
    nn::Linear* out = nullptr;
};

std::unique_ptr<Model> make_model(int hidden) {
    auto m = std::make_unique<Model>();
    m->hidden = hidden;
    m->seed_first = 100 + static_cast<unsigned>(hidden);
    m->seed_out = 200 + static_cast<unsigned>(hidden);
    if (hidden == 0) {
        auto l = std::make_unique<nn::Linear>(2, 2, 0.0f, m->seed_first);   // init_std 0: TorC's Kaiming default
        m->first = m->out = l.get();
        m->seq.add(std::move(l));
    } else {
        auto l1 = std::make_unique<nn::Linear>(2, hidden, 0.0f, m->seed_first);
        auto l2 = std::make_unique<nn::Linear>(hidden, 2, 0.0f, m->seed_out);
        m->first = l1.get();
        m->out = l2.get();
        m->seq.add(std::move(l1));
        m->seq.add(std::make_unique<nn::ReLU>());
        m->seq.add(std::move(l2));
    }
    return m;
}

struct Eval { float loss; int correct; };
Eval evaluate(Model& m, const Tensor& x, const Tensor& t) {
    Variable::NoGradGuard ng;
    const Variable xv(x, false), tv(t, false);
    Variable s = m.seq(xv);
    nn::CrossEntropyLoss ce;
    Variable L = ce(s, tv);
    int correct = 0;
    for (int i = 0; i < x.shape()[0]; ++i) {
        const int pred = s.data().data()[2 * i + 1] > s.data().data()[2 * i] ? 1 : 0;
        if (pred == static_cast<int>(t.data()[i])) ++correct;
    }
    return {L.data().data()[0], correct};
}

std::vector<float> prob_grid(Model& m, const Tensor& grid_points) {
    Variable::NoGradGuard ng;
    const Variable g(grid_points, false);
    Variable s = m.seq(g);
    Variable p = softmax(s, 1);
    std::vector<float> out;
    for (int i = 0; i < grid_points.shape()[0]; ++i) out.push_back(p.data().data()[2 * i + 1]);
    return out;
}

J checkpoint_json(Model& m, int step, const Tensor& xtr, const Tensor& ttr, const Tensor& xte, const Tensor& tte,
                  const Tensor& grid_points) {
    const Eval tr = evaluate(m, xtr, ttr), te = evaluate(m, xte, tte);
    J c = J::object();
    c.set("step", step).set("train_loss", tr.loss).set("test_loss", te.loss)
        .set("train_correct", tr.correct).set("train_accuracy", static_cast<double>(tr.correct) / N_TRAIN)
        .set("test_correct", te.correct).set("test_accuracy", static_cast<double>(te.correct) / N_TEST);
    const Tensor& w = m.first->named_parameters().at("weight").data();   // (out, 2), row-major
    const Tensor& b = m.first->named_parameters().at("bias").data();
    if (m.hidden > 0) {
        Variable::NoGradGuard ng;
        const Variable xv(xtr, false);
        Variable z = (*m.first)(xv);                                      // (200, H): each unit's w1*x + w2*y + b
        J units = J::array();
        for (int k = 0; k < m.hidden; ++k) {
            int active = 0;                                               // training points on the side where relu passes
            for (int i = 0; i < N_TRAIN; ++i) active += z.data().data()[i * m.hidden + k] > 0.0f ? 1 : 0;
            J u = J::object();
            u.set("unit", k).set("w1", w.data()[2 * k]).set("w2", w.data()[2 * k + 1]).set("b", b.data()[k])
                .set("active_train_points", active);
            units.push(std::move(u));
        }
        c.set("units", std::move(units));
    } else {
        // score 1 - score 0 = (w[1] - w[0]) . (x, y) + (b[1] - b[0]); the decision line is where it is 0 (derived, double)
        J line = J::object();
        line.set("w1", static_cast<double>(w.data()[2]) - w.data()[0]).set("w2", static_cast<double>(w.data()[3]) - w.data()[1])
            .set("b", static_cast<double>(b.data()[1]) - b.data()[0]);
        J d = J::object();
        d.set("decision_line", std::move(line));
        c.set("derived", std::move(d));
    }
    J layers = J::array();
    for (nn::Linear* l : (m.hidden > 0 ? std::vector<nn::Linear*>{m.first, m.out} : std::vector<nn::Linear*>{m.first})) {
        J lj = J::object();
        lj.set("weight", trace::tensor_json(l->named_parameters().at("weight").data()))
            .set("bias", trace::tensor_json(l->named_parameters().at("bias").data()));
        layers.push(std::move(lj));
    }
    c.set("layers", std::move(layers)).set("prob1", trace::jarr(prob_grid(m, grid_points)));
    return c;
}

}  // namespace

int main() {
    bool ok = true;

    // ---- the data ---------------------------------------------------------------------------------------------------------
    std::mt19937 rng(DATA_SEED);
    const Points train = make_points(rng, N_TRAIN), test = make_points(rng, N_TEST);
    const Tensor xtr = trace::make_tensor({N_TRAIN, 2}, train.xy), ttr = trace::make_tensor({N_TRAIN}, train.label);
    const Tensor xte = trace::make_tensor({N_TEST, 2}, test.xy), tte = trace::make_tensor({N_TEST}, test.label);
    std::vector<float> axis, gp;
    for (int k = 0; k < GRID_N; ++k) axis.push_back(static_cast<float>(k - GRID_N / 2) / 10.0f);   // -2.0 .. 2.0, exact steps
    for (int j = 0; j < GRID_N; ++j)
        for (int i = 0; i < GRID_N; ++i) { gp.push_back(axis[i]); gp.push_back(axis[j]); }
    const Tensor grid_points = trace::make_tensor({GRID_N * GRID_N, 2}, gp);

    // ---- four models ------------------------------------------------------------------------------------------------------
    J models = J::array();
    for (int hidden : HIDDEN) {
        auto m = make_model(hidden);
        optim::Adam opt(m->seq.parameters(), LR);
        nn::CrossEntropyLoss ce;
        const Variable x(xtr, false), t(ttr, false);
        J checkpoints = J::array();
        size_t next = 0;
        for (int step = 0; step <= STEPS; ++step) {
            if (next < CHECKPOINTS.size() && CHECKPOINTS[next] == step) {
                checkpoints.push(checkpoint_json(*m, step, xtr, ttr, xte, tte, grid_points));
                ++next;
            }
            if (step == STEPS) break;
            Variable s = m->seq(x);
            Variable L = ce(s, t);
            L.backward();
            opt.step();
            opt.zero_grad();
        }
        int n_params = 0;
        for (Variable* p : m->seq.parameters()) n_params += p->data().numel();
        J mj = J::object();
        mj.set("hidden", hidden)
            .set("architecture", hidden == 0 ? "Linear(2, 2)" : "Linear(2, " + std::to_string(hidden) + ") -> ReLU -> Linear(" + std::to_string(hidden) + ", 2)")
            .set("parameters", n_params).set("seeds", trace::jarr(hidden == 0 ? std::vector<int>{static_cast<int>(m->seed_first)}
                                                                                : std::vector<int>{static_cast<int>(m->seed_first), static_cast<int>(m->seed_out)}))
            .set("checkpoints", std::move(checkpoints));
        models.push(std::move(mj));
    }

    // ---- softmax of two scores = sigmoid of their difference ------------------------------------------------------------
    const std::vector<std::pair<float, float>> pairs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {-1.0f, 2.0f}, {2.0f, -3.0f}, {4.0f, 4.5f}};
    J ident = J::array();
    for (auto [s0, s1] : pairs) {
        const Variable s(trace::make_tensor({1, 2}, {s0, s1}), false);
        Variable p = softmax(s, 1);
        const Variable a(trace::make_tensor({1}, {s1}), false), b(trace::make_tensor({1}, {s0}), false);
        Variable d = sub(a, b);
        Variable sg = sigmoid(d);
        const float diff = std::fabs(p.data().data()[1] - sg.data().data()[0]);
        ok = ok && diff < 1e-6f;
        J e = J::object();
        e.set("scores", trace::jarr(std::vector<float>{s0, s1})).set("softmax", trace::jarr(trace::values_of(p.data())))
            .set("difference", d.data().data()[0]).set("sigmoid", sg.data().data()[0]).set("abs_diff", diff);
        ident.push(std::move(e));
    }

    J setup = J::object();
    setup.set("train_samples", N_TRAIN).set("test_samples", N_TEST).set("data_seed", DATA_SEED)
        .set("class_1", "uniform over the disc of radius 0.9").set("class_0", "ring, radius uniform in [1.1, 1.8), angle uniform")
        .set("r_inner", R_INNER).set("r_ring", trace::jarr(std::vector<double>{R_RING_LO, R_RING_HI}))
        .set("hidden", trace::jarr(HIDDEN)).set("optimizer", "optim::Adam").set("lr", static_cast<double>(LR)).set("steps", STEPS)
        .set("batch", "full (all 200 training points every step)").set("loss", "nn::CrossEntropyLoss")
        .set("init", "nn::Linear default (Kaiming normal, std sqrt(2 / in)), seeds per model")
        .set("checkpoints", trace::jarr(CHECKPOINTS));
    J data = J::object();
    data.set("train_xy", trace::jarr(train.xy)).set("train_label", trace::jarr(train.label))
        .set("test_xy", trace::jarr(test.xy)).set("test_label", trace::jarr(test.label));
    J grid = J::object();
    grid.set("axis", trace::jarr(axis)).set("n", GRID_N).set("lo", GRID_LO).set("hi", GRID_HI)
        .set("order", "prob1[j * n + i] is the point (axis[i], axis[j]): i moves along x, j along y");

    J doc = trace::envelope("classification", "classification", "tools/trace/trace_classification.cpp");
    doc.set("setup", std::move(setup)).set("data", std::move(data)).set("grid", std::move(grid)).set("models", std::move(models))
        .set("sigmoid_identity", std::move(ident));
    std::fputs(doc.dump().c_str(), stdout);
    if (!ok) { std::cerr << "trace_classification: softmax of two scores differs from the sigmoid of their difference\n"; return 1; }
    return 0;
}
