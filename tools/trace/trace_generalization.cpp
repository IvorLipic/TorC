// trace_generalization.cpp  ->  site/data/generalization.json   (lesson `generalization`)
//
// A model that is too flexible for its data: nn::Linear(15, 1) on the features x, x^2, ..., x^15 (a polynomial of degree 15),
// trained with optim::AdamW and nn::MSELoss on 10 noisy samples of sin(pi x). 100 other noisy samples are the test set, which
// the model never trains on. Three runs differ only in the weight decay: 0, 0.1 and 1. At checkpoints from step 0 to 20000:
// the training loss, the test loss, the sum of the squared weights, and the model's curve on a grid.
// A third set of 30 noisy samples is the validation set (C2): the model never trains on it either, and its loss is recorded at
// every checkpoint. For each run, the checkpoint with the lowest validation loss is the "validation stop" (the step a learner
// would choose without looking at the test set), recorded with the test loss at that step; the checkpoint with the lowest test
// loss is recorded next to it for comparison. The validation samples are drawn after the test samples, so the training and test
// data are the same as before the validation set existed. `settings` repeats the experiment's real settings in one place, so
// the code shown on the page can match the run (C10, D11).
//
// The data are made here, not by TorC: x is evenly spaced (training) or uniform in [-1, 1) (test), and the noise is uniform in
// [-0.3, 0.3). The randomness comes from std::mt19937, whose output the C++ standard fixes, and plain arithmetic. The targets
// use std::sin from the C library, which can differ in the last bit between C libraries; the drift check's tolerance covers that. The start weights are 0, set by hand. The data loaders are data::TensorDataset and
// data::DataLoader, without shuffle (one batch holds all 10 training samples, so one epoch is one step).
// Everything under "runs" is library output; "data" is input, and "derived" is plain arithmetic.
#include "trace_util.hpp"

#include "torc/data.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

#include <cmath>
#include <iostream>
#include <random>

using namespace torc;
using trace::J;

namespace {

constexpr int DEGREE = 15, N_TRAIN = 10, N_TEST = 100, N_VAL = 30, STEPS = 20000, DATA_SEED = 3, BATCH = 64;
constexpr float NOISE = 0.3f, LR = 0.05f;
const std::vector<int> CHECKPOINTS = {0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000,
                                      3000, 5000, 7000, 10000, 15000, 20000};
constexpr double PI = 3.14159265358979323846;

float uniform(std::mt19937& rng) { return static_cast<float>(static_cast<double>(rng()) / 4294967296.0); }   // [0, 1)

std::vector<float> features(const std::vector<float>& xs) {
    std::vector<float> f;
    for (float x : xs) {
        float p = 1.0f;
        for (int k = 0; k < DEGREE; ++k) { p *= x; f.push_back(p); }
    }
    return f;
}

float loss_on(nn::Linear& model, data::DataLoader& loader) {
    Variable::NoGradGuard ng;
    nn::MSELoss mse;
    loader.reset();
    auto [xb, tb] = loader.next_batch();            // each loader holds its whole set in one batch
    const Variable x(xb, false), t(tb, false);
    Variable y = model(x);
    Variable L = mse(y, t);
    return L.data().data()[0];
}

std::vector<float> curve(nn::Linear& model, const Tensor& grid_features) {
    Variable::NoGradGuard ng;
    const Variable x(grid_features, false);
    Variable y = model(x);
    return trace::values_of(y.data());
}

}  // namespace

int main() {
    // ---- the data ---------------------------------------------------------------------------------------------------------
    std::mt19937 rng(DATA_SEED);
    std::vector<float> xtr, ttr, xte, tte, xva, tva, grid;
    for (int i = 0; i < N_TRAIN; ++i) {
        const float x = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N_TRAIN - 1);
        xtr.push_back(x);
        ttr.push_back(static_cast<float>(std::sin(PI * x)) + NOISE * (2.0f * uniform(rng) - 1.0f));
    }
    for (int i = 0; i < N_TEST; ++i) {
        const float x = 2.0f * uniform(rng) - 1.0f;
        xte.push_back(x);
        tte.push_back(static_cast<float>(std::sin(PI * x)) + NOISE * (2.0f * uniform(rng) - 1.0f));
    }
    for (int i = 0; i < N_VAL; ++i) {                          // after the test set: the earlier draws stay the same
        const float x = 2.0f * uniform(rng) - 1.0f;
        xva.push_back(x);
        tva.push_back(static_cast<float>(std::sin(PI * x)) + NOISE * (2.0f * uniform(rng) - 1.0f));
    }
    for (int k = 0; k <= 80; ++k) grid.push_back(static_cast<float>(k - 40) / 40.0f);
    const data::TensorDataset train_set(trace::make_tensor({N_TRAIN, DEGREE}, features(xtr)), trace::make_tensor({N_TRAIN, 1}, ttr));
    const data::TensorDataset test_set(trace::make_tensor({N_TEST, DEGREE}, features(xte)), trace::make_tensor({N_TEST, 1}, tte));
    const data::TensorDataset val_set(trace::make_tensor({N_VAL, DEGREE}, features(xva)), trace::make_tensor({N_VAL, 1}, tva));
    const Tensor grid_features = trace::make_tensor({81, DEGREE}, features(grid));

    // ---- three runs -----------------------------------------------------------------------------------------------------------
    const std::vector<double> DECAYS = {0.0, 0.1, 1.0};
    J runs = J::array();
    for (double wd : DECAYS) {                        // written to the JSON as given; AdamW takes the float
        nn::Linear model(DEGREE, 1, 0.0f, 1);
        for (auto& [name, p] : model.named_parameters())
            for (int i = 0; i < p.data().numel(); ++i) p.data().data()[i] = 0.0f;
        optim::AdamW opt(model.parameters(), LR, static_cast<float>(wd));
        nn::MSELoss loss_fn;
        data::DataLoader train(train_set, BATCH, /*shuffle=*/false);
        data::DataLoader train_eval(train_set, 64), test(test_set, 256), val(val_set, 64);
        struct Point { int step; float train, val, test; };
        std::vector<Point> points;
        J checkpoints = J::array();
        size_t next = 0;
        for (int step = 0; step <= STEPS; ++step) {
            if (next < CHECKPOINTS.size() && CHECKPOINTS[next] == step) {
                const Tensor& w = model.named_parameters().at("weight").data();
                J c = J::object();
                const Point p{step, loss_on(model, train_eval), loss_on(model, val), loss_on(model, test)};
                c.set("step", step).set("train_loss", p.train).set("test_loss", p.test)
                    .set("weight_sum_sq", w.mul(w).sum()).set("curve", trace::jarr(curve(model, grid_features)))
                    .set("val_loss", p.val);
                points.push_back(p);
                checkpoints.push(std::move(c));
                ++next;
            }
            if (step == STEPS) break;
            train.reset();                                  // one epoch = one batch = one step
            while (train.has_next()) {
                auto [xb, tb] = train.next_batch();
                Variable x(xb, false), t(tb, false);
                Variable y = model(x);
                Variable loss = loss_fn(y, t);
                loss.backward();
                opt.step();
                opt.zero_grad();
            }
        }
        // the lowest validation loss over the checkpoints (first one on a tie), and the lowest test loss for comparison
        Point vs = points[0], bt = points[0];
        for (const Point& p : points) {
            if (p.val < vs.val) vs = p;
            if (p.test < bt.test) bt = p;
        }
        J stop = J::object();
        stop.set("step", vs.step).set("val_loss", vs.val).set("test_loss", vs.test).set("train_loss", vs.train);
        J best = J::object();
        best.set("step", bt.step).set("test_loss", bt.test).set("val_loss", bt.val);
        J r = J::object();
        r.set("weight_decay", wd).set("checkpoints", std::move(checkpoints)).set("val_stop", std::move(stop))
            .set("best_test", std::move(best));
        runs.push(std::move(r));
    }

    J setup = J::object();
    setup.set("degree", DEGREE).set("train_samples", N_TRAIN).set("test_samples", N_TEST).set("noise", 0.3)
        .set("lr", 0.05).set("steps", STEPS).set("optimizer", "optim::AdamW").set("loss", "nn::MSELoss");
    J data = J::object();
    data.set("train_x", trace::jarr(xtr)).set("train_t", trace::jarr(ttr)).set("test_x", trace::jarr(xte)).set("test_t", trace::jarr(tte))
        .set("grid", trace::jarr(grid)).set("val_x", trace::jarr(xva)).set("val_t", trace::jarr(tva));
    // the experiment's real settings in one object (what the code on the page must show)
    J settings = J::object();
    settings.set("train_samples", N_TRAIN).set("val_samples", N_VAL).set("test_samples", N_TEST).set("noise", 0.3)
        .set("data_seed", DATA_SEED).set("degree", DEGREE).set("features", "x, x^2, ..., x^15")
        .set("model", "nn::Linear(15, 1)").set("start_weights", "all 0").set("optimizer", "optim::AdamW").set("lr", 0.05)
        .set("weight_decays", trace::jarr(DECAYS)).set("steps", STEPS).set("batch_size", BATCH).set("shuffle", false)
        .set("steps_per_epoch", 1).set("loss", "nn::MSELoss");
    J derived = J::object();
    derived.set("noise_variance", 0.3 * 0.3 / 3.0);   // noise uniform in [-a, a) with a = 0.3: variance a^2 / 3

    J doc = trace::envelope("generalization", "generalization", "tools/trace/trace_generalization.cpp");
    doc.set("setup", std::move(setup)).set("data", std::move(data)).set("runs", std::move(runs)).set("derived", std::move(derived))
        .set("settings", std::move(settings));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
