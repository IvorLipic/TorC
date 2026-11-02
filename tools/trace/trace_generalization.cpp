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
// "More data" (C10, appended keys only): the same model without weight decay, trained on the 10 training samples plus 90 more,
// drawn after the validation samples (so every earlier draw stays the same), with a full-batch loader. `derived` gets the
// typical misses (square roots of the losses) that the lesson quotes, the loss of the true curve sin(pi x) on each set, and the
// factor 1 - lr * weight_decay by which AdamW scales every parameter at each step, besides its Adam update.
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
constexpr int N_MORE = 90, BATCH_MORE = 128;   // the "more data" run: 10 + 90 training samples, one batch per step
constexpr double LR_SET = 0.05;                     // the learning rate as configured (written to the JSON)
constexpr float NOISE = 0.3f, LR = static_cast<float>(LR_SET);   // what AdamW gets: 0.05f
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
    std::vector<float> xmo = xtr, tmo = ttr;                // the more-data training set: the 10 samples, then 90 more
    for (int i = 0; i < N_MORE; ++i) {                         // after the validation set: the earlier draws stay the same
        const float x = 2.0f * uniform(rng) - 1.0f;
        xmo.push_back(x);
        tmo.push_back(static_cast<float>(std::sin(PI * x)) + NOISE * (2.0f * uniform(rng) - 1.0f));
    }
    for (int k = 0; k <= 80; ++k) grid.push_back(static_cast<float>(k - 40) / 40.0f);
    const data::TensorDataset train_set(trace::make_tensor({N_TRAIN, DEGREE}, features(xtr)), trace::make_tensor({N_TRAIN, 1}, ttr));
    const data::TensorDataset test_set(trace::make_tensor({N_TEST, DEGREE}, features(xte)), trace::make_tensor({N_TEST, 1}, tte));
    const data::TensorDataset val_set(trace::make_tensor({N_VAL, DEGREE}, features(xva)), trace::make_tensor({N_VAL, 1}, tva));
    const data::TensorDataset more_set(trace::make_tensor({N_TRAIN + N_MORE, DEGREE}, features(xmo)), trace::make_tensor({N_TRAIN + N_MORE, 1}, tmo));
    const Tensor grid_features = trace::make_tensor({81, DEGREE}, features(grid));

    // ---- the runs ---------------------------------------------------------------------------------------------------------------
    // One training run of the polynomial from weights 0: AdamW on `train_set` in batches of `batch` (no shuffle), with the losses
    // on the training, validation and test sets at every checkpoint. `out` gets the validation stop and the lowest test loss.
    struct Point { int step; float train, val, test; };
    struct Run { J checkpoints = J::array(); Point stop{}, best{}, last{}; int parameters = 0; };
    auto train_run = [&](const data::TensorDataset& train_set, int batch, double wd) {
        Run out;
        nn::Linear model(DEGREE, 1, 0.0f, 1);
        for (auto& [name, p] : model.named_parameters())
            for (int i = 0; i < p.data().numel(); ++i) p.data().data()[i] = 0.0f;
        for (auto* p : model.parameters()) out.parameters += p->data().numel();
        optim::AdamW opt(model.parameters(), LR, static_cast<float>(wd));
        nn::MSELoss loss_fn;
        data::DataLoader train(train_set, batch, /*shuffle=*/false);
        data::DataLoader train_eval(train_set, 128), test(test_set, 256), val(val_set, 64);
        std::vector<Point> points;
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
                out.checkpoints.push(std::move(c));
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
        out.stop = out.best = points[0];
        for (const Point& p : points) {
            if (p.val < out.stop.val) out.stop = p;
            if (p.test < out.best.test) out.best = p;
        }
        out.last = points.back();
        return out;
    };
    auto stop_json = [](const Point& vs) {
        J stop = J::object();
        stop.set("step", vs.step).set("val_loss", vs.val).set("test_loss", vs.test).set("train_loss", vs.train);
        return stop;
    };
    auto best_json = [](const Point& bt) {
        J best = J::object();
        best.set("step", bt.step).set("test_loss", bt.test).set("val_loss", bt.val);
        return best;
    };
    // typical misses (square roots of the losses, plain arithmetic in double) at the validation stop and at the last step
    auto misses_json = [](const Run& r) {
        auto at = [](const Point& p) {
            J m = J::object();
            const double tr = std::sqrt(static_cast<double>(p.train)), va = std::sqrt(static_cast<double>(p.val));
            m.set("step", p.step).set("train", tr).set("val", va).set("test", std::sqrt(static_cast<double>(p.test))).set("ratio", va / tr);
            return m;
        };
        J m = J::object();
        m.set("stop", at(r.stop)).set("last", at(r.last));
        return m;
    };

    // three runs on the 10 training samples that differ only in the weight decay
    const std::vector<double> DECAYS = {0.0, 0.1, 1.0};
    J runs = J::array(), run_misses = J::array();
    int parameters = 0;
    for (double wd : DECAYS) {                        // written to the JSON as given; AdamW takes the float
        Run out = train_run(train_set, BATCH, wd);
        parameters = out.parameters;
        J r = J::object();
        r.set("weight_decay", wd).set("checkpoints", std::move(out.checkpoints)).set("val_stop", stop_json(out.stop))
            .set("best_test", best_json(out.best));
        runs.push(std::move(r));
        J m = misses_json(out);
        m.set("weight_decay", wd);
        run_misses.push(std::move(m));
    }
    // more data: the same model without weight decay on 10 + 90 training samples, in one batch per step
    Run more = train_run(more_set, BATCH_MORE, 0.0);
    J more_data = J::object();
    more_data.set("train_samples", N_TRAIN + N_MORE).set("batch_size", BATCH_MORE).set("weight_decay", 0.0)
        .set("train_x", trace::jarr(xmo)).set("train_t", trace::jarr(tmo)).set("checkpoints", std::move(more.checkpoints))
        .set("val_stop", stop_json(more.stop)).set("best_test", best_json(more.best));

    J setup = J::object();
    setup.set("degree", DEGREE).set("train_samples", N_TRAIN).set("test_samples", N_TEST).set("noise", 0.3)
        .set("lr", LR_SET).set("steps", STEPS).set("optimizer", "optim::AdamW").set("loss", "nn::MSELoss");
    J data = J::object();
    data.set("train_x", trace::jarr(xtr)).set("train_t", trace::jarr(ttr)).set("test_x", trace::jarr(xte)).set("test_t", trace::jarr(tte))
        .set("grid", trace::jarr(grid)).set("val_x", trace::jarr(xva)).set("val_t", trace::jarr(tva));
    // the experiment's real settings in one object (what the code on the page must show)
    J settings = J::object();
    settings.set("train_samples", N_TRAIN).set("val_samples", N_VAL).set("test_samples", N_TEST).set("noise", 0.3)
        .set("data_seed", DATA_SEED).set("degree", DEGREE).set("features", "x, x^2, ..., x^15")
        .set("model", "nn::Linear(15, 1)").set("start_weights", "all 0").set("optimizer", "optim::AdamW").set("lr", LR_SET)
        .set("weight_decays", trace::jarr(DECAYS)).set("steps", STEPS).set("batch_size", BATCH).set("shuffle", false)
        .set("steps_per_epoch", 1).set("loss", "nn::MSELoss").set("parameters", parameters);
    J derived = J::object();
    derived.set("noise_variance", 0.3 * 0.3 / 3.0);   // noise uniform in [-a, a) with a = 0.3: variance a^2 / 3
    // the mean squared miss of the true curve sin(pi x) on each set: what the noise alone costs (plain arithmetic in double)
    auto wave_loss = [](const std::vector<float>& xs, const std::vector<float>& ts) {
        double s = 0.0;
        for (size_t i = 0; i < xs.size(); ++i) { const double r = std::sin(PI * static_cast<double>(xs[i])) - static_cast<double>(ts[i]); s += r * r; }
        return s / static_cast<double>(xs.size());
    };
    J wave = J::object();
    wave.set("train", wave_loss(xtr, ttr)).set("val", wave_loss(xva, tva)).set("test", wave_loss(xte, tte)).set("more_train", wave_loss(xmo, tmo));
    J decay = J::array();
    for (double wd : DECAYS) decay.push(1.0 - LR_SET * wd);   // AdamW: p <- p - lr * update - lr * wd * p, so each step keeps 1 - lr * wd of p
    J more_misses = misses_json(more);
    derived.set("true_wave_loss", std::move(wave)).set("decay_factor", std::move(decay)).set("misses", std::move(run_misses))
        .set("more_data_misses", std::move(more_misses));

    J doc = trace::envelope("generalization", "generalization", "tools/trace/trace_generalization.cpp");
    doc.set("setup", std::move(setup)).set("data", std::move(data)).set("runs", std::move(runs)).set("derived", std::move(derived))
        .set("settings", std::move(settings)).set("more_data", std::move(more_data));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
