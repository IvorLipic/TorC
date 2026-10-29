// trace_training.cpp  ->  site/data/training.json   (lesson `training-loop`)
//
// 1. data::SyntheticRegression(100, 1, 3, 1, 0.1, seed 42): the samples, y = 3x + 1 + noise.
// 2. data::DataLoader batch orders: batch size 10 and 32, without and with shuffle, two epochs (reset() between them). The
//    loader is run on a TensorDataset whose x values are the sample indices 0..99, so each batch says which samples it holds.
//    The shuffled order depends only on the seed and the length, and a self-check confirms that a loader with the same seed on
//    the regression data gives the same samples. Also the error next_batch() throws after the last batch.
// 3. Three training runs of the loop in examples/linear_regression: Linear(1, 1), nn::MSELoss, optim::SGD, shuffled batches of
//    100, 32 and 10, the same learning rate and number of epochs. The start weight and bias are set to 0 by hand, so no random
//    generator decides the start. After each epoch: the loss on all 100 samples (no gradient), the weight and the bias.
// 4. The same run with batch size 10, without loader.reset(): the steps taken in each epoch.
// 5. The loss of the true line y = 3x + 1 on the data (the noise that no model can learn).
// 6. (v3, brief C1 and C5) A fair batch-size comparison, `batch_compare`: batch 10 and batch 100, the same lr, start (0, 0) and
//    loop as in 3., for the loader seeds 7, 8 and 9 (the seed decides the shuffled order, the only random part of these runs).
//    Each run takes 300 steps and records the loss on all 100 samples after every step, so the runs can be compared at an equal
//    number of steps (`equal_steps[]`: 30 and 300 steps each) and at an equal number of samples seen (`equal_samples[0]`: 3000
//    each, i.e. 300 steps of 10 and 30 steps of 100), with the min/max/mean over the seeds; `band` = min/max per step. With
//    seed 7 the loss after 3000 samples equals the 30-epoch loss of the runs in 3. bit for bit (a self-check). Also
//    `gradient_estimate`: at w = b = 0, the gradient of each batch of 10 in the first shuffled epoch (seed 7) against the
//    gradient on all 100 samples.
// Everything is library output, except `derived` (plain arithmetic) and the min/max/mean summaries (plain arithmetic on it).
#include "trace_util.hpp"

#include "torc/data.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

#include <algorithm>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

constexpr int N = 100;
constexpr float TRUE_W = 3.0f, TRUE_B = 1.0f, NOISE = 0.1f;
constexpr unsigned DATA_SEED = 42, LOADER_SEED = 7;
constexpr float LR = 0.1f;
constexpr int EPOCHS = 30;

std::vector<int> indices_of(const Tensor& xb) {
    std::vector<int> out;
    for (int i = 0; i < xb.numel(); ++i) out.push_back(static_cast<int>(xb.data()[i]));
    return out;
}

float full_loss(nn::Linear& model, const data::Dataset& ds) {
    Variable::NoGradGuard ng;
    auto [xs, ys] = ds.get_batch(0, ds.len());
    const Variable x(xs, false), y(ys, false);
    nn::MSELoss mse;
    Variable out = model(x);
    Variable L = mse(out, y);
    return L.data().data()[0];
}

void zero_start(nn::Linear& model) {
    model.named_parameters().at("weight").data().data()[0] = 0.0f;
    model.named_parameters().at("bias").data().data()[0] = 0.0f;
}

J epoch_row(int epoch, int steps, nn::Linear& model, const data::Dataset& ds) {
    J e = J::object();
    e.set("epoch", epoch).set("steps", steps).set("loss", full_loss(model, ds))
        .set("weight", model.named_parameters().at("weight").data().data()[0])
        .set("bias", model.named_parameters().at("bias").data().data()[0]);
    return e;
}

// The loop of examples/linear_regression/linear_regression.cpp, one epoch.
int train_epoch(nn::Linear& model, data::DataLoader& loader, optim::SGD& opt) {
    nn::MSELoss loss_fn;
    int steps = 0;
    while (loader.has_next()) {
        auto [x_batch, y_batch] = loader.next_batch();
        Variable x(x_batch, false);
        Variable y(y_batch, false);
        Variable out = model(x);
        Variable loss = loss_fn(out, y);
        loss.backward();
        opt.step();
        opt.zero_grad();
        ++steps;
    }
    return steps;
}


// ---- 6. fair batch-size comparison -----------------------------------------------------------------------------------------
constexpr int CMP_STEPS = 300, CMP_SAMPLES = 3000;
const std::vector<unsigned> CMP_SEEDS = {7, 8, 9};

// The loop of train_epoch, stopped after `max_steps` steps; loss[k] = loss on all samples after k steps (loss[0] = the start).
std::vector<float> train_steps(const data::Dataset& ds, int bs, unsigned seed, int max_steps) {
    nn::Linear model(1, 1, 0.0f, 1);
    zero_start(model);
    optim::SGD opt(model.parameters(), LR);
    data::DataLoader loader(ds, bs, true, seed);
    nn::MSELoss loss_fn;
    std::vector<float> curve{full_loss(model, ds)};
    int steps = 0;
    while (steps < max_steps) {
        loader.reset();
        while (loader.has_next() && steps < max_steps) {
            auto [x_batch, y_batch] = loader.next_batch();
            Variable x(x_batch, false);
            Variable y(y_batch, false);
            Variable out = model(x);
            Variable loss = loss_fn(out, y);
            loss.backward();
            opt.step();
            opt.zero_grad();
            ++steps;
            curve.push_back(full_loss(model, ds));
        }
    }
    return curve;
}

J summary_of(int bs, const std::vector<float>& v) {
    J o = J::object();
    double mean = 0;
    for (float f : v) mean += static_cast<double>(f);
    o.set("batch_size", bs).set("min", *std::min_element(v.begin(), v.end())).set("max", *std::max_element(v.begin(), v.end()))
        .set("mean", mean / static_cast<double>(v.size()));
    return o;
}

}  // namespace

int main() {
    bool ok = true;
    const data::SyntheticRegression dataset(N, 1, TRUE_W, TRUE_B, NOISE, DATA_SEED);

    // ---- 1. the samples --------------------------------------------------------------------------------------------------
    std::vector<float> xs, ys;
    for (size_t i = 0; i < dataset.len(); ++i) {
        auto [x, y] = dataset.get(i);
        xs.push_back(x.data()[0]);
        ys.push_back(y.data()[0]);
    }

    // ---- 2. batch orders -------------------------------------------------------------------------------------------------
    std::vector<float> ids(N), zeros(N, 0.0f);
    for (int i = 0; i < N; ++i) ids[i] = static_cast<float>(i);
    const data::TensorDataset index_ds(trace::make_tensor({N, 1}, ids), trace::make_tensor({N, 1}, zeros));
    J loaders = J::array();
    std::string exhausted_type, exhausted_what;
    for (int bs : {10, 32})
        for (bool shuffle : {false, true}) {
            data::DataLoader loader(index_ds, bs, shuffle, LOADER_SEED);
            data::DataLoader check(dataset, bs, shuffle, LOADER_SEED);
            J epochs = J::array();
            for (int epoch = 0; epoch < 2; ++epoch) {
                if (epoch > 0) { loader.reset(); check.reset(); }
                J batches = J::array();
                while (loader.has_next()) {
                    const std::vector<int> idx = indices_of(loader.next_batch().first);
                    const Tensor cx = check.next_batch().first;
                    for (size_t k = 0; k < idx.size(); ++k) ok = ok && cx.data()[k] == xs[idx[k]];
                    batches.push(trace::jarr(idx));
                }
                epochs.push(std::move(batches));
            }
            if (bs == 32 && !shuffle) {
                auto c = trace::catch_call([&] { (void)loader.next_batch(); });
                exhausted_type = c.type;
                exhausted_what = c.what;
            }
            J l = J::object();
            l.set("batch_size", bs).set("shuffle", shuffle).set("epochs", std::move(epochs));
            loaders.push(std::move(l));
        }
    J exhausted = J::object();
    exhausted.set("exception", exhausted_type).set("what", exhausted_what);

    // ---- 3. three training runs ------------------------------------------------------------------------------------------
    J runs = J::array();
    std::map<int, float> run_final;  // batch size -> loss after the last epoch (checked against 6.)
    for (int bs : {100, 32, 10}) {
        nn::Linear model(1, 1, 0.0f, 1);
        zero_start(model);
        optim::SGD opt(model.parameters(), LR);
        data::DataLoader loader(dataset, bs, true, LOADER_SEED);
        J epochs = J::array();
        epochs.push(epoch_row(0, 0, model, dataset));
        int steps = 0;
        for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
            loader.reset();
            steps += train_epoch(model, loader, opt);
            epochs.push(epoch_row(epoch, steps, model, dataset));
        }
        run_final[bs] = full_loss(model, dataset);
        J r = J::object();
        r.set("batch_size", bs).set("epochs", std::move(epochs));
        runs.push(std::move(r));
    }

    // ---- 4. no reset() -----------------------------------------------------------------------------------------------------
    J no_reset = J::array();
    {
        nn::Linear model(1, 1, 0.0f, 1);
        zero_start(model);
        optim::SGD opt(model.parameters(), LR);
        data::DataLoader loader(dataset, 10, true, LOADER_SEED);
        for (int epoch = 1; epoch <= 3; ++epoch) {
            J e = J::object();
            const int steps = train_epoch(model, loader, opt);       // no loader.reset() before the epoch
            e.set("epoch", epoch).set("steps", steps).set("loss", full_loss(model, dataset));
            no_reset.push(std::move(e));
        }
    }

    // ---- 5. the loss of the true line ---------------------------------------------------------------------------------------
    float true_line_loss = 0.0f;
    {
        auto [x_all, y_all] = dataset.get_batch(0, dataset.len());
        nn::MSELoss mse;
        const Variable pred(x_all.mul(TRUE_W).add(TRUE_B), false), y(y_all, false);
        true_line_loss = mse(pred, y).data().data()[0];
    }

    // ---- 6. fair batch-size comparison (C1, C5) ----------------------------------------------------------------------------
    J batch_compare = J::object();
    bool seed7_ok = true;
    {
        J cruns = J::array(), band = J::array();
        std::map<int, std::vector<std::vector<float>>> curves;  // batch size -> one loss curve per seed (CMP_SEEDS order)
        for (int bs : {10, 100}) {
            for (unsigned seed : CMP_SEEDS) {
                std::vector<float> c = train_steps(dataset, bs, seed, CMP_STEPS);
                // seed 7 is the loader seed of the runs in 3.: the same loop must give the same loss after 3000 samples (30 epochs)
                if (seed == LOADER_SEED) seed7_ok = seed7_ok && c[CMP_SAMPLES / bs] == run_final.at(bs);
                J r = J::object();
                r.set("batch_size", bs).set("seed", static_cast<int>(seed)).set("samples_per_step", bs).set("loss", trace::jarr(c));
                cruns.push(std::move(r));
                curves[bs].push_back(std::move(c));
            }
            J lo = J::array(), hi = J::array();
            for (int k = 0; k <= CMP_STEPS; ++k) {
                float mn = curves[bs][0][k], mx = curves[bs][0][k];
                for (auto& c : curves[bs]) { mn = std::min(mn, c[k]); mx = std::max(mx, c[k]); }
                lo.push(mn);
                hi.push(mx);
            }
            J bd = J::object();
            bd.set("batch_size", bs).set("min", std::move(lo)).set("max", std::move(hi));
            band.push(std::move(bd));
        }
        // one comparison: every run after steps_of(bs) steps, plus min/max/mean over the seeds per batch size
        auto compare = [&](const char* key, int value, auto steps_of) {
            J results = J::array(), summary = J::array();
            for (int bs : {10, 100}) {
                std::vector<float> at;
                for (size_t i = 0; i < CMP_SEEDS.size(); ++i) {
                    const int k = steps_of(bs);
                    J r = J::object();
                    r.set("batch_size", bs).set("seed", static_cast<int>(CMP_SEEDS[i])).set("steps", k).set("samples_seen", k * bs)
                        .set("loss", curves[bs][i][k]);
                    results.push(std::move(r));
                    at.push_back(curves[bs][i][k]);
                }
                summary.push(summary_of(bs, at));
            }
            J o = J::object();
            o.set(key, value).set("results", std::move(results)).set("summary", std::move(summary));
            return o;
        };
        J eq_steps = J::array(), eq_samples = J::array();
        for (int k : {30, CMP_STEPS}) eq_steps.push(compare("steps", k, [k](int) { return k; }));
        eq_samples.push(compare("samples", CMP_SAMPLES, [](int bs) { return CMP_SAMPLES / bs; }));
        // gradient estimate at the start: each batch of 10 (seed 7, first epoch) vs. all 100 samples
        J ge = J::object();
        {
            auto grad_on = [&](const Tensor& xb, const Tensor& yb) {
                nn::Linear m(1, 1, 0.0f, 1);
                zero_start(m);
                nn::MSELoss mse;
                const Variable x(xb, false), y(yb, false);
                Variable out = m(x);
                Variable L = mse(out, y);
                L.backward();
                J o = J::object();
                o.set("gw", m.named_parameters().at("weight").grad().data()[0]).set("gb", m.named_parameters().at("bias").grad().data()[0]);
                return o;
            };
            auto [xa, ya] = dataset.get_batch(0, dataset.len());
            J batches = J::array();
            data::DataLoader loader(dataset, 10, true, LOADER_SEED);
            loader.reset();
            while (loader.has_next()) {
                auto [xb, yb] = loader.next_batch();
                batches.push(grad_on(xb, yb));
            }
            ge.set("at", [] { J o = J::object(); o.set("w", 0).set("b", 0); return o; }()).set("batch_size", 10).set("seed", static_cast<int>(LOADER_SEED))
                .set("full", grad_on(xa, ya)).set("batches", std::move(batches));
        }
        batch_compare.set("batch_sizes", trace::jarr(std::vector<int>{10, 100})).set("seeds", trace::jarr(std::vector<int>{7, 8, 9}))
            .set("seed_decides", "the shuffled order of the samples (DataLoader seed); the data and the start w = b = 0 are the same in every run")
            .set("lr", static_cast<double>(LR)).set("steps", CMP_STEPS)
            .set("runs", std::move(cruns)).set("band", std::move(band)).set("equal_steps", std::move(eq_steps)).set("equal_samples", std::move(eq_samples))
            .set("gradient_estimate", std::move(ge));
    }

    J setup = J::object();
    setup.set("samples", N).set("true_weight", static_cast<double>(TRUE_W)).set("true_bias", static_cast<double>(TRUE_B))
        .set("noise_std", static_cast<double>(NOISE)).set("data_seed", static_cast<int>(DATA_SEED))
        .set("loader_seed", static_cast<int>(LOADER_SEED)).set("lr", static_cast<double>(LR)).set("epochs", EPOCHS);
    J data = J::object();
    data.set("x", trace::jarr(xs)).set("y", trace::jarr(ys));
    J derived = J::object();
    derived.set("noise_variance", static_cast<double>(NOISE) * static_cast<double>(NOISE));

    J doc = trace::envelope("training", "training-loop", "tools/trace/trace_training.cpp");
    doc.set("setup", std::move(setup)).set("data", std::move(data)).set("loaders", std::move(loaders))
        .set("exhausted", std::move(exhausted)).set("runs", std::move(runs)).set("no_reset", std::move(no_reset))
        .set("true_line_loss", true_line_loss).set("derived", std::move(derived)).set("batch_compare", std::move(batch_compare));
    std::fputs(doc.dump().c_str(), stdout);
    if (!seed7_ok) { std::cerr << "trace_training: batch_compare with seed 7 does not reproduce the runs after 30 epochs\n"; return 1; }
    if (!ok) { std::cerr << "trace_training: the index loader and the data loader disagree on the batch order\n"; return 1; }
    return 0;
}
