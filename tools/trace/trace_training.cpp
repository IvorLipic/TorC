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
// Everything is library output, except `derived` (plain arithmetic).
#include "trace_util.hpp"

#include "torc/data.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

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
        .set("true_line_loss", true_line_loss).set("derived", std::move(derived));
    std::fputs(doc.dump().c_str(), stdout);
    if (!ok) { std::cerr << "trace_training: the index loader and the data loader disagree on the batch order\n"; return 1; }
    return 0;
}
