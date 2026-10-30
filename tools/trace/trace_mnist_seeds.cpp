// trace_mnist_seeds.cpp  ->  site/data/mnist-seeds.json   (lessons `mnist-mlp`, `mnist-cnn`; brief C5)
//
// The headline comparison (MLP against CNN) depends on the random start weights and on the batch order. This trace measures how
// much: each model is trained with 3 seed sets at exactly the settings of its own trace (trace_mnist_mlp.cpp: AdamW lr 5e-4,
// 20 epochs; trace_mnist_cnn.cpp: AdamW lr 1e-3, 5 epochs; both batches of 128, shuffled, nn::CrossEntropyLoss, default
// Kaiming start weights). Seed set 1 is the run of those traces (layer seeds 1, 2, 3, loader seed 11); its numbers are read from
// mnist-mlp.json and mnist-cnn.json, not trained again. Seed sets 2 and 3 (layer seeds 101..103 / 201..203, loader seeds 111 /
// 211) are trained here, all four runs in parallel threads (each run is independent: own model, optimizer and loader; TorC's
// no-grad switch is thread_local). The values depend on libstdc++'s normal_distribution and std::shuffle, like the traces above.
// Keys:
//   setup   {seed_sets: [{id, layer_seeds[3], loader_seed, source}], note}
//   mlp/cnn {settings {epochs, lr, batch_size, optimizer}, runs: [{seed_set, test_correct_by_epoch[epochs + 1] (epoch 0 = before
//           training), test_correct, test_accuracy}], min_accuracy, max_accuracy, mean_accuracy (derived over the 3 runs)}
//   derived {cnn_min_minus_mlp_max}: the smallest CNN accuracy minus the largest MLP accuracy (plain arithmetic)
// MNIST comes from MNIST_DIR (tools/trace/run.sh sets it).
#include "trace_mnist.hpp"

#include "torc/optim.hpp"

#include <algorithm>
#include <iostream>
#include <thread>

using namespace torc;
using trace::J;
namespace tmn = trace_mnist;

namespace {

struct SeedSet { int id; unsigned s1, s2, s3, loader; };
const std::vector<SeedSet> SEEDS = {{1, 1, 2, 3, 11}, {2, 101, 102, 103, 111}, {3, 201, 202, 203, 211}};
constexpr int BATCH = 128, MLP_EPOCHS = 20, CNN_EPOCHS = 5;
constexpr double MLP_LR = 0.0005, CNN_LR = 0.001;

// the training loop of trace_mnist_mlp.cpp / trace_mnist_cnn.cpp; returns the test count before training and after each epoch
std::vector<int> train(bool cnn, const SeedSet& s, const data::Dataset& train_set, const data::Dataset& test_set) {
    auto m = cnn ? tmn::make_cnn(s.s1, s.s2, s.s3) : tmn::make_mlp(s.s1, s.s2, s.s3);
    auto params = m->seq.parameters();
    optim::AdamW opt(params, static_cast<float>(cnn ? CNN_LR : MLP_LR));
    nn::CrossEntropyLoss loss_fn;
    data::DataLoader loader(train_set, BATCH, /*shuffle=*/true, s.loader);
    std::vector<int> counts{tmn::evaluate(m->seq, test_set).correct};
    for (int epoch = 1; epoch <= (cnn ? CNN_EPOCHS : MLP_EPOCHS); ++epoch) {
        loader.reset();
        while (loader.has_next()) {
            auto [xb, yb] = loader.next_batch();
            Variable x(xb, false), y(yb, false);
            Variable out = m->seq(x);
            Variable loss = loss_fn(out, y);
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        counts.push_back(tmn::evaluate(m->seq, test_set).correct);
    }
    return counts;
}

J model_json(const char* name, int epochs, double lr, const std::vector<std::vector<int>>& counts, int n) {
    J settings = J::object();
    settings.set("epochs", epochs).set("lr", lr).set("batch_size", BATCH).set("optimizer", "optim::AdamW");
    J runs = J::array();
    std::vector<double> acc;
    for (size_t i = 0; i < SEEDS.size(); ++i) {
        acc.push_back(static_cast<double>(counts[i].back()) / n);
        J r = J::object();
        r.set("seed_set", SEEDS[i].id).set("test_correct_by_epoch", trace::jarr(counts[i])).set("test_correct", counts[i].back())
            .set("test_accuracy", acc.back());
        runs.push(std::move(r));
    }
    double mean = 0.0;
    for (double a : acc) mean += a / acc.size();
    J o = J::object();
    o.set("model", name).set("settings", std::move(settings)).set("runs", std::move(runs))
        .set("min_accuracy", *std::min_element(acc.begin(), acc.end())).set("max_accuracy", *std::max_element(acc.begin(), acc.end()))
        .set("mean_accuracy", mean);
    return o;
}

std::vector<int> recorded(const tmn::JV& doc) {
    std::vector<int> v;
    for (const auto& e : doc["epochs"].a) v.push_back(e["test_correct"].i());
    return v;
}

}  // namespace

int main() {
    const std::string dir = tmn::mnist_dir("trace_mnist_seeds");
    const tmn::JV mlp_doc = tmn::read_trace("mnist-mlp.json"), cnn_doc = tmn::read_trace("mnist-cnn.json");
    const data::MNISTDataset train_flat(dir + "/mnist_train.csv"), test_flat(dir + "/mnist_test.csv");
    const data::MNISTDataset train_img(dir + "/mnist_train.csv", 0, {1, 28, 28}), test_img(dir + "/mnist_test.csv", 0, {1, 28, 28});
    const int n = static_cast<int>(test_flat.len());

    std::vector<std::vector<int>> mlp(SEEDS.size()), cnn(SEEDS.size());
    mlp[0] = recorded(mlp_doc);
    cnn[0] = recorded(cnn_doc);
    std::vector<std::thread> th;
    for (size_t i = 1; i < SEEDS.size(); ++i) {
        th.emplace_back([&, i] { cnn[i] = train(true, SEEDS[i], train_img, test_img); });
        th.emplace_back([&, i] { mlp[i] = train(false, SEEDS[i], train_flat, test_flat); });
    }
    for (auto& t : th) t.join();
    for (size_t i = 0; i < SEEDS.size(); ++i)
        std::cerr << "seed set " << SEEDS[i].id << ": MLP " << mlp[i].back() << ", CNN " << cnn[i].back() << "\n";

    J sets = J::array();
    for (const auto& s : SEEDS) {
        J o = J::object();
        o.set("id", s.id).set("layer_seeds", trace::jarr(std::vector<int>{static_cast<int>(s.s1), static_cast<int>(s.s2), static_cast<int>(s.s3)}))
            .set("loader_seed", static_cast<int>(s.loader))
            .set("source", s.id == 1 ? "mnist-mlp.json and mnist-cnn.json (the runs of those traces)" : "trained by this trace");
        sets.push(std::move(o));
    }
    J setup = J::object();
    setup.set("seed_sets", std::move(sets)).set("test_samples", n)
        .set("note", "Each model keeps the settings of its own trace, so the two models still differ in epochs and learning rate.");
    J m = model_json("MLP", MLP_EPOCHS, MLP_LR, mlp, n), c = model_json("CNN", CNN_EPOCHS, CNN_LR, cnn, n);
    int mlp_max = 0, cnn_min = n;
    for (size_t i = 0; i < SEEDS.size(); ++i) { mlp_max = std::max(mlp_max, mlp[i].back()); cnn_min = std::min(cnn_min, cnn[i].back()); }
    J derived = J::object();
    derived.set("cnn_min_minus_mlp_max", static_cast<double>(cnn_min - mlp_max) / n).set("cnn_always_ahead", cnn_min > mlp_max);

    J o = trace::envelope("mnist-seeds", "mnist-cnn", "tools/trace/trace_mnist_seeds.cpp");
    o.set("setup", std::move(setup)).set("mlp", std::move(m)).set("cnn", std::move(c)).set("derived", std::move(derived));
    std::fputs(o.dump().c_str(), stdout);
    return 0;
}
