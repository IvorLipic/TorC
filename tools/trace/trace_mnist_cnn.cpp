// trace_mnist_cnn.cpp  ->  site/data/mnist-cnn.json   (lesson `mnist-cnn`)
//
// The model of examples/mnist_cnn/mnist_cnn.cpp, trained by TorC on MNIST: Sequential Conv2d(1, 8, 3, stride 2, padding 1), ReLU,
// Conv2d(8, 16, 3, 2, 1), ReLU, Flatten, Linear(784, 10); nn::CrossEntropyLoss; optim::AdamW (lr 1e-3, default weight decay
// 0.01); batches of 128, shuffled. 5 epochs: the example uses 10, but each epoch takes about 26 s here, and 5 keep the trace near
// two minutes. The images are read with sample_shape {1, 28, 28}. The seeds of the three layers with weights and of the
// DataLoader are fixed, so the run is the same each time (with libstdc++, whose normal_distribution and std::shuffle the start
// weights and the batch order come from).
// Data: data::MNISTDataset on the CSV files that tools/trace/fetch_mnist.sh writes (raw pixels 0..255; the loader divides by 255).
// The directory comes from the environment variable MNIST_DIR (run.sh sets it).
// Recorded: after each epoch, the mean training loss of the epoch and the accuracy on the 10000 test images (overall and per
// class); at the end, all weights and biases (so the browser can run the same forward pass), 20 test images (the first 16, and
// the first 4 that the model gets wrong) with their pixels, labels and the logits TorC computes for them, and the output of the
// first convolution after relu for the first of them (the browser port is tested against these).
#include "trace_util.hpp"

#include "torc/data.hpp"
#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/conv.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"
#include "torc/optim.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

constexpr int BATCH = 128, EPOCHS = 5, FIRST = 16, WRONG = 4;   // see the note on the epochs above
constexpr float LR = 0.001f;

struct Eval { int correct = 0; std::vector<int> per_correct = std::vector<int>(10, 0), per_total = std::vector<int>(10, 0), wrong; };

int argmax10(const float* row) {
    int best = 0;
    for (int j = 1; j < 10; ++j) if (row[j] > row[best]) best = j;
    return best;
}

Eval evaluate(nn::Sequential& model, const data::MNISTDataset& ds) {
    Variable::NoGradGuard ng;
    data::DataLoader loader(ds, 1000, false);
    Eval e;
    int index = 0;
    while (loader.has_next()) {
        auto [xb, yb] = loader.next_batch();
        const Variable x(xb, false);
        Variable out = model(x);
        for (int i = 0; i < xb.shape()[0]; ++i) {
            const int pred = argmax10(out.data().data() + i * 10), label = static_cast<int>(yb.data()[i]);
            e.per_total[label] += 1;
            if (pred == label) { e.correct += 1; e.per_correct[label] += 1; }
            else e.wrong.push_back(index);
            ++index;
        }
    }
    return e;
}

int count_params(nn::Module& m) {
    int n = 0;
    for (Variable* p : m.parameters()) n += p->data().numel();
    return n;
}

J layer_json(nn::Module& layer) {
    J o = J::object();
    const Tensor& w = layer.named_parameters().at("weight").data();
    const Tensor& b = layer.named_parameters().at("bias").data();
    o.set("weight_shape", trace::jarr(w.shape())).set("weight", trace::jarr(trace::values_of(w))).set("bias", trace::jarr(trace::values_of(b)));
    return o;
}

}  // namespace

int main() {
    const char* dir = std::getenv("MNIST_DIR");
    if (!dir) { std::cerr << "trace_mnist_cnn: set MNIST_DIR (tools/trace/run.sh does)\n"; return 1; }
    const std::string base(dir);
    const data::MNISTDataset train_set(base + "/mnist_train.csv", 0, {1, 28, 28});
    const data::MNISTDataset test_set(base + "/mnist_test.csv", 0, {1, 28, 28});

    nn::Sequential model;
    std::vector<nn::Module*> linears;                                      // the three modules with weights
    nn::Module* first_conv = nullptr;
    {
        auto c1 = std::make_unique<nn::Conv2d>(1, 8, 3, 2, 1, 0.0f, 1);   // (N, 8, 14, 14)
        auto c2 = std::make_unique<nn::Conv2d>(8, 16, 3, 2, 1, 0.0f, 2);  // (N, 16, 7, 7)
        auto l3 = std::make_unique<nn::Linear>(16 * 7 * 7, 10, 0.0f, 3);
        linears = {c1.get(), c2.get(), l3.get()};
        first_conv = c1.get();
        model.add(std::move(c1));
        model.add(std::make_unique<nn::ReLU>());
        model.add(std::move(c2));
        model.add(std::make_unique<nn::ReLU>());
        model.add(std::make_unique<nn::Flatten>());
        model.add(std::move(l3));
    }
    auto params = model.parameters();
    optim::AdamW opt(params, LR);
    nn::CrossEntropyLoss loss_fn;
    data::DataLoader train(train_set, BATCH, /*shuffle=*/true, /*seed=*/11);

    J epochs = J::array();
    {
        const Eval e = evaluate(model, test_set);
        J row = J::object();
        row.set("epoch", 0).set("test_correct", e.correct).set("per_class_correct", trace::jarr(e.per_correct));
        epochs.push(std::move(row));
    }
    int steps = 0;
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();
        train.reset();
        double loss_sum = 0.0;
        int batches = 0;
        while (train.has_next()) {
            auto [xb, yb] = train.next_batch();
            Variable x(xb, false), y(yb, false);
            Variable out = model(x);
            Variable loss = loss_fn(out, y);
            loss_sum += loss.data().data()[0];
            ++batches;
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        steps += batches;
        const Eval e = evaluate(model, test_set);
        J row = J::object();
        row.set("epoch", epoch).set("steps", steps).set("train_loss", static_cast<float>(loss_sum / batches))
            .set("test_correct", e.correct).set("per_class_correct", trace::jarr(e.per_correct));
        epochs.push(std::move(row));
        std::cerr << "epoch " << epoch << " loss " << loss_sum / batches << " test " << e.correct << " ("
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n";
    }
    const Eval final_eval = evaluate(model, test_set);

    // test images, their pixels (0..255), labels and TorC's logits: the first FIRST, then the first WRONG that the model gets wrong
    std::vector<size_t> picked;
    for (int i = 0; i < FIRST; ++i) picked.push_back(static_cast<size_t>(i));
    for (int k = 0, n = 0; k < static_cast<int>(final_eval.wrong.size()) && n < WRONG; ++k)
        if (final_eval.wrong[k] >= FIRST) { picked.push_back(static_cast<size_t>(final_eval.wrong[k])); ++n; }
    J shown = J::array();
    {
        Variable::NoGradGuard ng;
        auto [xb, yb] = test_set.get_indices(picked);
        const Variable x(xb, false);
        Variable out = model(x);
        for (size_t i = 0; i < picked.size(); ++i) {
            std::vector<int> px(784);
            for (int k = 0; k < 784; ++k) px[k] = static_cast<int>(std::lround(xb.data()[i * 784 + k] * 255.0f));
            std::vector<float> logits(out.data().data() + i * 10, out.data().data() + i * 10 + 10);
            J s = J::object();
            s.set("index", static_cast<int>(picked[i])).set("label", static_cast<int>(yb.data()[i])).set("pixels", trace::jarr(px))
                .set("logits", trace::jarr(logits));
            shown.push(std::move(s));
        }
    }
    // the first convolution after relu, for the first shown image: (1, 8, 14, 14)
    J first_maps = J::object();
    {
        Variable::NoGradGuard ng;
        auto [xb, yb] = test_set.get_batch(0, 1);
        const Variable x(xb, false);
        Variable c = (*first_conv)(x);
        Variable r = relu(c);
        first_maps.set("index", 0).set("shape", trace::jarr(r.data().shape())).set("values", trace::jarr(trace::values_of(r.data())));
    }

    J layers = J::array();
    for (nn::Module* l : linears) layers.push(layer_json(*l));

    J setup = J::object();
    setup.set("train_samples", static_cast<int>(train_set.len())).set("test_samples", static_cast<int>(test_set.len()))
        .set("batch_size", BATCH).set("epochs", EPOCHS).set("lr", 0.001).set("weight_decay", 0.01).set("optimizer", "optim::AdamW")
        .set("loss", "nn::CrossEntropyLoss").set("image_side", 28).set("pixel_max", 255).set("parameters", count_params(model));
    J test_per_class = J::object();
    test_per_class.set("correct", trace::jarr(final_eval.per_correct)).set("total", trace::jarr(final_eval.per_total));

    J doc = trace::envelope("mnist-cnn", "mnist-cnn", "tools/trace/trace_mnist_cnn.cpp");
    doc.set("setup", std::move(setup)).set("epochs", std::move(epochs)).set("test_per_class", std::move(test_per_class))
        .set("layers", std::move(layers)).set("shown", std::move(shown)).set("first_maps", std::move(first_maps));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
