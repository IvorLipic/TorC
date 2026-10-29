// trace_mnist_mlp.cpp  ->  site/data/mnist-mlp.json   (lessons `mnist-mlp`, `one-step`)
//
// The model of examples/mnist_mlp/mnist_mlp.cpp, trained by TorC on MNIST: Sequential Linear(784, 32), ReLU, Linear(32, 32),
// ReLU, Linear(32, 10); nn::CrossEntropyLoss; optim::AdamW (lr 5e-4, default weight decay 0.01); batches of 128, shuffled.
// The seeds of the three Linear layers and of the DataLoader are fixed, so the run is the same each time (with libstdc++, whose
// normal_distribution and std::shuffle the start weights and the batch order come from).
// Data: data::MNISTDataset on the CSV files that tools/trace/fetch_mnist.sh writes (raw pixels 0..255; the loader divides by 255).
// The directory comes from the environment variable MNIST_DIR (run.sh sets it).
// Recorded: after each epoch, the mean training loss of the epoch and the accuracy on the 10000 test images (overall and per
// class); at the end, all weights and biases (so the browser can run the same forward pass), and 20 test images (the first 16,
// and the first 4 that the model gets wrong) with their pixels, labels and the logits TorC computes for them (the browser port
// is tested against these).
//
// Added for v3 (all additive; the trained model, its seeds and settings are unchanged, so `layers` and every older key keep their
// values):
//   setup.pixels, setup.classes            784 (values per image, read from the dataset) and 10 (outputs of the last Linear)
//   settings                               the real training settings in one object (C9), plus those of the TorC example
//   epochs[].test_accuracy, .test_loss     test accuracy as a fraction (D9; correct / 10000, derived in double) and the mean
//                                          nn::CrossEntropyLoss on the test images
//   epochs[].train_subset                  {correct, total, accuracy, loss} of the same model on a fixed subset of its own
//                                          training images (the first 10000), measured like the test images: train vs. test
//   test_per_class.accuracy[10]            correct / total per digit (derived)
//   final                                  {test_correct, test_samples, test_accuracy, test_mistakes, train_subset}
//   confusion                              the 10 x 10 confusion matrix of the final model on all 10000 test images
//                                          (matrix[true][predicted]); its diagonal sums to final.test_correct
//   mistakes                               the first 24 test images (in test-set order) the final model gets wrong: index, true
//                                          label, predicted label and TorC's softmax probabilities. Their pixels are in
//                                          mnist-mistakes.json (trace_mnist_mistakes.cpp), so this file stays small.
//   validation_run                         C2: a second run of the same recipe (same seeds and settings) on the first 55000
//                                          training images, with the last 5000 held out as a validation set. Per epoch: the
//                                          training loss, accuracy and loss on the first 5000 of its training images and on the
//                                          5000 validation images. The main model is not changed by this: holding images out
//                                          of the main run would change its batch order and so its weights, which the site's
//                                          digit pad and tests use. Its test accuracy is measured once, at the end.
#include "trace_mnist.hpp"

#include "torc/optim.hpp"

#include <chrono>
#include <iostream>
#include <thread>

using namespace torc;
using trace::J;
namespace tmn = trace_mnist;

namespace {

constexpr int BATCH = 128, EPOCHS = 20, FIRST = 16, WRONG = 4;   // the example trains for 20 epochs
constexpr float LR = 0.0005f;
constexpr int TRAIN_SUBSET = 10000, MISTAKES = 24, HOLD_OUT = 5000, VAL_TRAIN_SUBSET = 5000;

J layer_json(nn::Module& layer) {
    J o = J::object();
    const Tensor& w = layer.named_parameters().at("weight").data();
    const Tensor& b = layer.named_parameters().at("bias").data();
    o.set("weight_shape", trace::jarr(w.shape())).set("weight", trace::jarr(trace::values_of(w))).set("bias", trace::jarr(trace::values_of(b)));
    return o;
}

// C2: the same recipe on the first 55000 training images; the last 5000 are the validation set.
J validation_run(const data::MNISTDataset& train_set, const data::MNISTDataset& test_set) {
    const size_t n = train_set.len(), cut = n - HOLD_OUT;
    const tmn::Range fit(train_set, 0, cut), val(train_set, cut, n), fit_subset(train_set, 0, VAL_TRAIN_SUBSET);
    auto m = tmn::make_mlp(1, 2, 3);
    auto params = m->seq.parameters();
    optim::AdamW opt(params, LR);
    nn::CrossEntropyLoss loss_fn;
    data::DataLoader train(fit, BATCH, /*shuffle=*/true, /*seed=*/11);
    J epochs = J::array();
    auto point = [&](int epoch, int steps, double loss) {
        J row = J::object();
        row.set("epoch", epoch);
        if (epoch > 0) row.set("steps", steps).set("train_loss", static_cast<float>(loss));
        row.set("train_subset", tmn::eval_point(tmn::evaluate(m->seq, fit_subset))).set("validation", tmn::eval_point(tmn::evaluate(m->seq, val)));
        epochs.push(std::move(row));
    };
    point(0, 0, 0.0);
    int steps = 0;
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        train.reset();
        double loss_sum = 0.0;
        int batches = 0;
        while (train.has_next()) {
            auto [xb, yb] = train.next_batch();
            Variable x(xb, false), y(yb, false);
            Variable out = m->seq(x);
            Variable loss = loss_fn(out, y);
            loss_sum += loss.data().data()[0];
            ++batches;
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        steps += batches;
        point(epoch, steps, loss_sum / batches);
    }
    const tmn::Eval t = tmn::evaluate(m->seq, test_set);
    J o = J::object();
    o.set("note", "Same model, seeds and settings as the main run, trained on the first train_samples images only. The last "
                  "validation_samples training images are held out and never trained on. The test set is measured once, after training.")
        .set("train_samples", static_cast<int>(cut)).set("validation_samples", HOLD_OUT)
        .set("validation_first_index", static_cast<int>(cut)).set("train_subset_samples", VAL_TRAIN_SUBSET)
        .set("steps_per_epoch", steps / EPOCHS).set("epochs", std::move(epochs)).set("final_test", tmn::eval_point(t));
    return o;
}

}  // namespace

int main() {
    const std::string base = tmn::mnist_dir("trace_mnist_mlp");
    const data::MNISTDataset train_set(base + "/mnist_train.csv");
    const data::MNISTDataset test_set(base + "/mnist_test.csv");
    const tmn::Range train_subset(train_set, 0, TRAIN_SUBSET);

    // the held-out run is independent of the main run (own model, optimizer and loader), so it runs on a second thread
    J val_json;
    std::thread val_thread([&] { val_json = validation_run(train_set, test_set); });

    nn::Sequential model;
    std::vector<nn::Module*> linears;
    {
        auto l1 = std::make_unique<nn::Linear>(784, 32, 0.0f, 1);
        auto l2 = std::make_unique<nn::Linear>(32, 32, 0.0f, 2);
        auto l3 = std::make_unique<nn::Linear>(32, 10, 0.0f, 3);
        linears = {l1.get(), l2.get(), l3.get()};
        model.add(std::move(l1));
        model.add(std::make_unique<nn::ReLU>());
        model.add(std::move(l2));
        model.add(std::make_unique<nn::ReLU>());
        model.add(std::move(l3));
    }
    auto params = model.parameters();
    optim::AdamW opt(params, LR);
    nn::CrossEntropyLoss loss_fn;
    data::DataLoader train(train_set, BATCH, /*shuffle=*/true, /*seed=*/11);

    auto add_v3 = [&](J& row, const tmn::Eval& e) {   // the keys added for v3, after the older ones
        row.set("test_accuracy", e.accuracy()).set("test_loss", static_cast<float>(e.loss))
            .set("train_subset", tmn::eval_point(tmn::evaluate(model, train_subset)));
    };
    J epochs = J::array();
    {
        const tmn::Eval e = tmn::evaluate(model, test_set);
        J row = J::object();
        row.set("epoch", 0).set("test_correct", e.correct).set("per_class_correct", trace::jarr(e.per_correct));
        add_v3(row, e);
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
        const tmn::Eval e = tmn::evaluate(model, test_set);
        J row = J::object();
        row.set("epoch", epoch).set("steps", steps).set("train_loss", static_cast<float>(loss_sum / batches))
            .set("test_correct", e.correct).set("per_class_correct", trace::jarr(e.per_correct));
        add_v3(row, e);
        epochs.push(std::move(row));
        std::cerr << "epoch " << epoch << " loss " << loss_sum / batches << " test " << e.correct << " ("
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n";
    }
    const tmn::Eval final_eval = tmn::evaluate(model, test_set);
    const tmn::Eval final_train = tmn::evaluate(model, train_subset);

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

    // the first MISTAKES test images the final model gets wrong, with TorC's probabilities (softmax over the ten scores)
    J mistakes = J::object();
    {
        Variable::NoGradGuard ng;
        std::vector<size_t> idx;
        for (int k = 0; k < static_cast<int>(final_eval.wrong.size()) && k < MISTAKES; ++k) idx.push_back(static_cast<size_t>(final_eval.wrong[k]));
        auto [xb, yb] = test_set.get_indices(idx);
        const Variable x(xb, false);
        Variable out = model(x);
        Variable prob = torc::softmax(out, 1);
        J items = J::array();
        for (size_t i = 0; i < idx.size(); ++i) {
            const int pred = tmn::argmax10(out.data().data() + i * 10);
            if (pred != final_eval.predicted[idx[i]] || pred == static_cast<int>(yb.data()[i])) throw std::runtime_error("mistakes: prediction differs");
            std::vector<float> p(prob.data().data() + i * 10, prob.data().data() + i * 10 + 10);
            J s = J::object();
            s.set("index", static_cast<int>(idx[i])).set("label", static_cast<int>(yb.data()[i])).set("predicted", pred).set("probabilities", trace::jarr(p));
            items.push(std::move(s));
        }
        mistakes.set("rule", "the first wrong test images in test-set order").set("total_wrong", static_cast<int>(final_eval.wrong.size()))
            .set("count", static_cast<int>(idx.size())).set("pixels_in", "mnist-mistakes.json").set("items", std::move(items));
    }

    val_thread.join();

    J layers = J::array();
    for (nn::Module* l : linears) layers.push(layer_json(*l));

    const int pixels = train_set.get(0).first.numel();
    const int classes = linears.back()->named_parameters().at("weight").data().shape()[0];
    J setup = J::object();
    setup.set("train_samples", static_cast<int>(train_set.len())).set("test_samples", static_cast<int>(test_set.len()))
        .set("batch_size", BATCH).set("epochs", EPOCHS).set("lr", 0.0005).set("weight_decay", 0.01).set("optimizer", "optim::AdamW")
        .set("loss", "nn::CrossEntropyLoss").set("image_side", 28).set("pixel_max", 255).set("parameters", tmn::count_params(model))
        .set("pixels", pixels).set("classes", classes);
    J test_per_class = J::object();
    test_per_class.set("correct", trace::jarr(final_eval.per_correct)).set("total", trace::jarr(final_eval.per_total))
        .set("accuracy", trace::jarr(tmn::per_class_accuracy(final_eval)));

    J example = J::object();   // examples/mnist_mlp/mnist_mlp.cpp at the pinned ref (batch_size, epochs, lr there)
    example.set("source", "examples/mnist_mlp/mnist_mlp.cpp").set("epochs", 20).set("lr", 0.0005).set("batch_size", 128);
    J settings = J::object();
    settings.set("model", "Linear(784, 32), ReLU, Linear(32, 32), ReLU, Linear(32, 10)").set("parameters", tmn::count_params(model))
        .set("optimizer", "optim::AdamW").set("lr", 0.0005).set("weight_decay", 0.01).set("batch_size", BATCH).set("epochs", EPOCHS)
        .set("steps", steps).set("steps_per_epoch", steps / EPOCHS).set("loss", "nn::CrossEntropyLoss").set("shuffle", true)
        .set("loader_seed", 11).set("init", "nn::Linear default (Kaiming normal)").set("init_seeds", trace::jarr(std::vector<int>{1, 2, 3}))
        .set("train_samples", static_cast<int>(train_set.len())).set("example", std::move(example));
    J fin = J::object();
    fin.set("test_correct", final_eval.correct).set("test_samples", final_eval.total).set("test_accuracy", final_eval.accuracy())
        .set("test_mistakes", final_eval.total - final_eval.correct).set("test_loss", static_cast<float>(final_eval.loss))
        .set("train_subset", tmn::eval_point(final_train));

    J doc = trace::envelope("mnist-mlp", "mnist-mlp", "tools/trace/trace_mnist_mlp.cpp");
    doc.set("setup", std::move(setup)).set("epochs", std::move(epochs)).set("test_per_class", std::move(test_per_class))
        .set("layers", std::move(layers)).set("shown", std::move(shown))
        .set("settings", std::move(settings)).set("final", std::move(fin)).set("train_subset_samples", TRAIN_SUBSET)
        .set("confusion", tmn::confusion_json(final_eval)).set("mistakes", std::move(mistakes)).set("validation_run", std::move(val_json));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
