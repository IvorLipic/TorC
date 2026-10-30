// trace_mnist.hpp
// Shared helpers for the MNIST trace programs (trace_mnist_mlp.cpp, trace_mnist_cnn.cpp, trace_one_step.cpp,
// trace_mnist_mistakes.cpp, trace_shift.cpp, trace_mnist_seeds.cpp). Nothing here re-implements library math: the models are
// real torc::nn modules, every forward pass, loss, softmax, gradient and optimizer step is TorC's. The helpers only
//   - build the two example models (the MLP of examples/mnist_mlp, the CNN of examples/mnist_cnn) with given seeds,
//   - evaluate a model on a dataset (counts, a 10 x 10 confusion matrix, the mean nn::CrossEntropyLoss),
//   - read a trace JSON file back (the trained weights are stored with 9 significant digits, so std::strtof gives back the
//     exact float32 that TorC trained) and load those weights into a model,
//   - move images by whole pixels with zero fill (input preparation, done by the producer, not by the library),
//   - give a contiguous range of a dataset as a dataset (a view; DataLoader then shuffles indices 0..n-1 of the range).
#pragma once

#include "trace_util.hpp"

#include "torc/data.hpp"
#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/conv.hpp"
#include "torc/nn/linear.hpp"
#include "torc/nn/losses.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace trace_mnist {

using namespace torc;

constexpr int CLASSES = 10, SIDE = 28;

// ---- the two example models ---------------------------------------------------------------------------------------------
struct Model {
    nn::Sequential seq;
    std::vector<nn::Module*> weighted;   // the modules with weights, in order (3 for both models)
    nn::Module* first_conv = nullptr;    // CNN only
};

// examples/mnist_mlp: Linear(784, 32), ReLU, Linear(32, 32), ReLU, Linear(32, 10). Default (Kaiming normal) start weights.
inline std::unique_ptr<Model> make_mlp(unsigned s1, unsigned s2, unsigned s3) {
    auto m = std::make_unique<Model>();
    auto l1 = std::make_unique<nn::Linear>(784, 32, 0.0f, s1);
    auto l2 = std::make_unique<nn::Linear>(32, 32, 0.0f, s2);
    auto l3 = std::make_unique<nn::Linear>(32, 10, 0.0f, s3);
    m->weighted = {l1.get(), l2.get(), l3.get()};
    m->seq.add(std::move(l1));
    m->seq.add(std::make_unique<nn::ReLU>());
    m->seq.add(std::move(l2));
    m->seq.add(std::make_unique<nn::ReLU>());
    m->seq.add(std::move(l3));
    return m;
}

// examples/mnist_cnn: Conv2d(1, 8, 3, stride 2, padding 1), ReLU, Conv2d(8, 16, 3, 2, 1), ReLU, Flatten, Linear(784, 10).
inline std::unique_ptr<Model> make_cnn(unsigned s1, unsigned s2, unsigned s3) {
    auto m = std::make_unique<Model>();
    auto c1 = std::make_unique<nn::Conv2d>(1, 8, 3, 2, 1, 0.0f, s1);    // (N, 8, 14, 14)
    auto c2 = std::make_unique<nn::Conv2d>(8, 16, 3, 2, 1, 0.0f, s2);   // (N, 16, 7, 7)
    auto l3 = std::make_unique<nn::Linear>(16 * 7 * 7, 10, 0.0f, s3);
    m->weighted = {c1.get(), c2.get(), l3.get()};
    m->first_conv = c1.get();
    m->seq.add(std::move(c1));
    m->seq.add(std::make_unique<nn::ReLU>());
    m->seq.add(std::move(c2));
    m->seq.add(std::make_unique<nn::ReLU>());
    m->seq.add(std::make_unique<nn::Flatten>());
    m->seq.add(std::move(l3));
    return m;
}

inline int count_params(const nn::Module& m) {
    int n = 0;
    for (Variable* p : m.parameters()) n += p->data().numel();
    return n;
}

inline int argmax10(const float* row) {
    int best = 0;
    for (int j = 1; j < CLASSES; ++j) if (row[j] > row[best]) best = j;
    return best;
}

// ---- evaluation ---------------------------------------------------------------------------------------------------------------
struct Eval {
    int correct = 0, total = 0;
    std::vector<int> per_correct = std::vector<int>(CLASSES, 0), per_total = std::vector<int>(CLASSES, 0);
    std::vector<int> wrong;                                       // dataset positions the model gets wrong, in order
    std::vector<int> predicted;                                   // the predicted class of every sample, in order
    std::array<std::array<int, CLASSES>, CLASSES> confusion{};    // [true digit][predicted digit]
    double loss = 0.0;                                            // mean nn::CrossEntropyLoss over the samples
    [[nodiscard]] double accuracy() const { return total ? static_cast<double>(correct) / total : 0.0; }
};

// Batches of 1000 in order (DataLoader without shuffling), no gradient. The loss of a batch is TorC's mean over the batch; the
// mean over the dataset weights each batch by its size (in double).
inline Eval evaluate(const nn::Module& model, const data::Dataset& ds) {
    Variable::NoGradGuard ng;
    data::DataLoader loader(ds, 1000, false, 0);
    nn::CrossEntropyLoss loss_fn;
    Eval e;
    double loss_sum = 0.0;
    while (loader.has_next()) {
        auto [xb, yb] = loader.next_batch();
        const Variable x(xb, false), y(yb, false);
        Variable out = model(x);
        const int n = xb.shape()[0];
        loss_sum += static_cast<double>(loss_fn(out, y).data().data()[0]) * n;
        for (int i = 0; i < n; ++i) {
            const int pred = argmax10(out.data().data() + i * CLASSES), label = static_cast<int>(yb.data()[i]);
            e.per_total[label] += 1;
            e.confusion[label][pred] += 1;
            e.predicted.push_back(pred);
            if (pred == label) { e.correct += 1; e.per_correct[label] += 1; }
            else e.wrong.push_back(e.total);
            ++e.total;
        }
    }
    e.loss = e.total ? loss_sum / e.total : 0.0;
    return e;
}

// {correct, total, accuracy, loss}: what a curve point needs (accuracy = correct / total, derived in double)
inline trace::J eval_point(const Eval& e) {
    trace::J o = trace::J::object();
    o.set("correct", e.correct).set("total", e.total).set("accuracy", e.accuracy()).set("loss", static_cast<float>(e.loss));
    return o;
}

inline std::vector<double> per_class_accuracy(const Eval& e) {
    std::vector<double> a(CLASSES);
    for (int k = 0; k < CLASSES; ++k) a[k] = e.per_total[k] ? static_cast<double>(e.per_correct[k]) / e.per_total[k] : 0.0;
    return a;
}

inline trace::J confusion_json(const Eval& e) {
    trace::J rows = trace::J::array();
    int diag = 0, sum = 0;
    for (int t = 0; t < CLASSES; ++t) {
        rows.push(trace::jarr(std::vector<int>(e.confusion[t].begin(), e.confusion[t].end())));
        diag += e.confusion[t][t];
        for (int p = 0; p < CLASSES; ++p) sum += e.confusion[t][p];
    }
    if (diag != e.correct || sum != e.total) throw std::runtime_error("confusion matrix does not add up");
    trace::J o = trace::J::object();
    o.set("rows", "true digit").set("columns", "predicted digit").set("matrix", std::move(rows)).set("total", sum)
        .set("correct", diag).set("accuracy", e.accuracy());
    return o;
}

// ---- a contiguous range of a dataset, as a dataset ---------------------------------------------------------------------------
struct Range : data::Dataset {
    const data::Dataset& d;
    size_t lo, n;
    Range(const data::Dataset& base, size_t from, size_t to) : d(base), lo(from), n(to - from) {}
    [[nodiscard]] size_t len() const override { return n; }
    [[nodiscard]] std::pair<Tensor, Tensor> get(size_t i) const override { return d.get(lo + i); }
    [[nodiscard]] std::pair<Tensor, Tensor> gather(const std::vector<size_t>& idx) const override {
        std::vector<size_t> j(idx);
        for (auto& v : j) v += lo;
        return d.gather(j);
    }
};

// ---- shifting images ------------------------------------------------------------------------------------------------------------
// x holds N images of 28 x 28 values (shape (N, 784) or (N, 1, 28, 28)). Returns a copy moved by dx pixels to the right
// (negative: left) and dy pixels down (negative: up). Pixels that leave the image are dropped; the uncovered border is 0.
inline Tensor shift_images(const Tensor& x, int dx, int dy) {
    Tensor out(x.shape());
    const int n = x.shape()[0], hw = SIDE * SIDE;
    if (x.numel() != n * hw) throw std::runtime_error("shift_images: expected 28 x 28 images");
    for (int s = 0; s < n; ++s)
        for (int r = 0; r < SIDE; ++r)
            for (int c = 0; c < SIDE; ++c) {
                const int r0 = r - dy, c0 = c - dx;
                out.data()[s * hw + r * SIDE + c] = (r0 >= 0 && r0 < SIDE && c0 >= 0 && c0 < SIDE) ? x.data()[s * hw + r0 * SIDE + c0] : 0.0f;
            }
    return out;
}

// ---- reading a trace JSON file back ----------------------------------------------------------------------------------------------
struct JV {
    enum class K { Null, Bool, Num, Str, Arr, Obj } k = K::Null;
    std::string s;                                  // Num: the literal text, Str: the string, Bool: "true"/"false"
    std::vector<JV> a;                              // Arr: the items; Obj: the values, in the order of `keys`
    std::vector<std::string> keys;                  // Obj: the keys

    [[nodiscard]] const JV& operator[](const std::string& key) const {
        for (size_t j = 0; j < keys.size(); ++j) if (keys[j] == key) return a[j];
        throw std::runtime_error("trace json: no key '" + key + "'");
    }
    [[nodiscard]] const JV& operator[](size_t i) const {
        if (k != K::Arr || i >= a.size()) throw std::runtime_error("trace json: index out of range");
        return a[i];
    }
    [[nodiscard]] const JV& back() const { return (*this)[a.size() - 1]; }
    [[nodiscard]] size_t size() const { return a.size(); }
    [[nodiscard]] float f() const { return std::strtof(num().c_str(), nullptr); }   // exact for "%.9g" float32 literals
    [[nodiscard]] double d() const { return std::strtod(num().c_str(), nullptr); }
    [[nodiscard]] int i() const { return static_cast<int>(std::strtol(num().c_str(), nullptr, 10)); }
    [[nodiscard]] const std::string& str() const { if (k != K::Str) throw std::runtime_error("trace json: not a string"); return s; }
    [[nodiscard]] std::vector<float> floats() const { std::vector<float> v; for (const auto& x : a) v.push_back(x.f()); return v; }
    [[nodiscard]] std::vector<int> ints() const { std::vector<int> v; for (const auto& x : a) v.push_back(x.i()); return v; }

private:
    [[nodiscard]] const std::string& num() const { if (k != K::Num) throw std::runtime_error("trace json: not a number"); return s; }
};

class JReader {
public:
    explicit JReader(std::string t) : t_(std::move(t)) {}
    JV parse() { JV v = value(); ws(); if (p_ != t_.size()) fail("trailing text"); return v; }

private:
    std::string t_;
    size_t p_ = 0;
    [[noreturn]] void fail(const char* m) const { throw std::runtime_error(std::string("trace json: ") + m + " at " + std::to_string(p_)); }
    void ws() { while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\t' || t_[p_] == '\r')) ++p_; }
    char peek() { ws(); if (p_ >= t_.size()) fail("unexpected end"); return t_[p_]; }
    void expect(char c) { if (peek() != c) fail("unexpected character"); ++p_; }
    std::string string_lit() {
        expect('"');
        std::string out;
        while (p_ < t_.size() && t_[p_] != '"') {
            char c = t_[p_++];
            if (c == '\\') {
                if (p_ >= t_.size()) fail("bad escape");
                const char e = t_[p_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'u': out += '?'; p_ += 4; break;   // not used by the trace writer for printable text
                    default: out += e;
                }
            } else out += c;
        }
        expect('"');
        return out;
    }
    JV value() {
        JV v;
        const char c = peek();
        if (c == '{') {
            v.k = JV::K::Obj; ++p_;
            if (peek() == '}') { ++p_; return v; }
            for (;;) {
                std::string key = string_lit();
                expect(':');
                v.keys.push_back(std::move(key));
                v.a.push_back(value());
                if (peek() == ',') { ++p_; continue; }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            v.k = JV::K::Arr; ++p_;
            if (peek() == ']') { ++p_; return v; }
            for (;;) {
                v.a.push_back(value());
                if (peek() == ',') { ++p_; continue; }
                expect(']');
                return v;
            }
        }
        if (c == '"') { v.k = JV::K::Str; v.s = string_lit(); return v; }
        if (t_.compare(p_, 4, "true") == 0) { v.k = JV::K::Bool; v.s = "true"; p_ += 4; return v; }
        if (t_.compare(p_, 5, "false") == 0) { v.k = JV::K::Bool; v.s = "false"; p_ += 5; return v; }
        if (t_.compare(p_, 4, "null") == 0) { p_ += 4; return v; }
        const size_t s = p_;
        while (p_ < t_.size() && (std::isdigit(static_cast<unsigned char>(t_[p_])) || t_[p_] == '-' || t_[p_] == '+' || t_[p_] == '.' || t_[p_] == 'e' || t_[p_] == 'E')) ++p_;
        if (s == p_) fail("bad value");
        v.k = JV::K::Num; v.s = t_.substr(s, p_ - s);
        return v;
    }
};

// The directory the other trace files are read from: $TRACE_DATA_DIR, else site/data of this checkout. tools/trace/run.sh lists
// the MNIST traces first, so a full run writes mnist-mlp.json and mnist-cnn.json before the programs that read them; under
// run.sh --check the committed files are read, so these outputs are always computed from the committed trained weights.
inline std::string data_dir() {
    if (const char* d = std::getenv("TRACE_DATA_DIR")) return d;
    return (std::filesystem::path(__FILE__).parent_path() / ".." / ".." / "site" / "data").lexically_normal().string();
}

inline JV read_trace(const std::string& file) {
    const std::string path = data_dir() + "/" + file;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path + " (run tools/trace/run.sh, which writes it first)");
    std::stringstream ss;
    ss << in.rdbuf();
    return JReader(ss.str()).parse();
}

// Copies the recorded weights and biases (layers[] of mnist-mlp.json / mnist-cnn.json) into the model's parameters.
inline void load_layers(Model& m, const JV& layers) {
    if (layers.size() != m.weighted.size()) throw std::runtime_error("load_layers: layer count differs");
    for (size_t i = 0; i < m.weighted.size(); ++i) {
        Tensor& w = m.weighted[i]->named_parameters().at("weight").data();
        Tensor& b = m.weighted[i]->named_parameters().at("bias").data();
        const auto wv = layers[i]["weight"].floats(), bv = layers[i]["bias"].floats();
        if (layers[i]["weight_shape"].ints() != w.shape() || static_cast<int>(wv.size()) != w.numel() || static_cast<int>(bv.size()) != b.numel())
            throw std::runtime_error("load_layers: shape differs");
        for (int k = 0; k < w.numel(); ++k) w.data()[k] = wv[k];
        for (int k = 0; k < b.numel(); ++k) b.data()[k] = bv[k];
    }
}

// Self-check after load_layers: the model gives bit for bit the logits the training trace recorded for its shown images.
inline bool reproduces_shown(const Model& m, const JV& doc, const data::Dataset& test_set) {
    Variable::NoGradGuard ng;
    std::vector<size_t> idx;
    for (const auto& s : doc["shown"].a) idx.push_back(static_cast<size_t>(s["index"].i()));
    auto [xb, yb] = test_set.get_indices(idx);
    const Variable x(xb, false);
    Variable out = m.seq(x);
    for (size_t i = 0; i < idx.size(); ++i) {
        const auto logits = doc["shown"][i]["logits"].floats();
        for (int j = 0; j < CLASSES; ++j)
            if (out.data().data()[i * CLASSES + j] != logits[j]) return false;
    }
    return true;
}

inline std::vector<int> pixels255(const Tensor& x, int sample) {   // back to the CSV's integers 0..255
    std::vector<int> px(SIDE * SIDE);
    for (int k = 0; k < SIDE * SIDE; ++k) px[k] = static_cast<int>(std::lround(x.data()[sample * SIDE * SIDE + k] * 255.0f));
    return px;
}

inline std::string mnist_dir(const char* prog) {
    const char* dir = std::getenv("MNIST_DIR");
    if (!dir) throw std::runtime_error(std::string(prog) + ": set MNIST_DIR (tools/trace/run.sh does)");
    return dir;
}

}  // namespace trace_mnist
