// trace_conv.cpp  ->  site/data/conv.json   (lesson `convolution`)
//
// 1. torc::conv2d on a 5 x 5 input with a vertical edge (columns 0, 0, 4, 4, 4, and one brighter pixel) and a 3 x 3 kernel set by hand, for stride 1 / padding 0, stride 1 / padding 1
//    and stride 2 / padding 1: the input, the kernel and each output.
// 2. Two edge kernels (set by hand) on the first MNIST test image (a 7), with stride 1 and padding 1: the image (pixels 0..1, as
//    data::MNISTDataset gives them, read with sample_shape {1, 28, 28}) and both feature maps, before and after relu.
// 3. The layers of examples/mnist_cnn/mnist_cnn.cpp: Conv2d(1, 8, 3, stride 2, padding 1), ReLU, Conv2d(8, 16, 3, 2, 1), ReLU,
//    Flatten, Linear(784, 10), run on a batch of shape (2, 1, 28, 28): each module's output shape and parameter count, and a
//    Linear layer from the same image to 8 outputs for comparison.
// 4. The errors conv2d throws for a channel mismatch and for a kernel larger than the padded input.
// Everything is library output, except the kernel, stride and padding of each convolution in "cnn" (the producer writes the
// arguments it passes to nn::Conv2d) and the order of "params" (the order of an unordered_map). MNIST comes from MNIST_DIR (tools/trace/run.sh sets it; tools/trace/fetch_mnist.sh fills it).
#include "trace_util.hpp"

#include "torc/data.hpp"
#include "torc/nn.hpp"
#include "torc/nn/activations.hpp"
#include "torc/nn/conv.hpp"
#include "torc/nn/linear.hpp"

#include <cstdlib>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

int count_params(nn::Module& m) {
    int n = 0;
    for (Variable* p : m.parameters()) n += p->data().numel();
    return n;
}

}  // namespace

int main() {
    // ---- 1. a small example -------------------------------------------------------------------------------------------------------
    std::vector<float> x25(25);
    for (int i = 0; i < 25; ++i) x25[i] = (i % 5 < 2) ? 0.0f : 4.0f;   // dark on the left, bright on the right
    x25[12] = 8.0f;                                                       // one brighter pixel in the middle
    const Tensor x = trace::make_tensor({1, 1, 5, 5}, x25);
    const Tensor k = trace::make_tensor({1, 1, 3, 3}, {1, 0, -1, 2, 0, -2, 1, 0, -1});   // a vertical-edge kernel
    J small = J::array();
    for (auto [stride, padding] : std::vector<std::pair<int, int>>{{1, 0}, {1, 1}, {2, 1}}) {
        const Tensor y = x.conv2d(k, stride, padding);
        J e = J::object();
        e.set("stride", stride).set("padding", padding).set("output", trace::tensor_json(y));
        small.push(std::move(e));
    }
    J example = J::object();
    example.set("input", trace::tensor_json(x)).set("kernel", trace::tensor_json(k)).set("runs", std::move(small));

    // ---- 2. edge kernels on a digit -------------------------------------------------------------------------------------------------
    const char* dir = std::getenv("MNIST_DIR");
    if (!dir) { std::cerr << "trace_conv: set MNIST_DIR (tools/trace/run.sh does)\n"; return 1; }
    const data::MNISTDataset test_set(std::string(dir) + "/mnist_test.csv", 1, {1, 28, 28});
    auto [img, label] = test_set.get_batch(0, 1);                        // shape (1, 1, 28, 28)
    struct K { const char* id; std::vector<float> w; };
    const std::vector<K> kernels = {{"vertical", {1, 0, -1, 2, 0, -2, 1, 0, -1}}, {"horizontal", {1, 2, 1, 0, 0, 0, -1, -2, -1}}};
    J maps = J::array();
    for (const auto& kk : kernels) {
        const Variable xv(img, false), kv(trace::make_tensor({1, 1, 3, 3}, kk.w), false);
        Variable y = conv2d(xv, kv, 1, 1);
        Variable r = relu(y);
        J e = J::object();
        e.set("id", kk.id).set("kernel", trace::jarr(kk.w)).set("shape", trace::jarr(y.data().shape()))
            .set("output", trace::jarr(trace::values_of(y.data()))).set("relu", trace::jarr(trace::values_of(r.data())));
        maps.push(std::move(e));
    }
    J digit = J::object();
    digit.set("label", static_cast<int>(label.data()[0])).set("shape", trace::jarr(img.shape())).set("pixels", trace::jarr(trace::values_of(img)))
        .set("maps", std::move(maps));

    // ---- 3. the CNN of the example --------------------------------------------------------------------------------------------------
    J cnn = J::array();
    int total = 0;
    {
        struct Cfg { int kernel = 0, stride = 0, padding = 0; };
        const std::vector<Cfg> cfg = {{3, 2, 1}, {}, {3, 2, 1}, {}, {}, {}};
        std::vector<std::pair<std::string, std::unique_ptr<nn::Module>>> mods;
        mods.emplace_back("Conv2d(1, 8, 3, stride 2, padding 1)", std::make_unique<nn::Conv2d>(1, 8, 3, 2, 1, 0.0f, 1));
        mods.emplace_back("ReLU", std::make_unique<nn::ReLU>());
        mods.emplace_back("Conv2d(8, 16, 3, stride 2, padding 1)", std::make_unique<nn::Conv2d>(8, 16, 3, 2, 1, 0.0f, 2));
        mods.emplace_back("ReLU", std::make_unique<nn::ReLU>());
        mods.emplace_back("Flatten", std::make_unique<nn::Flatten>());
        mods.emplace_back("Linear(784, 10)", std::make_unique<nn::Linear>(784, 10, 0.0f, 3));
        std::vector<Variable> outs;
        outs.reserve(mods.size() + 1);
        outs.emplace_back(Tensor(std::vector<int>{2, 1, 28, 28}), false);
        for (size_t mi = 0; mi < mods.size(); ++mi) {
            auto& [label_text, m] = mods[mi];
            const std::vector<int> in_shape = outs.back().data().shape();
            outs.push_back((*m)(outs.back()));
            const int n = count_params(*m);
            total += n;
            J e = J::object();
            J params = J::array();
            for (const auto& [name, p] : m->named_parameters()) {
                J q = J::object();
                q.set("name", name).set("shape", trace::jarr(p.data().shape()));
                params.push(std::move(q));
            }
            e.set("label", label_text).set("in_shape", trace::jarr(in_shape)).set("out_shape", trace::jarr(outs.back().data().shape()))
                .set("parameters", n).set("params", std::move(params));
            if (cfg[mi].kernel) e.set("kernel", cfg[mi].kernel).set("stride", cfg[mi].stride).set("padding", cfg[mi].padding);
            cnn.push(std::move(e));
        }
    }
    nn::Linear dense(784, 8, 0.0f, 4);
    J compare = J::object();
    compare.set("label", "Linear(784, 8)").set("parameters", count_params(dense));

    // ---- 4. errors ------------------------------------------------------------------------------------------------------------------
    J errors = J::array();
    auto err = [&](const char* id, const std::vector<int>& xs, const std::vector<int>& ws, int stride, int padding) {
        auto c = trace::catch_call([&] { (void)Tensor(xs).conv2d(Tensor(ws), stride, padding); });
        J e = J::object();
        e.set("id", id).set("input_shape", trace::jarr(xs)).set("weight_shape", trace::jarr(ws)).set("exception", c.type).set("what", c.what);
        errors.push(std::move(e));
    };
    err("channels", {1, 3, 28, 28}, {8, 1, 3, 3}, 1, 1);
    err("too-small", {1, 1, 2, 2}, {1, 1, 5, 5}, 1, 0);

    J doc = trace::envelope("conv", "convolution", "tools/trace/trace_conv.cpp");
    doc.set("example", std::move(example)).set("digit", std::move(digit)).set("cnn", std::move(cnn)).set("cnn_parameters", total)
        .set("dense_compare", std::move(compare)).set("errors", std::move(errors));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
