// trace_mnist_mistakes.cpp  ->  site/data/mnist-mistakes.json   (lesson `mnist-mlp`, the grid of mistakes)
//
// The pixels of the test images listed in mnist-mlp.json `mistakes.items` (the first 24 test images, in test-set order, that the
// final MLP classifies wrongly), kept in their own file so that mnist-mlp.json, which several lessons load, stays small.
// The model is the final MLP of trace_mnist_mlp.cpp, its weights read back from mnist-mlp.json (exact float32; a self-check
// confirms that it reproduces the logits recorded there bit for bit). For each image TorC's forward pass is run again, and a
// self-check confirms the predicted digit and the probabilities that mnist-mlp.json records.
// Keys: source (the file and key the list comes from); total_wrong (of the 10000 test images); items[] = {index, label,
// predicted, probabilities[10] (torc softmax of the scores), pixels[784] (0..255, as in mnist-mlp.json `shown`)}; checks.
// MNIST comes from MNIST_DIR (tools/trace/run.sh sets it).
#include "trace_mnist.hpp"

#include <iostream>

using namespace torc;
using trace::J;
namespace tmn = trace_mnist;

int main() {
    const std::string dir = tmn::mnist_dir("trace_mnist_mistakes");
    const tmn::JV doc = tmn::read_trace("mnist-mlp.json");
    const data::MNISTDataset test_set(dir + "/mnist_test.csv");
    auto m = tmn::make_mlp(1, 2, 3);
    tmn::load_layers(*m, doc["layers"]);
    const bool reproduced = tmn::reproduces_shown(*m, doc, test_set);

    const tmn::JV& list = doc["mistakes"]["items"];
    std::vector<size_t> idx;
    for (const auto& it : list.a) idx.push_back(static_cast<size_t>(it["index"].i()));
    Variable::NoGradGuard ng;
    auto [xb, yb] = test_set.get_indices(idx);
    Variable out = m->seq(Variable(xb, false));
    Variable prob = torc::softmax(out, 1);
    J items = J::array();
    bool same = true;
    for (size_t i = 0; i < idx.size(); ++i) {
        const int label = static_cast<int>(yb.data()[i]), pred = tmn::argmax10(out.data().data() + i * 10);
        std::vector<float> p(prob.data().data() + i * 10, prob.data().data() + i * 10 + 10);
        same = same && label == list[i]["label"].i() && pred == list[i]["predicted"].i() && p == list[i]["probabilities"].floats() && pred != label;
        J s = J::object();
        s.set("index", static_cast<int>(idx[i])).set("label", label).set("predicted", pred).set("probabilities", trace::jarr(p))
            .set("pixels", trace::jarr(tmn::pixels255(xb, static_cast<int>(i))));
        items.push(std::move(s));
    }
    if (!reproduced || !same) { std::cerr << "trace_mnist_mistakes: a self-check failed\n"; return 1; }
    J checks = J::object();
    checks.set("weights_reproduce_mnist_mlp_logits", reproduced).set("predictions_match_mnist_mlp", same);
    J o = trace::envelope("mnist-mistakes", "mnist-mlp", "tools/trace/trace_mnist_mistakes.cpp");
    o.set("source", "mnist-mlp.json mistakes.items").set("rule", doc["mistakes"]["rule"].str())
        .set("total_wrong", doc["mistakes"]["total_wrong"].i()).set("items", std::move(items)).set("checks", std::move(checks));
    std::fputs(o.dump().c_str(), stdout);
    return 0;
}
