// trace_one_step.cpp  ->  site/data/one-step.json   (lesson `one-step`)
//
// One real training step of the trained MNIST MLP, followed end to end. The model is the final model of trace_mnist_mlp.cpp:
// its weights are read back from mnist-mlp.json (written with 9 significant digits, so they are the exact float32 values TorC
// trained; a self-check confirms that the loaded model gives bit for bit the logits recorded there for the 20 shown images).
// The step:
//   1. One training image (the rule is in `image.rule`): its pixels, the 32 + 32 hidden values before and after ReLU, the ten
//      scores (logits), the probabilities (torc::softmax over the scores) and the nn::CrossEntropyLoss, all from TorC's forward
//      pass, calling the five modules one at a time (a self-check confirms that the Sequential gives the same scores).
//   2. loss.backward(): the gradient of the loss at the scores, at the hidden values before ReLU, and at four named parameters.
//   3. optim::SGD(parameters, lr) with no momentum, one step(): each named parameter is read before and after. SGD without
//      momentum does p = p - lr * grad (src/optim.cpp), so "change = -lr * gradient" holds exactly; a self-check confirms
//      that each recorded `after` equals before - lr * grad computed in float32 the same way, bit for bit.
//   4. The same image again: the probabilities and the loss after the step.
// This step uses SGD and a batch of one image so that the arithmetic is visible. The real training run used optim::AdamW and
// batches of 128 (recorded in `full_run`, read from mnist-mlp.json), which changes each weight by a different rule.
//
// Keys: setup {lr, momentum, optimizer, batch_size, loss, weights_from}; full_run {steps, epochs, batch_size, train_samples,
// steps_per_epoch, optimizer, lr, weight_decay} (from mnist-mlp.json: steps = epochs[-1].steps, the number of step() calls of the
// real run); image {rule, train_index, label, pixels[784] 0..255}; forward {hidden1 {pre[32], post[32]}, hidden2 {pre, post},
// scores[10], probabilities[10], predicted, p_true, loss}; backward {scores_grad[10], hidden1_pre_grad[32], hidden2_pre_grad[32]};
// weights[] {id, what, layer (0..2, the Linear), param, row, col, before, grad, lr_times_grad, after, change, path{...},
// update_exact}; after {probabilities[10], predicted, p_true, loss}; derived {p_true_change, loss_change, loss_check}
// (plain arithmetic on recorded numbers); changed_parameters (how many of the 26506 numbers step() changed); checks.
#include "trace_mnist.hpp"

#include "torc/optim.hpp"

#include <cmath>
#include <iostream>

using namespace torc;
using trace::J;
namespace tmn = trace_mnist;

namespace {

constexpr double LR_CONFIG = 0.001;   // plain SGD, no momentum: change = -lr * gradient. Chosen so that one step on one image moves
                                      // the probability of the true digit visibly but not to 1 (lr 0.01 already gives 0.9999)
constexpr float LR = static_cast<float>(LR_CONFIG);
constexpr double P_LO = 0.5, P_HI = 0.9;

std::vector<float> vals(const Tensor& t) { return trace::values_of(t); }

}  // namespace

int main() {
    const std::string dir = tmn::mnist_dir("trace_one_step");
    const tmn::JV doc = tmn::read_trace("mnist-mlp.json");
    const data::MNISTDataset train_set(dir + "/mnist_train.csv");
    const data::MNISTDataset test_set(dir + "/mnist_test.csv");

    auto m = tmn::make_mlp(1, 2, 3);
    tmn::load_layers(*m, doc["layers"]);
    const bool reproduced = tmn::reproduces_shown(*m, doc, test_set);
    if (!reproduced) { std::cerr << "trace_one_step: the loaded weights do not reproduce mnist-mlp.json\n"; return 1; }

    // 1. the image: the first training image (in training-set order) that the model classifies correctly with a probability of
    //    the true digit in [P_LO, P_HI): right, but not sure, so one step changes the probability visibly
    int chosen = -1;
    {
        Variable::NoGradGuard ng;
        data::DataLoader loader(train_set, 1000, false, 0);
        int base = 0;
        while (loader.has_next() && chosen < 0) {
            auto [xb, yb] = loader.next_batch();
            Variable out = m->seq(Variable(xb, false));
            const Tensor p = out.data().softmax(1);
            for (int i = 0; i < xb.shape()[0]; ++i) {
                const int label = static_cast<int>(yb.data()[i]);
                const double pt = p.data()[i * 10 + label];
                if (tmn::argmax10(out.data().data() + i * 10) == label && pt >= P_LO && pt < P_HI) { chosen = base + i; break; }
            }
            base += xb.shape()[0];
        }
    }
    if (chosen < 0) { std::cerr << "trace_one_step: no image matches the rule\n"; return 1; }
    auto [xb, yb] = train_set.get_batch(static_cast<size_t>(chosen), static_cast<size_t>(chosen) + 1);   // (1, 784), (1, 1)
    const int label = static_cast<int>(yb.data()[0]);

    nn::Module& l1 = *m->weighted[0];
    nn::Module& l2 = *m->weighted[1];
    nn::Module& l3 = *m->weighted[2];
    const Variable x(xb, false), y(yb, false);
    Variable h1 = l1(x);
    Variable a1 = relu(h1);
    Variable h2 = l2(a1);
    Variable a2 = relu(h2);
    Variable z = l3(a2);
    nn::CrossEntropyLoss loss_fn;
    Variable loss = loss_fn(z, y);
    const Tensor prob = z.data().softmax(1);
    bool same_as_sequential = true;
    {
        Variable::NoGradGuard ng;
        // a second copy of the model: calling m->seq here would clear the modules' forward caches that the graph above needs
        auto m2 = tmn::make_mlp(1, 2, 3);
        tmn::load_layers(*m2, doc["layers"]);
        Variable zs = m2->seq(Variable(xb, false));
        same_as_sequential = trace::tensors_identical(zs.data(), z.data());
    }
    const float p_true = prob.data()[label];
    const int predicted = tmn::argmax10(z.data().data());

    // 2. backward
    loss.backward();
    const Tensor dz = z.grad(), dh1 = h1.grad(), dh2 = h2.grad();

    // the named parameters
    Variable& W1 = l1.named_parameters().at("weight");
    Variable& W3 = l3.named_parameters().at("weight");
    Variable& b3 = l3.named_parameters().at("bias");
    int pix = 0;                                     // the brightest pixel (first in row-major order)
    for (int k = 1; k < 784; ++k) if (xb.data()[k] > xb.data()[pix]) pix = k;
    int unit = 0;                                    // the first-layer unit whose weight from that pixel has the largest |gradient|
    for (int j = 1; j < 32; ++j) if (std::fabs(W1.grad().data()[j * 784 + pix]) > std::fabs(W1.grad().data()[unit * 784 + pix])) unit = j;
    int k2 = 0;                                      // the most active second-layer unit (after ReLU)
    for (int k = 1; k < 32; ++k) if (a2.data().data()[k] > a2.data().data()[k2]) k2 = k;
    int rival = label == 0 ? 1 : 0;                  // the most probable wrong digit
    for (int c = 0; c < 10; ++c) if (c != label && prob.data()[c] > prob.data()[rival]) rival = c;

    struct Named { std::string id, what; int layer; std::string param; Variable* v; int row, col, flat; J path; };
    std::vector<Named> named;
    {
        J p = J::object();
        const float xp = xb.data()[pix], g = dh1.data()[unit];
        p.set("pixel", pix).set("pixel_row", pix / 28).set("pixel_col", pix % 28).set("pixel_value", xp)
            .set("pixel_255", static_cast<int>(std::lround(xp * 255.0f))).set("unit", unit).set("unit_pre", h1.data().data()[unit])
            .set("unit_post", a1.data().data()[unit]).set("unit_grad", g).set("product", xp * g)
            .set("grad_equals_product", W1.grad().data()[unit * 784 + pix] == xp * g);
        named.push_back({"first_layer", "the first-layer weight from the brightest pixel to the hidden unit where it matters most",
                         0, "weight", &W1, unit, pix, unit * 784 + pix, std::move(p)});
    }
    for (const auto& [id, c] : std::vector<std::pair<std::string, int>>{{"output_true", label}, {"output_rival", rival}}) {
        J p = J::object();
        const float a = a2.data().data()[k2], g = dz.data()[c];
        p.set("unit", k2).set("unit_post", a).set("class", c).set("score_grad", g).set("product", a * g)
            .set("grad_equals_product", W3.grad().data()[c * 32 + k2] == a * g);
        named.push_back({id, id == "output_true" ? "the output weight from the most active second-layer unit to the true digit's score"
                                                 : "the output weight from the same unit to the score of the most probable wrong digit",
                         2, "weight", &W3, c, k2, c * 32 + k2, std::move(p)});
    }
    {
        J p = J::object();
        p.set("class", label).set("score_grad", dz.data()[label]).set("p_true_minus_one", p_true - 1.0f);
        named.push_back({"output_bias_true", "the bias of the true digit's score", 2, "bias", &b3, label, -1, label, std::move(p)});
    }
    std::vector<float> before, grads;
    for (const auto& n : named) { before.push_back(n.v->data().data()[n.flat]); grads.push_back(n.v->grad().data()[n.flat]); }

    // all parameters before the step, to count how many change
    auto params = m->seq.parameters();
    std::vector<Tensor> all_before;
    for (Variable* p : params) all_before.push_back(p->data());

    // 3. one SGD step
    optim::SGD opt(params, LR);
    opt.step();
    int changed = 0, total = 0;
    for (size_t i = 0; i < params.size(); ++i)
        for (int k = 0; k < params[i]->data().numel(); ++k, ++total) changed += params[i]->data().data()[k] != all_before[i].data()[k];

    J weights = J::array();
    bool all_exact = true;
    for (size_t i = 0; i < named.size(); ++i) {
        const auto& n = named[i];
        const float after = n.v->data().data()[n.flat];
        const float lr_g = grads[i] * LR;                 // as Tensor::mul(lr) computes it
        const bool exact = after == before[i] - lr_g;     // as Tensor::sub computes it
        all_exact = all_exact && exact;
        J w = J::object();
        w.set("id", n.id).set("what", n.what).set("layer", n.layer).set("param", n.param).set("row", n.row);
        if (n.col >= 0) w.set("col", n.col);
        w.set("before", before[i]).set("grad", grads[i]).set("lr_times_grad", lr_g).set("after", after).set("change", after - before[i])
            .set("path", n.path).set("update_exact", exact);
        weights.push(std::move(w));
    }

    // 4. the same image after the step
    Tensor prob_after(std::vector<int>{1, 10});
    float loss_after;
    int pred_after;
    {
        Variable::NoGradGuard ng;
        Variable z2 = m->seq(Variable(xb, false));
        prob_after = z2.data().softmax(1);
        loss_after = loss_fn(z2, Variable(yb, false)).data().data()[0];
        pred_after = tmn::argmax10(z2.data().data());
    }

    const tmn::JV& setup_m = doc["setup"];
    J full = J::object();
    full.set("steps", doc["epochs"].back()["steps"].i()).set("epochs", setup_m["epochs"].i()).set("batch_size", setup_m["batch_size"].i())
        .set("train_samples", setup_m["train_samples"].i())
        .set("steps_per_epoch", doc["epochs"].back()["steps"].i() / setup_m["epochs"].i())
        .set("optimizer", setup_m["optimizer"].str()).set("lr", setup_m["lr"].d()).set("weight_decay", setup_m["weight_decay"].d())
        .set("source", "mnist-mlp.json (epochs[-1].steps and setup)");

    J setup = J::object();
    setup.set("optimizer", "optim::SGD").set("lr", LR_CONFIG).set("momentum", 0).set("batch_size", 1)
        .set("loss", "nn::CrossEntropyLoss").set("weights_from", "mnist-mlp.json layers (the final trained MLP)")
        .set("model", "Linear(784, 32), ReLU, Linear(32, 32), ReLU, Linear(32, 10)").set("parameters", total)
        .set("rule", "SGD without momentum: after = before - lr * grad, so change = -lr * grad");
    J image = J::object();
    image.set("rule", "the first training image that the trained model classifies correctly with a probability of the true digit "
                      "of at least 0.5 and below 0.9")
        .set("set", "train").set("train_index", chosen).set("label", label).set("pixels", trace::jarr(tmn::pixels255(xb, 0)));
    J hid1 = J::object(), hid2 = J::object();
    hid1.set("pre", trace::jarr(vals(h1.data()))).set("post", trace::jarr(vals(a1.data())));
    hid2.set("pre", trace::jarr(vals(h2.data()))).set("post", trace::jarr(vals(a2.data())));
    J fwd = J::object();
    fwd.set("hidden1", std::move(hid1)).set("hidden2", std::move(hid2)).set("scores", trace::jarr(vals(z.data())))
        .set("probabilities", trace::jarr(vals(prob))).set("predicted", predicted).set("p_true", p_true).set("rival", rival)
        .set("p_rival", prob.data()[rival]).set("loss", loss.data().data()[0]);
    J bwd = J::object();
    bwd.set("scores_grad", trace::jarr(vals(dz))).set("hidden1_pre_grad", trace::jarr(vals(dh1))).set("hidden2_pre_grad", trace::jarr(vals(dh2)));
    J aft = J::object();
    aft.set("probabilities", trace::jarr(vals(prob_after))).set("predicted", pred_after).set("p_true", prob_after.data()[label])
        .set("p_rival", prob_after.data()[rival]).set("loss", loss_after);
    J derived = J::object();
    derived.set("p_true_change", static_cast<double>(prob_after.data()[label]) - p_true)
        .set("loss_change", static_cast<double>(loss_after) - loss.data().data()[0])
        .set("loss_check", -std::log(static_cast<double>(p_true)))
        .set("note", "plain arithmetic on the recorded numbers (double); loss_check = -ln(p_true), which the loss equals to float precision");
    J checks = J::object();
    checks.set("weights_reproduce_mnist_mlp_logits", reproduced).set("modules_one_at_a_time_equal_sequential", same_as_sequential)
        .set("every_update_exact", all_exact).set("p_true_rose", prob_after.data()[label] > p_true);
    if (!same_as_sequential || !all_exact) { std::cerr << "trace_one_step: a self-check failed\n"; return 1; }

    J out = trace::envelope("one-step", "one-step", "tools/trace/trace_one_step.cpp");
    out.set("setup", std::move(setup)).set("full_run", std::move(full)).set("image", std::move(image)).set("forward", std::move(fwd))
        .set("backward", std::move(bwd)).set("weights", std::move(weights)).set("after", std::move(aft))
        .set("changed_parameters", changed).set("derived", std::move(derived)).set("checks", std::move(checks));
    std::fputs(out.dump().c_str(), stdout);
    return 0;
}
