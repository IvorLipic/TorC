// trace_losses.cpp  ->  site/data/losses.json   (lesson `losses`)
//
// 1. nn::MSELoss on three predictions of equal shape to the targets: the loss and the gradient from backward().
// 2. nn::CrossEntropyLoss for one sample with three classes: three rows of logits times three target classes, with the loss,
//    the softmax of the row and the gradient with respect to the logits from backward().
// 3. A batch of two samples (the loss is the mean over the batch).
// 4. Two-class logits [z, 0] with target 0 for z from -6 to 6: cross-entropy and its gradient, and the squared error between
//    softmax probabilities and the one-hot target (softmax then nn::MSELoss) and its gradient, both from backward().
// 5. The errors CrossEntropyLoss throws for bad targets and logits.
// 6. nn::MSELoss on a prediction of shape (3, 1) and targets of shape (3): the exception forward() throws and its what() text
//    (TorC after PR #6 requires equal shapes; v0.1.0 broadcast them to (3, 3) and threw nothing).
// Everything is library output.
#include "trace_util.hpp"

#include "torc/nn/losses.hpp"

#include <iostream>

using namespace torc;
using trace::J;

namespace {

J ce_case(const std::vector<float>& row, int target) {
    nn::CrossEntropyLoss ce;
    Variable z(trace::make_tensor({1, 3}, row), true);
    const Variable t(trace::make_tensor({1}, {static_cast<float>(target)}), false);
    Variable L = ce(z, t);
    L.backward();
    J o = J::object();
    o.set("target", target).set("loss", L.data().data()[0]).set("softmax", trace::jarr(trace::values_of(z.data().softmax(1))))
        .set("grad", trace::jarr(trace::values_of(z.grad())));
    return o;
}

J error_case(const char* id, const std::vector<int>& zshape, const std::vector<float>& zv, const std::vector<int>& tshape, const std::vector<float>& tv) {
    nn::CrossEntropyLoss ce;
    const Variable z(trace::make_tensor(zshape, zv), true), t(trace::make_tensor(tshape, tv), false);
    auto c = trace::catch_call([&] { (void)ce(z, t); });
    J o = J::object();
    o.set("id", id).set("targets", trace::jarr(tv)).set("exception", c.type).set("what", c.what);
    return o;
}

}  // namespace

int main() {
    // ---- 1. mean squared error -----------------------------------------------------------------------------------------
    nn::MSELoss mse;
    Variable y(trace::make_tensor({3, 1}, {2.5f, 0.0f, 2.0f}), true);
    const Variable t(trace::make_tensor({3, 1}, {3.0f, -0.5f, 2.0f}), false);
    Variable Lm = mse(y, t);
    Lm.backward();
    J m = J::object();
    m.set("prediction", trace::jarr(trace::values_of(y.data()))).set("target", trace::jarr(trace::values_of(t.data())))
        .set("loss", Lm.data().data()[0]).set("grad", trace::jarr(trace::values_of(y.grad())));

    // ---- 2. cross-entropy, one sample ------------------------------------------------------------------------------------
    struct Row { const char* id; std::vector<float> v; };
    const std::vector<Row> rows = {{"uniform", {0, 0, 0}}, {"mild", {2, 1, 0.1f}}, {"confident", {4, 1, -2}}};
    J ce_rows = J::array();
    for (const auto& r : rows) {
        J cases = J::array();
        for (int target = 0; target < 3; ++target) cases.push(ce_case(r.v, target));
        J e = J::object();
        e.set("id", r.id).set("logits", trace::jarr(r.v)).set("cases", std::move(cases));
        ce_rows.push(std::move(e));
    }

    // ---- 3. a batch of two ------------------------------------------------------------------------------------------------
    nn::CrossEntropyLoss ce;
    Variable zb(trace::make_tensor({2, 3}, {2, 1, 0.1f, 4, 1, -2}), true);
    const Variable tb(trace::make_tensor({2}, {0, 2}), false);
    Variable Lb = ce(zb, tb);
    Lb.backward();
    J batch = J::object();
    batch.set("logits", trace::jarr(trace::values_of(zb.data()))).set("targets", trace::jarr(trace::values_of(tb.data())))
        .set("loss", Lb.data().data()[0]).set("grad", trace::jarr(trace::values_of(zb.grad())));

    // ---- 4. cross-entropy against squared error on probabilities, two classes ------------------------------------------------
    J zs = J::array(), ce_loss = J::array(), ce_grad = J::array(), sq_loss = J::array(), sq_grad = J::array();
    for (int k = 0; k <= 48; ++k) {
        const float zk = -6.0f + 0.25f * static_cast<float>(k);
        zs.push(zk);
        {
            Variable z(trace::make_tensor({1, 2}, {zk, 0.0f}), true);
            const Variable t0(trace::make_tensor({1}, {0.0f}), false);
            Variable L = ce(z, t0);
            L.backward();
            ce_loss.push(L.data().data()[0]);
            ce_grad.push(z.grad().data()[0]);
        }
        {
            Variable z(trace::make_tensor({1, 2}, {zk, 0.0f}), true);
            const Variable onehot(trace::make_tensor({1, 2}, {1.0f, 0.0f}), false);
            Variable p = softmax(z, 1);
            Variable L = mse(p, onehot);
            L.backward();
            sq_loss.push(L.data().data()[0]);
            sq_grad.push(z.grad().data()[0]);
        }
    }
    J curves = J::object();
    curves.set("z", std::move(zs)).set("ce_loss", std::move(ce_loss)).set("ce_grad", std::move(ce_grad))
        .set("sq_loss", std::move(sq_loss)).set("sq_grad", std::move(sq_grad));

    // ---- 5. errors ----------------------------------------------------------------------------------------------------------
    J errors = J::array();
    errors.push(error_case("not-integer", {1, 3}, {2, 1, 0.1f}, {1}, {1.5f}))
        .push(error_case("out-of-range", {1, 3}, {2, 1, 0.1f}, {1}, {3}))
        .push(error_case("rank-1-logits", {3}, {2, 1, 0.1f}, {1}, {0}))
        .push(error_case("length", {2, 3}, {2, 1, 0.1f, 4, 1, -2}, {3}, {0, 1, 2}))
        .push(error_case("column-targets", {2, 3}, {2, 1, 0.1f, 4, 1, -2}, {2, 1}, {0, 2}));

    // ---- 6. MSELoss with shapes that do not match ----------------------------------------------------------------------------
    J mse_mismatch = J::object();
    {
        const Variable yb(trace::make_tensor({3, 1}, {2.5f, 0.0f, 2.0f}), true);
        const Variable tb3(trace::make_tensor({3}, {3.0f, -0.5f, 2.0f}), false);
        auto c = trace::catch_call([&] { (void)mse(yb, tb3); });
        mse_mismatch.set("prediction_shape", trace::jarr(yb.data().shape())).set("target_shape", trace::jarr(tb3.data().shape()))
            .set("exception", c.type).set("what", c.what);
    }

    J doc = trace::envelope("losses", "losses", "tools/trace/trace_losses.cpp");
    doc.set("mse", std::move(m)).set("ce", std::move(ce_rows)).set("batch", std::move(batch)).set("curves", std::move(curves))
        .set("errors", std::move(errors)).set("mse_mismatch", std::move(mse_mismatch));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
