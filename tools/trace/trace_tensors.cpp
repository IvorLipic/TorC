// trace_tensors.cpp  ->  site/data/tensors.json   (lesson `tensors`)
//
// Shape and row-major storage, reshape (a copy), element-wise broadcasting (results and the real ShapeError texts), reductions
// along an axis, transpose and matmul (results and errors), all computed by the real Tensor class. The example tensor is
// a = [[1, 2, 3], [4, 5, 6]]. Every value and message is library output, with two exceptions: `flat` = i*3 + j is computed here
// (a self-check confirms that a.data()[flat] holds a[i, j]), and `changing_the_copy_changes_a` compares a before and after.
#include "trace_util.hpp"

#include <cstdlib>
#include <iostream>

using namespace torc;
using trace::J;

namespace {

J shape_json(const std::vector<int>& s) { return trace::jarr(s); }

// 1, 2, ..., n: small distinct values, so every entry of a product says which numbers went into it
std::vector<float> iota(int n) {
    std::vector<float> v(n);
    for (int k = 0; k < n; ++k) v[k] = static_cast<float>(k + 1);
    return v;
}

// The result of a tensor-valued call, or the exception it threw: {shape, values} or {exception, what}.
J result_or_error(const std::function<Tensor()>& f) {
    J o = J::object();
    try {
        Tensor r = f();
        o.set("shape", shape_json(r.shape())).set("values", trace::jarr(trace::values_of(r))).set("exception", "none");
    } catch (const ShapeError& e) {
        o.set("exception", "ShapeError").set("what", e.what());
    } catch (const TorcError& e) {
        o.set("exception", "TorcError").set("what", e.what());
    }
    return o;
}

}  // namespace

int main() {
    bool ok = true;
    const Tensor a({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3});

    // ---- shape and row-major storage ------------------------------------------------------------------------------
    J index = J::array();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) {
            const float v = a[i, j];                                   // Tensor::operator[] (checked, row-major)
            const int flat = i * 3 + j;
            ok = ok && a.data()[flat] == v;                            // the flat position i*3 + j holds the same number
            J e = J::object();
            e.set("i", i).set("j", j).set("flat", flat).set("value", v);
            index.push(std::move(e));
        }
    J example = J::object();
    example.set("shape", shape_json(a.shape())).set("numel", a.numel()).set("values", trace::jarr(trace::values_of(a))).set("index", std::move(index));

    // ---- reshape copies -------------------------------------------------------------------------------------------
    Tensor r = a.reshape({3, 2});
    const std::vector<float> before = trace::values_of(a);
    r.data()[0] = 100.f;                                                // change the reshaped tensor
    const bool copies = trace::values_of(a) == before;                  // ... and a is unchanged
    J reshape = J::object();
    reshape.set("shape", shape_json(a.reshape({3, 2}).shape())).set("values", trace::jarr(trace::values_of(a.reshape({3, 2}))))
        .set("changing_the_copy_changes_a", !copies)
        .set("error", result_or_error([&] { return a.reshape({4, 2}); }));

    // ---- broadcasting: a (2, 3) plus b of several shapes ------------------------------------------------------------
    struct B { const char* id; std::vector<int> shape; std::vector<float> values; };
    const std::vector<B> bs = {
        {"same", {2, 3}, {10, 20, 30, 40, 50, 60}},
        {"row", {3}, {10, 20, 30}},
        {"row2d", {1, 3}, {10, 20, 30}},
        {"column", {2, 1}, {100, 200}},
        {"scalar", {1}, {100}},
        {"wrong-row", {2}, {10, 20}},
        {"wrong-column", {3, 1}, {100, 200, 300}},
    };
    J broadcast = J::array();
    for (const auto& b : bs) {
        Tensor bt = trace::make_tensor(b.shape, b.values);
        J e = J::object();
        e.set("id", b.id).set("b", trace::tensor_json(bt)).set("sum", result_or_error([&] { return a.add(bt); }));
        broadcast.push(std::move(e));
    }
    // shape-only examples of the rule (more dimensions)
    struct P { std::vector<int> x, y; };
    const std::vector<P> ps = {{{4, 1}, {3}}, {{4, 1, 3}, {2, 1}}, {{5, 4}, {4}}, {{5, 4}, {5}}};
    J pairs = J::array();
    for (const auto& p : ps) {
        J e = J::object();
        e.set("x", shape_json(p.x)).set("y", shape_json(p.y));
        auto c = trace::catch_call([&] { (void)broadcast_shape(p.x, p.y); });
        if (c.type == "none") e.set("result", shape_json(broadcast_shape(p.x, p.y))).set("exception", "none");
        else e.set("exception", c.type).set("what", c.what);
        pairs.push(std::move(e));
    }

    // ---- reductions -----------------------------------------------------------------------------------------------
    J reductions = J::array();
    auto red = [&](const char* op, int axis, const std::function<Tensor()>& f) {
        J e = J::object();
        e.set("op", op).set("axis", axis).set("result", result_or_error(f));
        reductions.push(std::move(e));
    };
    for (int axis : {0, 1}) {
        red("sum", axis, [&] { return a.sum(axis); });
        red("mean", axis, [&] { return a.mean(axis); });
        red("max", axis, [&] { return a.max(axis); });
    }
    J whole = J::object();
    whole.set("sum", a.sum()).set("mean", a.mean()).set("max", a.max());
    J red_error = result_or_error([&] { return a.sum(2); });
    J red_negative = result_or_error([&] { return a.sum(-1); });       // axis -1 is the last axis

    // ---- transpose and matmul ------------------------------------------------------------------------------------
    const Tensor at = a.transpose({1, 0});
    J matmul = J::object();
    matmul.set("transpose", trace::tensor_json(at))
        .set("a_at", result_or_error([&] { return a.matmul(at); }))                         // (2, 3)(3, 2) -> (2, 2)
        .set("at_a", result_or_error([&] { return at.matmul(a); }))                         // (3, 2)(2, 3) -> (3, 3)
        .set("a_a", result_or_error([&] { return a.matmul(a); }))                           // inner sizes 3 and 2: error
        .set("batched", result_or_error([&] { return trace::make_tensor({4, 2, 3}, iota(24)).matmul(at); }))  // (4, 2, 3)(3, 2)
        .set("batched_both", result_or_error([&] {                                         // (2, 1, 2, 3)(3, 3, 2) -> (2, 3, 2, 2)
            return trace::make_tensor({2, 1, 2, 3}, iota(12)).matmul(trace::make_tensor({3, 3, 2}, iota(18)));
        }));

    // ---- the shapes of the check question: x of shape (4, 5), x.sum(0) ------------------------------------------------
    const Tensor x45(std::vector<int>{4, 5});
    J check_sum = J::object();
    check_sum.set("x_shape", shape_json(x45.shape())).set("sum0_shape", shape_json(x45.sum(0).shape()));

    J doc = trace::envelope("tensors", "tensors", "tools/trace/trace_tensors.cpp");
    doc.set("example", std::move(example)).set("reshape", std::move(reshape)).set("broadcast", std::move(broadcast))
        .set("broadcast_shapes", std::move(pairs)).set("reductions", std::move(reductions)).set("whole", std::move(whole))
        .set("reduce_error", std::move(red_error)).set("reduce_negative_axis", std::move(red_negative)).set("matmul", std::move(matmul))
        .set("check_sum", std::move(check_sum));
    std::fputs(doc.dump().c_str(), stdout);
    if (!ok) { std::cerr << "trace_tensors: row-major self-check failed\n"; return 1; }
    return 0;
}
