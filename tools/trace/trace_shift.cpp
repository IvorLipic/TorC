// trace_shift.cpp  ->  site/data/shift.json   (lessons `convolution`, `mnist-cnn`; brief C8)
//
// How the two trained models cope with digits that are moved, measured on all 10000 test images. The models are the final MLP
// of trace_mnist_mlp.cpp and the final CNN of trace_mnist_cnn.cpp, their weights read back from mnist-mlp.json and
// mnist-cnn.json (exact float32; self-checks confirm that each reproduces the logits recorded there bit for bit, and that the
// unshifted accuracy equals the recorded final test count).
// The producer moves every test image by 0, 1, 2, 3 or 4 whole pixels left, right, up or down: the pixels that leave the
// 28 x 28 image are dropped and the uncovered border is 0 (black). This is input preparation, not library output; everything
// after it (each forward pass, each prediction) is TorC's. The site's digit pad re-centres a drawing by its centre of mass, which
// undoes a shift, so these numbers cannot be reproduced by drawing.
// Keys:
//   setup        {test_samples, shifts [0..4], directions, fill, note}
//   accuracy     {mlp, cnn}: each {parameters, source, left|right|up|down: [{shift, correct, accuracy}] for shift 0..4}
//                (accuracy = correct / 10000, derived in double; the shift-0 entry is the same evaluation in all four lists)
//   feature_maps the first convolution (Conv2d(1, 8, 3, stride 2, padding 1)) after relu for one test image, and for the image
//                moved right by `shift_px` = the stride (2 px): {index, label, pixels[784] (0..255), stride, shift_px,
//                direction, cell_shift (= shift_px / stride), shape (1, 8, 14, 14), original[], shifted[], check}. `check`
//                compares shifted[c][i][j + 1] with original[c][i][j] for every channel c, row i and column j = 0..12
//                (the cells whose input stays inside the image): {compared_cells, max_abs_diff, equal}. So the map moves one cell
//                with the digit, exactly. `not_a_multiple` does the same for a 1 px move, which is not a multiple of the stride:
//                {shift_px, max_abs_diff_unmoved, max_abs_diff_moved_one_cell}, both non-zero: the map changes, it does not move.
//   single_image {index, label, right: [{shift, mlp_predicted, mlp_p_true, cnn_predicted, cnn_p_true}]} for that image
//                (p_true = TorC's softmax probability of the true digit)
//   checks       self-checks
// MNIST comes from MNIST_DIR (tools/trace/run.sh sets it).
#include "trace_mnist.hpp"

#include <cmath>
#include <iostream>
#include <thread>

using namespace torc;
using trace::J;
namespace tmn = trace_mnist;

namespace {

const std::vector<std::pair<std::string, std::pair<int, int>>> DIRS = {{"left", {-1, 0}}, {"right", {1, 0}}, {"up", {0, -1}}, {"down", {0, 1}}};
constexpr int MAX_SHIFT = 4, IMAGE = 0;

Tensor as_cnn(const Tensor& x) { return x.reshape({x.shape()[0], 1, tmn::SIDE, tmn::SIDE}); }

// correct counts for one model: [direction][shift], shift 0 evaluated once
std::vector<std::vector<int>> shifted_counts(bool cnn, const tmn::JV& doc, const Tensor& xs, const Tensor& ys) {
    std::vector<std::vector<int>> res(DIRS.size(), std::vector<int>(MAX_SHIFT + 1, 0));
    auto run = [&](int d0, int d1, bool zero) {   // directions [d0, d1), each thread with its own model (forward caches)
        auto m = cnn ? tmn::make_cnn(1, 2, 3) : tmn::make_mlp(1, 2, 3);
        tmn::load_layers(*m, doc["layers"]);
        auto eval = [&](int dx, int dy) {
            Tensor x = tmn::shift_images(xs, dx, dy);
            const data::TensorDataset ds(cnn ? as_cnn(x) : x, ys);
            return tmn::evaluate(m->seq, ds).correct;
        };
        if (zero) { const int c = eval(0, 0); for (auto& r : res) r[0] = c; }
        for (int d = d0; d < d1; ++d)
            for (int s = 1; s <= MAX_SHIFT; ++s) res[d][s] = eval(DIRS[d].second.first * s, DIRS[d].second.second * s);
    };
    std::vector<std::thread> th;
    for (int d = 0; d < static_cast<int>(DIRS.size()); ++d) th.emplace_back(run, d, d + 1, d == 0);
    for (auto& t : th) t.join();
    return res;
}

J accuracy_json(const std::vector<std::vector<int>>& c, int n, int params, const char* source) {
    J o = J::object();
    o.set("parameters", params).set("source", source);
    for (size_t d = 0; d < DIRS.size(); ++d) {
        J a = J::array();
        for (int s = 0; s <= MAX_SHIFT; ++s) {
            J p = J::object();
            p.set("shift", s).set("correct", c[d][s]).set("accuracy", static_cast<double>(c[d][s]) / n);
            a.push(std::move(p));
        }
        o.set(DIRS[d].first, std::move(a));
    }
    return o;
}

}  // namespace

int main() {
    const std::string dir = tmn::mnist_dir("trace_shift");
    const tmn::JV mlp_doc = tmn::read_trace("mnist-mlp.json"), cnn_doc = tmn::read_trace("mnist-cnn.json");
    const data::MNISTDataset test_set(dir + "/mnist_test.csv");
    const int n = static_cast<int>(test_set.len());
    auto [xs, ys] = test_set.get_batch(0, test_set.len());   // (10000, 784), (10000, 1)
    const data::TensorDataset cnn_view(as_cnn(xs), ys);

    auto mlp = tmn::make_mlp(1, 2, 3);
    tmn::load_layers(*mlp, mlp_doc["layers"]);
    auto cnn = tmn::make_cnn(1, 2, 3);
    tmn::load_layers(*cnn, cnn_doc["layers"]);
    const bool mlp_ok = tmn::reproduces_shown(*mlp, mlp_doc, test_set), cnn_ok = tmn::reproduces_shown(*cnn, cnn_doc, cnn_view);

    std::vector<std::vector<int>> mlp_c, cnn_c;
    std::thread t_mlp([&] { mlp_c = shifted_counts(false, mlp_doc, xs, ys); });
    cnn_c = shifted_counts(true, cnn_doc, xs, ys);
    t_mlp.join();
    const bool zero_ok = mlp_c[0][0] == mlp_doc["epochs"].back()["test_correct"].i() && cnn_c[0][0] == cnn_doc["epochs"].back()["test_correct"].i();

    // feature maps of the first convolution for one image, and for the image moved right by the stride
    Variable::NoGradGuard ng;
    const int stride = 2, cells = 14, ch = 8;
    auto [x1, y1] = test_set.get_batch(IMAGE, IMAGE + 1);
    const int label = static_cast<int>(y1.data()[0]);
    auto maps = [&](int dx) { return relu((*cnn->first_conv)(Variable(as_cnn(tmn::shift_images(x1, dx, 0)), false))).data(); };
    const Tensor m0 = maps(0), m2 = maps(stride), m1 = maps(1);
    auto moved_diff = [&](const Tensor& a, int cell_shift, int& compared) {   // a[c][i][j + cell_shift] vs m0[c][i][j]
        float d = 0.0f;
        compared = 0;
        for (int c = 0; c < ch; ++c)
            for (int i = 0; i < cells; ++i)
                for (int j = 0; j + 1 < cells; ++j) {
                    d = std::fmax(d, std::fabs(a.data()[(c * cells + i) * cells + j + cell_shift] - m0.data()[(c * cells + i) * cells + j]));
                    ++compared;
                }
        return d;
    };
    int compared = 0, unused = 0;
    const float d2 = moved_diff(m2, 1, compared), d1_still = moved_diff(m1, 0, unused), d1_moved = moved_diff(m1, 1, unused);
    int ink_lost = 0;   // ink in the last `stride` columns, which the move pushes out of the image
    for (int r = 0; r < tmn::SIDE; ++r) for (int c = tmn::SIDE - stride; c < tmn::SIDE; ++c) ink_lost += x1.data()[r * tmn::SIDE + c] > 0.0f;
    J check = J::object();
    check.set("compared_cells", compared).set("max_abs_diff", d2).set("equal", d2 == 0.0f).set("ink_pixels_pushed_out", ink_lost);
    J notm = J::object();
    notm.set("shift_px", 1).set("max_abs_diff_unmoved", d1_still).set("max_abs_diff_moved_one_cell", d1_moved);
    J fm = J::object();
    fm.set("index", IMAGE).set("label", label).set("pixels", trace::jarr(tmn::pixels255(x1, 0))).set("stride", stride)
        .set("shift_px", stride).set("direction", "right").set("cell_shift", 1).set("shape", trace::jarr(m0.shape()))
        .set("original", trace::jarr(trace::values_of(m0))).set("shifted", trace::jarr(trace::values_of(m2)))
        .set("check", std::move(check)).set("not_a_multiple", std::move(notm));

    J single = J::array();
    for (int s = 0; s <= MAX_SHIFT; ++s) {
        const Tensor xsft = tmn::shift_images(x1, s, 0);
        const Tensor zm = mlp->seq(Variable(xsft, false)).data(), zc = cnn->seq(Variable(as_cnn(xsft), false)).data();
        const Tensor pm = zm.softmax(1), pc = zc.softmax(1);
        J r = J::object();
        r.set("shift", s).set("mlp_predicted", tmn::argmax10(zm.data())).set("mlp_p_true", pm.data()[label])
            .set("cnn_predicted", tmn::argmax10(zc.data())).set("cnn_p_true", pc.data()[label]);
        single.push(std::move(r));
    }
    J single_image = J::object();
    single_image.set("index", IMAGE).set("label", label).set("right", std::move(single));

    const bool maps_ok = d2 == 0.0f && ink_lost == 0 && trace::values_of(m0) == cnn_doc["first_maps"]["values"].floats();
    if (!mlp_ok || !cnn_ok || !zero_ok || !maps_ok) { std::cerr << "trace_shift: a self-check failed\n"; return 1; }

    J setup = J::object();
    J shifts = J::array();
    for (int s = 0; s <= MAX_SHIFT; ++s) shifts.push(s);
    J dirs = J::array();
    for (const auto& d : DIRS) dirs.push(d.first);
    setup.set("test_samples", n).set("shifts", std::move(shifts)).set("directions", std::move(dirs)).set("fill", "zero")
        .set("note", "Every test image is moved by whole pixels; pixels that leave the image are dropped and the border is 0.");
    J acc = J::object();
    acc.set("mlp", accuracy_json(mlp_c, n, tmn::count_params(mlp->seq), "mnist-mlp.json layers"))
        .set("cnn", accuracy_json(cnn_c, n, tmn::count_params(cnn->seq), "mnist-cnn.json layers"));
    J checks = J::object();
    checks.set("mlp_weights_reproduce_logits", mlp_ok).set("cnn_weights_reproduce_logits", cnn_ok)
        .set("shift_0_equals_recorded_test_correct", zero_ok).set("original_maps_equal_mnist_cnn_first_maps", maps_ok);

    J o = trace::envelope("shift", "convolution", "tools/trace/trace_shift.cpp");
    o.set("setup", std::move(setup)).set("accuracy", std::move(acc)).set("feature_maps", std::move(fm))
        .set("single_image", std::move(single_image)).set("checks", std::move(checks));
    std::fputs(o.dump().c_str(), stdout);
    return 0;
}
