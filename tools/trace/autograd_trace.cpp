// autograd_trace.cpp
// Runs four small computations with the real torc::Variable autograd and writes
// everything the visualizations need (graph edges taken from the real tape,
// forward values, accumulated gradients, finite-difference check) as JSON.
//
// Output schema: docs/trace-schema.md (envelope + scenes[]). The site's engine tests compare every value and gradient
// to this output; prose numbers are bound to it by tests.
#include "trace_util.hpp"

using namespace torc;
using trace::Graph;
using trace::Scene;

int main() {
    std::vector<Scene> scenes;

    // Scene 1 — a Variable used twice: y = x*x + x at x = 3.
    {
        Scene s;
        s.id = "reuse"; s.title = "One variable, two uses"; s.expression = "y = x*x + x";
        s.note = "x feeds both inputs of mul and the add, so backward accumulates one contribution per use into x.grad.";
        s.leaves = {{"x", Tensor({3.0f}, {1})}};
        s.build = [](Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
            Variable& x = *L["x"];
            Variable& m = g.op("m", "mul", mul(x, x));
            return g.op("y", "add", add(m, x));
        };
        scenes.push_back(std::move(s));
    }

    // Scene 2 — chain rule through relu: f = sum(relu(a*b + c)); one element is clipped by relu.
    {
        Scene s;
        s.id = "chain"; s.title = "Chain rule through ReLU"; s.expression = "f = sum(relu(a*b + c))";
        s.note = "The third element has a*b+c < 0, so relu blocks its gradient (local derivative 0).";
        s.leaves = {{"a", Tensor({1.0f, -2.0f, 3.0f}, {3})},
                    {"b", Tensor({2.0f, 2.0f, -1.0f}, {3})},
                    {"c", Tensor({0.5f, 5.0f, 1.0f}, {3})}};
        s.build = [](Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
            Variable& ab = g.op("ab", "mul", mul(*L["a"], *L["b"]));
            Variable& z = g.op("z", "add", add(ab, *L["c"]));
            Variable& r = g.op("r", "relu", relu(z));
            return g.op("f", "sum", sum(r));
        };
        scenes.push_back(std::move(s));
    }

    // Scene 3 — tiny MLP: 2 -> 2 (ReLU) -> 1, MSE loss, batch of 1.
    {
        Scene s;
        s.id = "mlp"; s.title = "A tiny network (batch of 2)"; s.expression = "L = mse( relu(x@W1 + b1)@W2 + b2, t )";
        s.note = "Built from explicit ops so every intermediate is visible. Two entries of h (h[0][1] and h[1][0]) are negative, so ReLU blocks them: zero gradient for those entries.";
        s.leaves = {{"x",  Tensor({1.0f, 2.0f, -1.0f, 1.0f}, {2, 2})},
                    {"W1", Tensor({0.5f, -1.0f, 0.25f, 0.75f}, {2, 2})},
                    {"b1", Tensor({0.1f, -0.8f}, {2})},
                    {"W2", Tensor({1.0f, 0.5f}, {2, 1})},
                    {"b2", Tensor({0.0f}, {1})},
                    {"t",  Tensor({1.0f, 0.0f}, {2, 1})}};
        s.constants = {"x", "t"};
        s.build = [](Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
            Variable& xw = g.op("xW1", "matmul", matmul(*L["x"], *L["W1"]));
            Variable& h = g.op("h", "add", add(xw, *L["b1"]));
            Variable& a = g.op("a", "relu", relu(h));
            Variable& aw = g.op("aW2", "matmul", matmul(a, *L["W2"]));
            Variable& o = g.op("o", "add", add(aw, *L["b2"]));
            Variable& d = g.op("d", "sub", sub(o, *L["t"]));
            Variable& q = g.op("q", "mul", mul(d, d));
            return g.op("L", "mean", mean(q));
        };
        scenes.push_back(std::move(s));
    }

    // Scene 4 — the small network of the lesson `backprop`: 2 inputs -> 2 hidden units (ReLU) -> 1 output, ONE sample (no batch),
    // loss = (o - t)^2 written as sub, mul, mean (the mean of one number is that number). Both hidden units are on (h = 1.1 and 0.3,
    // far from the ReLU kink for the finite-difference step). The second output weight is negative, so the second hidden unit gets
    // a gradient of the opposite sign. W1[i][j] is the weight from input i to hidden unit j (x @ W1, as in scene 3).
    {
        Scene s;
        s.id = "net"; s.title = "A small network (one sample)"; s.expression = "L = mean( (relu(x@W1 + b1)@W2 + b2 - t)^2 )";
        s.note = "One sample, so the loss is the squared miss. Each weight's gradient is the gradient of the unit it feeds times the input it carries.";
        s.leaves = {{"x",  Tensor({1.0f, 2.0f}, {1, 2})},
                    {"W1", Tensor({0.5f, -0.5f, 0.25f, 0.5f}, {2, 2})},
                    {"b1", Tensor({0.1f, -0.2f}, {2})},
                    {"W2", Tensor({1.0f, -0.5f}, {2, 1})},
                    {"b2", Tensor({0.0f}, {1})},
                    {"t",  Tensor({2.0f}, {1, 1})}};
        s.constants = {"x", "t"};
        s.build = [](Graph& g, std::map<std::string, Variable*>& L) -> Variable& {
            Variable& xw = g.op("xW1", "matmul", matmul(*L["x"], *L["W1"]));
            Variable& h = g.op("h", "add", add(xw, *L["b1"]));
            Variable& a = g.op("a", "relu", relu(h));
            Variable& aw = g.op("aW2", "matmul", matmul(a, *L["W2"]));
            Variable& o = g.op("o", "add", add(aw, *L["b2"]));
            Variable& d = g.op("d", "sub", sub(o, *L["t"]));
            Variable& q = g.op("q", "mul", mul(d, d));
            return g.op("L", "mean", mean(q));
        };
        scenes.push_back(std::move(s));
    }

    trace::J doc = trace::envelope("autograd", "backprop-lab", "tools/trace/autograd_trace.cpp");
    trace::J arr = trace::J::array();
    int net_parameters = 0;   // the tracked leaves of scene `net`: every weight and bias (derived: a count of the scene's numbers)
    for (const Scene& sc : scenes) {
        if (sc.id == "net")
            for (const auto& l : sc.leaves)
                if (!trace::is_const(sc, l.first)) net_parameters += l.second.numel();
        arr.push(trace::run_scene(sc).json);
    }
    doc.set("scenes", std::move(arr));
    trace::J counts = trace::J::object();
    counts.set("inputs", 2).set("hidden", 2).set("outputs", 1).set("parameters", net_parameters);
    doc.set("net_counts", std::move(counts));
    std::fputs(doc.dump().c_str(), stdout);
    return 0;
}
