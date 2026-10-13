// autograd_trace.cpp
// Runs three small computations with the real torc::Variable autograd and writes
// everything the visualizations need (graph edges taken from the real tape,
// forward values, accumulated gradients, finite-difference check) as JSON.
//
// Output schema: see visualizations/README.md. The Manim scenes read every value and edge
// from this output. (The website prose quotes a few of these numbers by hand; they were
// checked against this output.)
#include "torc/autograd.hpp"
#include "torc/tensor.hpp"

#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace torc;

namespace {

struct Node {
    std::string name, op, role;  // role: "leaf" | "op"
    Variable* var;
};

// Owns Variables at stable addresses (the tape stores raw Variable* pointers).
struct Graph {
    std::deque<Variable> vars;
    std::vector<Node> nodes;
    std::map<const Variable*, std::string> names;

    Variable& leaf(const std::string& name, Tensor t, bool requires_grad) {
        vars.emplace_back(std::move(t), requires_grad);
        register_node(name, "leaf", "leaf", vars.back());
        return vars.back();
    }
    Variable& op(const std::string& name, const std::string& op_name, Variable v) {
        vars.emplace_back(std::move(v));
        register_node(name, op_name, "op", vars.back());
        return vars.back();
    }
    void register_node(const std::string& name, const std::string& op_name,
                       const std::string& role, Variable& v) {
        nodes.push_back({name, op_name, role, &v});
        names[&v] = name;
    }
};

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}

std::string num(float v) {
    if (v == 0.0f) v = 0.0f;  // normalise -0 -> 0 for display; the raw sign is irrelevant to the math
    char b[48];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

std::string json_tensor(const Tensor& t) {
    std::string s = "{\"shape\":[";
    for (size_t i = 0; i < t.shape().size(); ++i) s += (i ? "," : "") + std::to_string(t.shape()[i]);
    s += "],\"values\":[";
    for (size_t i = 0; i < t.numel(); ++i) s += (i ? "," : "") + num(t.data()[i]);
    return s + "]}";
}

// Scenes are written once as builders that take the leaf values, so the same code
// runs for the traced pass and for the finite-difference evaluations.
struct Scene {
    std::string id, title, expression, note;
    std::vector<std::pair<std::string, Tensor>> leaves;  // name -> value (all requires_grad unless listed const)
    std::vector<std::string> constants;                   // leaf names with requires_grad=false
    // builds the graph; the last node registered must be the scalar root
    std::function<Variable&(Graph&, std::map<std::string, Variable*>&)> build;
};

bool is_const(const Scene& s, const std::string& n) {
    for (auto& c : s.constants) if (c == n) return true;
    return false;
}

Variable& build_scene(const Scene& s, Graph& g, const std::vector<Tensor>& values) {
    std::map<std::string, Variable*> L;
    for (size_t i = 0; i < s.leaves.size(); ++i) {
        const auto& n = s.leaves[i].first;
        L[n] = &g.leaf(n, values[i], !is_const(s, n));
    }
    return s.build(g, L);
}

float eval_loss(const Scene& s, const std::vector<Tensor>& values) {
    Graph g;
    Variable::NoGradGuard ng;
    Variable& root = build_scene(s, g, values);
    return root.data().data()[0];
}

std::string run_scene(const Scene& s, bool last) {
    std::vector<Tensor> init;
    for (auto& l : s.leaves) init.push_back(l.second);

    Graph g;
    Variable& root = build_scene(s, g, init);

    // Edges are read from the real tapes BEFORE backward(): backward() clears the root's tape.
    std::map<const Variable*, std::vector<std::string>> edges;
    for (auto& n : g.nodes)
        for (auto& e : n.var->tape())
            for (auto* in : e.inputs) edges[n.var].push_back(g.names[in]);

    root.backward();

    std::string out = "    {\n";
    out += "      \"id\":\"" + s.id + "\",\"title\":\"" + esc(s.title) + "\",\"expression\":\"" + esc(s.expression) + "\",\n";
    out += "      \"note\":\"" + esc(s.note) + "\",\n";
    out += "      \"root\":\"" + g.names[&root] + "\",\n";
    out += "      \"root_has_grad\":" + std::string(root.has_grad() ? "true" : "false") + ",\n";
    out += "      \"nodes\":[\n";
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& n = g.nodes[i];
        const Variable& v = *n.var;
        out += "        {\"name\":\"" + n.name + "\",\"op\":\"" + n.op + "\",\"role\":\"" + n.role + "\"";
        out += ",\"requires_grad\":" + std::string(v.requires_grad() ? "true" : "false");
        out += ",\"inputs\":[";
        std::string ins;
        for (auto& in : edges[n.var]) ins += (ins.empty() ? "" : ",") + std::string("\"") + in + "\"";
        out += ins + "]";
        out += ",\"value\":" + json_tensor(v.data());
        out += ",\"grad\":" + (v.has_grad() ? json_tensor(v.grad()) : std::string("null"));
        out += "}" + std::string(i + 1 < g.nodes.size() ? "," : "") + "\n";
    }
    out += "      ],\n";

    // Finite-difference check on every requires-grad leaf (central differences, h=1e-2,
    // the project's calibrated setting; see docs/DESIGN.md "Gradient checking").
    const float h = 1e-2f;
    out += "      \"finite_difference\":{\"h\":" + num(h) + ",\"atol\":0.01,\"leaves\":[\n";
    bool first = true;
    for (size_t li = 0; li < s.leaves.size(); ++li) {
        const auto& name = s.leaves[li].first;
        if (is_const(s, name)) continue;
        Variable* lv = nullptr;
        for (auto& nd : g.nodes) if (nd.name == name) lv = nd.var;
        std::string fd = "[", an = "[";
        float max_err = 0.0f;
        for (size_t k = 0; k < init[li].numel(); ++k) {
            auto plus = init, minus = init;
            plus[li].data()[k] += h;
            minus[li].data()[k] -= h;
            float num_g = (eval_loss(s, plus) - eval_loss(s, minus)) / (2 * h);
            float ana_g = lv->has_grad() ? lv->grad().data()[k] : 0.0f;
            max_err = std::fmax(max_err, std::fabs(num_g - ana_g));
            fd += (k ? "," : "") + num(num_g);
            an += (k ? "," : "") + num(ana_g);
        }
        out += std::string(first ? "" : ",\n") + "        {\"name\":\"" + name + "\",\"numerical\":" + fd + "],\"analytic\":" + an + "],\"max_abs_err\":" + num(max_err) + ",\"ok\":" + (max_err <= 0.01f ? "true" : "false") + "}";
        first = false;
    }
    out += "\n      ]}\n    }" + std::string(last ? "" : ",") + "\n";
    return out;
}

}  // namespace

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

    std::string out = "{\n  \"library\":\"torc\",\n  \"generator\":\"visualizations/examples/autograd_trace.cpp\",\n  \"scenes\":[\n";
    for (size_t i = 0; i < scenes.size(); ++i) out += run_scene(scenes[i], i + 1 == scenes.size());
    out += "  ]\n}\n";
    std::fputs(out.c_str(), stdout);
    return 0;
}
