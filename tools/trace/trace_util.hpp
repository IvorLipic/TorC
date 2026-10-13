// trace_util.hpp
// Shared helpers for the trace programs (tools/trace/*.cpp): every program runs the REAL TorC library and writes JSON
// that the site is built from. Nothing here re-implements library math; it only records what the library did.
//
//   trace::J            tiny JSON value tree (ordered objects, deterministic number formatting)
//   trace::Graph        owns tracked Variables at stable addresses and registers nodes by name
//   trace::Scene        a named computation written once as a builder; run_scene() records nodes, edges (read from the
//                       real tape BEFORE backward), values, grads and a finite-difference check
//   trace::envelope()   the common header every JSON file starts with (schema "torc-trace/1", see docs/trace-schema.md)
//
// Number formatting: float32 values (what the library computes) are written with "%.9g" (round-trips a float exactly);
// double values (configuration such as a learning rate, or quantities derived in double) are written shortest-round-trip.
#pragma once

#include "torc/autograd.hpp"
#include "torc/tensor.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <deque>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Set by CMake (tools/trace/CMakeLists.txt); the fallbacks only matter for ad-hoc compiles.
#ifndef TORC_TRACE_REF
#define TORC_TRACE_REF "unknown"
#endif
#ifndef TORC_TRACE_FLAGS
#define TORC_TRACE_FLAGS "unknown"
#endif

namespace trace {

// ---------------------------------------------------------------------------------------------------------------------
// JSON value tree
// ---------------------------------------------------------------------------------------------------------------------
class J {
public:
    enum class Kind { Null, Bool, Num, Str, Arr, Obj };

    J() = default;
    J(bool b) : kind_(Kind::Bool), s_(b ? "true" : "false") {}
    J(const char* s) : kind_(Kind::Str), s_(s) {}
    J(std::string s) : kind_(Kind::Str), s_(std::move(s)) {}
    J(float v) : kind_(Kind::Num), s_(fmt_float(v)) {}
    J(double v) : kind_(Kind::Num), s_(fmt_double(v)) {}
    template <std::integral T>
        requires(!std::same_as<T, bool>)
    J(T v) : kind_(Kind::Num), s_(std::to_string(v)) {}

    static J array() { J j; j.kind_ = Kind::Arr; return j; }
    static J object() { J j; j.kind_ = Kind::Obj; return j; }

    // Array append / object insert (insertion order is the output order). Return *this for chaining.
    J& push(J v) { arr_.push_back(std::move(v)); return *this; }
    J& set(std::string key, J v) { obj_.emplace_back(std::move(key), std::move(v)); return *this; }

    [[nodiscard]] Kind kind() const { return kind_; }
    [[nodiscard]] bool is_container() const { return kind_ == Kind::Arr || kind_ == Kind::Obj; }

    [[nodiscard]] std::string dump() const {
        std::string out;
        pretty(out, 0);
        out += "\n";
        return out;
    }

    // float32 as the library holds it. -0 is written as 0 (sign of zero is irrelevant to the mathematics).
    static std::string fmt_float(float v) {
        if (!std::isfinite(v)) throw std::runtime_error("trace: non-finite float in output");
        if (v == 0.0f) v = 0.0f;
        char b[48];
        std::snprintf(b, sizeof b, "%.9g", static_cast<double>(v));
        return b;
    }
    static std::string fmt_double(double v) {
        if (!std::isfinite(v)) throw std::runtime_error("trace: non-finite double in output");
        if (v == 0.0) v = 0.0;
        char b[64];
        auto r = std::to_chars(b, b + sizeof b, v);  // shortest representation that round-trips
        return std::string(b, r.ptr);
    }

private:
    Kind kind_ = Kind::Null;
    std::string s_;                                   // Bool/Num: literal text, Str: raw (unescaped) text
    std::vector<J> arr_;
    std::vector<std::pair<std::string, J>> obj_;

    static std::string quote(const std::string& s) {
        std::string o = "\"";
        for (unsigned char c : s) {
            switch (c) {
                case '"': o += "\\\""; break;
                case '\\': o += "\\\\"; break;
                case '\n': o += "\\n"; break;
                case '\t': o += "\\t"; break;
                case '\r': o += "\\r"; break;
                default:
                    if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                    else o += static_cast<char>(c);
            }
        }
        return o + "\"";
    }

    [[nodiscard]] std::string compact() const {
        switch (kind_) {
            case Kind::Null: return "null";
            case Kind::Bool:
            case Kind::Num: return s_;
            case Kind::Str: return quote(s_);
            case Kind::Arr: {
                std::string o = "[";
                for (size_t i = 0; i < arr_.size(); ++i) o += (i ? "," : "") + arr_[i].compact();
                return o + "]";
            }
            case Kind::Obj: {
                std::string o = "{";
                for (size_t i = 0; i < obj_.size(); ++i) o += (i ? "," : "") + quote(obj_[i].first) + ":" + obj_[i].second.compact();
                return o + "}";
            }
        }
        return "null";
    }

    [[nodiscard]] bool scalar_array() const {
        if (kind_ != Kind::Arr) return false;
        return std::none_of(arr_.begin(), arr_.end(), [](const J& j) { return j.is_container(); });
    }

    // Short containers (<= 240 chars) stay on one line; longer ones break one child per line (long number lists wrap at 12 per line).
    void pretty(std::string& out, int ind) const {
        if (!is_container()) { out += compact(); return; }
        std::string c = compact();
        if (c.size() <= 240 || (kind_ == Kind::Arr && arr_.empty()) || (kind_ == Kind::Obj && obj_.empty())) { out += c; return; }
        const std::string pad(static_cast<size_t>(ind) + 2, ' ');
        if (scalar_array()) {
            out += "[\n";
            for (size_t i = 0; i < arr_.size(); ++i) {
                if (i % 12 == 0) out += pad;
                out += arr_[i].compact();
                if (i + 1 < arr_.size()) out += (i % 12 == 11) ? ",\n" : ",";
            }
            out += "\n" + std::string(static_cast<size_t>(ind), ' ') + "]";
            return;
        }
        if (kind_ == Kind::Arr) {
            out += "[\n";
            for (size_t i = 0; i < arr_.size(); ++i) {
                out += pad;
                arr_[i].pretty(out, ind + 2);
                out += (i + 1 < arr_.size()) ? ",\n" : "\n";
            }
            out += std::string(static_cast<size_t>(ind), ' ') + "]";
        } else {
            out += "{\n";
            for (size_t i = 0; i < obj_.size(); ++i) {
                out += pad + quote(obj_[i].first) + ":";
                obj_[i].second.pretty(out, ind + 2);
                out += (i + 1 < obj_.size()) ? ",\n" : "\n";
            }
            out += std::string(static_cast<size_t>(ind), ' ') + "}";
        }
    }
};

template <class T>
J jarr(const std::vector<T>& v) {
    J a = J::array();
    for (const T& x : v) a.push(J(x));
    return a;
}

// ---------------------------------------------------------------------------------------------------------------------
// Envelope
// ---------------------------------------------------------------------------------------------------------------------
inline std::string compiler_string() {
    std::string c;
#if defined(__clang__)
    c = "clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    c = "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#else
    c = "unknown compiler";
#endif
#if defined(_LIBCPP_VERSION)
    c += " (libc++ " + std::to_string(_LIBCPP_VERSION) + ")";
#elif defined(_GLIBCXX_RELEASE)
    c += " (libstdc++ " + std::to_string(_GLIBCXX_RELEASE) + ")";
#endif
    return c;
}

// The header every trace JSON starts with. `kind` is the topic ("autograd", "why", "optim", "linear"), `lesson` the first
// lesson that uses the file, `generator` the source file that wrote it. Callers append their own top-level keys.
inline J envelope(const std::string& kind, const std::string& lesson, const std::string& generator) {
    J lib = J::object();
    lib.set("ref", TORC_TRACE_REF).set("compiler", compiler_string()).set("flags", TORC_TRACE_FLAGS);
    J j = J::object();
    j.set("schema", "torc-trace/1").set("library", std::move(lib)).set("generator", generator).set("kind", kind).set("lesson", lesson);
    return j;
}

// ---------------------------------------------------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------------------------------------------------
inline torc::Tensor make_tensor(const std::vector<int>& shape, const std::vector<float>& values) {
    torc::Tensor t(shape);
    if (static_cast<size_t>(t.numel()) != values.size()) throw std::runtime_error("make_tensor: value count does not match shape");
    for (size_t i = 0; i < values.size(); ++i) t.data()[i] = values[i];
    return t;
}

inline std::vector<float> values_of(const torc::Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

// {"shape":[...],"values":[...]} — the node-value format of the schema.
inline J tensor_json(const torc::Tensor& t) {
    J o = J::object();
    o.set("shape", jarr(t.shape())).set("values", jarr(values_of(t)));
    return o;
}

inline float max_abs_diff(const torc::Tensor& a, const torc::Tensor& b) {
    if (a.shape() != b.shape()) return INFINITY;
    float m = 0.0f;
    for (int i = 0; i < a.numel(); ++i) m = std::fmax(m, std::fabs(a.data()[i] - b.data()[i]));
    return m;
}

// Bitwise equality of shape and every element (what "equal" means for the parity checks).
inline bool tensors_identical(const torc::Tensor& a, const torc::Tensor& b) { return a.shape() == b.shape() && a == b; }

// ---------------------------------------------------------------------------------------------------------------------
// Graph: owns Variables at stable addresses (the tape stores raw Variable*), names them, remembers the op that made them
// ---------------------------------------------------------------------------------------------------------------------
struct Node {
    std::string name, op, role;  // role: "leaf" | "op"
    torc::Variable* var;
};

struct Graph {
    std::deque<torc::Variable> vars;
    std::vector<Node> nodes;
    std::map<const torc::Variable*, std::string> names;

    torc::Variable& leaf(const std::string& name, torc::Tensor t, bool requires_grad) {
        vars.emplace_back(std::move(t), requires_grad);
        register_node(name, "leaf", "leaf", vars.back());
        return vars.back();
    }
    // Takes ownership of the result of a real library op and registers it under `name`/`op_name`.
    torc::Variable& op(const std::string& name, const std::string& op_name, torc::Variable v) {
        vars.emplace_back(std::move(v));
        register_node(name, op_name, "op", vars.back());
        return vars.back();
    }
    // Registers a Variable that lives elsewhere (it must outlive the Graph).
    void external(const std::string& name, const std::string& op_name, const std::string& role, torc::Variable& v) {
        register_node(name, op_name, role, v);
    }
    void register_node(const std::string& name, const std::string& op_name, const std::string& role, torc::Variable& v) {
        nodes.push_back({name, op_name, role, &v});
        names[&v] = name;
    }
    [[nodiscard]] torc::Variable* find(const std::string& name) const {
        for (const auto& n : nodes)
            if (n.name == name) return n.var;
        return nullptr;
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Scene: written once as a builder over the leaf values, so the same code runs for the traced pass and for the
// finite-difference evaluations.
// ---------------------------------------------------------------------------------------------------------------------
struct Scene {
    std::string id, title, expression, note;
    std::vector<std::pair<std::string, torc::Tensor>> leaves;  // name -> value (requires_grad unless listed in `constants`)
    std::vector<std::string> constants;                        // leaf names with requires_grad=false
    // Builds the graph; returns the scalar root, which must be the last node registered.
    std::function<torc::Variable&(Graph&, std::map<std::string, torc::Variable*>&)> build;
    float fd_h = 1e-2f;   // central-difference step (the project's calibrated setting, docs/DESIGN.md "Gradient checking")
    double fd_atol = 0.01;
};

inline bool is_const(const Scene& s, const std::string& n) {
    return std::find(s.constants.begin(), s.constants.end(), n) != s.constants.end();
}

inline torc::Variable& build_scene(const Scene& s, Graph& g, const std::vector<torc::Tensor>& values) {
    std::map<std::string, torc::Variable*> L;
    for (size_t i = 0; i < s.leaves.size(); ++i) {
        const auto& n = s.leaves[i].first;
        L[n] = &g.leaf(n, values[i], !is_const(s, n));
    }
    return s.build(g, L);
}

inline float eval_loss(const Scene& s, const std::vector<torc::Tensor>& values) {
    Graph g;
    torc::Variable::NoGradGuard ng;
    torc::Variable& root = build_scene(s, g, values);
    return root.data().data()[0];
}

// Result of tracing a scene: the JSON object for scenes[] plus the raw tensors by node name (for parity checks etc.).
struct SceneResult {
    J json;
    std::map<std::string, torc::Tensor> value;  // every node
    std::map<std::string, torc::Tensor> grad;   // nodes with has_grad() after backward
    std::map<std::string, std::vector<std::string>> inputs;  // node -> input names read from the real tape
};

inline SceneResult run_scene(const Scene& s) {
    std::vector<torc::Tensor> init;
    for (auto& l : s.leaves) init.push_back(l.second);

    Graph g;
    torc::Variable& root = build_scene(s, g, init);

    // Edges are read from the real tapes BEFORE backward(): backward() clears the tape of every visited Variable.
    std::map<const torc::Variable*, std::vector<std::string>> edges;
    for (auto& n : g.nodes)
        for (auto& e : n.var->tape())
            for (auto* in : e.inputs) {
                auto it = g.names.find(in);
                if (it == g.names.end()) throw std::runtime_error("trace: tape input of '" + n.name + "' is not a registered node");
                edges[n.var].push_back(it->second);
            }

    root.backward();

    SceneResult res;
    J out = J::object();
    out.set("id", s.id).set("title", s.title).set("expression", s.expression).set("note", s.note);
    out.set("root", g.names[&root]).set("root_has_grad", root.has_grad());
    J nodes = J::array();
    for (const Node& n : g.nodes) {
        const torc::Variable& v = *n.var;
        J nj = J::object();
        nj.set("name", n.name).set("op", n.op).set("role", n.role).set("requires_grad", v.requires_grad());
        J ins = J::array();
        for (auto& in : edges[n.var]) ins.push(in);
        nj.set("inputs", std::move(ins)).set("value", tensor_json(v.data()));
        nj.set("grad", v.has_grad() ? tensor_json(v.grad()) : J());
        nodes.push(std::move(nj));
        res.value.emplace(n.name, v.data());
        if (v.has_grad()) res.grad.emplace(n.name, v.grad());
        res.inputs[n.name] = edges[n.var];
    }
    out.set("nodes", std::move(nodes));

    // Finite-difference check on every requires-grad leaf (central differences).
    const float h = s.fd_h;
    J fd_leaves = J::array();
    for (size_t li = 0; li < s.leaves.size(); ++li) {
        const auto& name = s.leaves[li].first;
        if (is_const(s, name)) continue;
        torc::Variable* lv = g.find(name);
        std::vector<float> fd, an;
        float max_err = 0.0f;
        for (int k = 0; k < init[li].numel(); ++k) {
            auto plus = init, minus = init;
            plus[li].data()[k] += h;
            minus[li].data()[k] -= h;
            float num_g = (eval_loss(s, plus) - eval_loss(s, minus)) / (2 * h);
            float ana_g = lv->has_grad() ? lv->grad().data()[k] : 0.0f;
            max_err = std::fmax(max_err, std::fabs(num_g - ana_g));
            fd.push_back(num_g);
            an.push_back(ana_g);
        }
        J lj = J::object();
        lj.set("name", name).set("numerical", jarr(fd)).set("analytic", jarr(an)).set("max_abs_err", max_err)
            .set("ok", max_err <= static_cast<float>(s.fd_atol));
        fd_leaves.push(std::move(lj));
    }
    J fdj = J::object();
    fdj.set("h", h).set("atol", s.fd_atol).set("leaves", std::move(fd_leaves));
    out.set("finite_difference", std::move(fdj));
    res.json = std::move(out);
    return res;
}

// Name of the exception class for error tables ("ShapeError", "TorcError", ...), plus its what() text.
struct Caught {
    std::string type, what;  // type "none" (and empty what) when the call did not throw
};

inline Caught catch_call(const std::function<void()>& f) {
    try {
        f();
    } catch (const torc::ShapeError& e) {
        return {"ShapeError", e.what()};
    } catch (const torc::TorcError& e) {
        return {"TorcError", e.what()};
    } catch (const std::exception& e) {
        return {"std::exception", e.what()};
    }
    return {"none", ""};
}

}  // namespace trace
