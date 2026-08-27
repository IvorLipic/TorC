# torc Autograd — Design Notes (Tape-Based)

*This document is the source of truth for the tape structure and backward algorithm as
implemented. `docs/DESIGN.md` covers the broader "why" and defers here for the mechanics —
if the two ever describe different algorithms again, this file wins for anything about tape
structure or the backward walk.*

---

## Why Tape-Based (Wengert List)?

Two main approaches exist for automatic differentiation:

| Approach | How it works | Complexity |
|----------|--------------|------------|
| **Tape-based** (Wengert list) | Record every op during forward pass in a flat list. Walk list in reverse for backward. | Low |
| **Expression-tree** | Each Variable points to parent nodes. Backward does topological sort + recursion. | High |

**We chose tape-based** because:
1. **Simplicity** — ~200 lines for core engine, no node hierarchy
2. **Sufficient for torc's scope** — toy models, small datasets (Milestone 5)
3. **Matches PyTorch eager** — same mental model
4. **Evolvable** — can swap to expression-tree later without changing Variable API

---

## Core Concepts

### Variable
```cpp
Variable x(2.0f, true);   // scalar convenience ctor: data=2.0, requires_grad=true
// x.data()          -> const Tensor&  (forward value)
// x.grad()           -> const Tensor& (gradient; only meaningful once has_grad() is true)
// x.requires_grad()   -> bool
// x.has_grad()         -> bool  (false until backward() has actually accumulated something)
// x.backward()          -> computes gradients (Variable must be scalar, i.e. numel()==1,
//                          unless an explicit grad_output is passed to the other overload)
```

### Tape (per-Variable)
Each Variable carries its own tape. During forward, ops append a `TapeEntry` to the
**output** Variable's tape. Each entry stores:
- A `backward` closure that knows how to compute gradients for its inputs
- Pointers to the input Variables (for graph traversal)

```cpp
struct TapeEntry {
    std::vector<Variable*> inputs;  // input Variables (for topological sort)
    std::function<void(const Tensor& grad_output, std::vector<Tensor>& input_grads)> backward;
};
```

### Backward Algorithm: Topological Sort (Not Recursion)

**Critical design decision:** The backward pass uses **topological sort + iterative execution**,
not recursive depth-first traversal. This matches how PyTorch, TinyTorch (Harvard), and all
production autograd engines work.

**Why not recursion?**
- Recursion stacks overflow on deep graphs (1000+ layer networks)
- Recursion double-visits shared subgraphs (DAGs, not just trees)
- Recursion makes gradient accumulation order-dependent

**The algorithm:**
1. **DFS post-order traversal** from the output Variable, following `TapeEntry.inputs` pointers
2. **Reverse** the resulting list to get backward execution order
3. **Iterate** through reversed list, executing each Variable's tape entries once
4. Gradients accumulate via `+=` in a map keyed by `Variable*`

```cpp
void Variable::backward_with_grad(const Tensor& upstream_grad) {
    if (tape_.empty()) return;

    // 1. Build topological order (DFS post-order over TapeEntry.inputs)
    std::vector<Variable*> topo = build_topo();
    std::reverse(topo.begin(), topo.end());  // backward order

    // 2. Gradient accumulation map
    std::unordered_map<Variable*, Tensor> grad_map;
    grad_map.emplace(this, upstream_grad);

    // 3. Execute backward in topological order
    for (Variable* v : topo) {
        if (v->tape_.empty()) continue;

        auto it = grad_map.find(v);
        if (it == grad_map.end()) continue;  // no gradient reached this node

        Tensor current_grad = it->second;

        // Walk this Variable's tape entries in reverse (most recent op first)
        for (auto eit = v->tape_.rbegin(); eit != v->tape_.rend(); ++eit) {
            const TapeEntry& entry = *eit;

            // Tensor has no default constructor, so placeholders need an explicit shape —
            // std::vector<Tensor> input_grads(n) will NOT compile.
            std::vector<Tensor> input_grads;
            input_grads.reserve(entry.inputs.size());
            for (size_t i = 0; i < entry.inputs.size(); ++i)
                input_grads.push_back(Tensor(std::vector<int>{1}));

            entry.backward(current_grad, input_grads);

            for (size_t i = 0; i < entry.inputs.size(); ++i) {
                Variable* input = entry.inputs[i];
                if (!input->requires_grad_) continue;

                // Broadcasting reduction happens HERE, centrally — never inside entry.backward
                Tensor input_grad = input_grads[i];
                if (input_grad.shape() != input->data_.shape())
                    input_grad = reduce_sum_to_shape(input_grad, input->data_.shape());

                input->accumulate_grad(input_grad);
                auto insert_result = grad_map.emplace(input, input_grad);
                if (!insert_result.second) {
                    insert_result.first->second = insert_result.first->second.add(input_grad);
                }
            }

            current_grad = input_grads.back();  // pass to next tape entry, if any
        }
    }

    tape_.clear();
}
```

**Why this is correct:**
- Each Variable is processed exactly once (no double-visiting)
- Gradients are computed in dependency order (all downstream grads ready before upstream needs them)
- Gradients for shared inputs are accumulated via `+=` in `accumulate_grad` and in `grad_map`
  (see note below)
- Stack usage for the DFS is a separate concern from the O(V) *iterative* backward walk above —
  `build_topo()` itself is currently still recursive (see Common Pitfalls)

> **Note on `grad_map` accumulation for shared subgraphs:** The current implementation
> accumulates gradients in `grad_map` via `grad_map.emplace(input, grad)` followed by
> `insert_result.first->second = insert_result.first->second.add(grad)` when the key already
> exists (i.e. `+=`). Combined with `input->accumulate_grad(grad)` on the Variable itself
> (also `+=`), this correctly handles diamond dependencies where an intermediate Variable is
> consumed by multiple downstream ops. The diamond-dependency test in
> `tests/test_autograd.cpp` (`GradMapAccumulation.DiamondDependencyAccumulates`) verifies
> that shared subgraphs accumulate gradients correctly.

**Sources:**
- PyTorch dev mailing list: "Simplified Introduction to PyTorch's Autograd" (zdevito, 2021)
- PyTorch autograd source: `torch/csrc/autograd/engine.cpp` (topological_sort + iterative execution)
- TinyTorch Module 06 (Harvard): recursive for educational simplicity, but notes production uses topological sort
- Red Hat Developer: "Optimize PyTorch training with the autograd engine" (2026) — explicit C++ pseudocode

---

## Forward to Backward Flow

```cpp
// User code
Variable x(2.0f, true);
Variable y(3.0f, true);
Variable z = torc::mul(x, y);   // Forward: 2 * 3 = 6
z.backward();                    // Backward: dz/dx = 3, dz/dy = 2
```

**Forward pass** (`torc::mul(x, y)`):
1. `Tensor::mul` computes `6.0`
2. `torc::mul` (a free function — ops are never `Variable` methods) wraps the result in a new `Variable`
3. Creates a `TapeEntry`:
   - `inputs = {&x, &y}` (raw pointers into `x` and `y` themselves)
   - The backward closure captures **copies** of `x.data()` and `y.data()` (not references to
     `x`/`y`) and computes `input_grads[0] = grad_output.mul(y_data)`,
     `input_grads[1] = grad_output.mul(x_data)`
4. Appends the entry to `z`'s `tape_`
5. Returns `z` with `data_=6.0`, `tape_=[entry]`

**Backward pass** (`z.backward()`):
1. Build topological order: `[x, y, z]` (post-order DFS)
2. Reverse: `[z, y, x]`
3. Process `z`: execute its tape entry, get `grad_x=3`, `grad_y=2`
4. Accumulate into `x`'s and `y`'s `grad_` (`has_grad_` becomes `true` for both)
5. Process `x` and `y`: both have empty tapes (they're leaves), no-op
6. Done — `z.tape_` is cleared

---

## Broadcasting Gradients (Critical)

**Forward**: `x (3,) + y (1,)` → `z (3,)` — `y` broadcast to `(3,)`
**Backward**: `grad_out (3,)` → `x.grad (3,)`, `y.grad (1,)` — must **sum** over the broadcast dim

`Tensor` has no `unsqueeze()` and no keepdim option on `sum(axis)`, so the real
`reduce_sum_to_shape` (in `src/autograd.cpp`) works around that directly:

```cpp
Tensor reduce_sum_to_shape(const Tensor& grad, const std::vector<int>& target_shape) {
    Tensor result = grad;
    int g_rank = (int)result.shape().size();
    int t_rank = (int)target_shape.size();
    int offset = g_rank - t_rank;

    // Leading dims that don't exist in target_shape at all: repeated sum(0)
    for (int i = 0; i < offset; ++i)
        result = result.sum(0);

    // Interior dims where target_shape[i] == 1 but result still has size > 1:
    // sum() removes the dim, then reshape() re-inserts it as size 1
    for (int i = 0; i < t_rank; ++i) {
        if (target_shape[i] == 1 && result.shape()[i] > 1) {
            result = result.sum(i);
            std::vector<int> new_shape = result.shape();
            new_shape.insert(new_shape.begin() + i, 1);
            result = result.reshape(new_shape);
        }
    }
    return result;
}
```

**This is called exactly once per input, centrally, in `Variable::backward_with_grad`** — see
the backward-algorithm code block above. It is *not* called inside individual ops' backward
closures. An op's backward closure (e.g. `add`'s) always returns gradients at the *output's*
shape; it never checks whether broadcasting happened or calls `reduce_sum_to_shape` itself.
Don't add that call inside a new op's closure — it would be redundant with, not a substitute
for, the centralized step.

---

## Gradient Functions for Key Ops

### Elementwise: z = x * y
```
dz/dx = y,  dz/dy = x
input_grads[0] = grad_output * y
input_grads[1] = grad_output * x
```

### Matmul: z = x @ y  (x: m×k, y: k×n)
```
dz/dx = grad_output @ y.T    (m×n @ n×k = m×k)
dz/dy = x.T @ grad_output    (k×m @ m×n = k×n)
```
This generalizes to batched matmul: the 2D formulas are applied per-batch, then any broadcast
*batch* dimensions are summed out via `reduce_sum_to_shape` on each input's gradient before
accumulation. The implementation uses `swap_last_two_axes` to transpose the last two dimensions
of `a`/`b`, then calls the existing `Tensor::matmul`, and finally applies `reduce_sum_to_shape`
centrally in `backward_with_grad` — same pattern as elementwise ops.

### Sum reduction: z = x.sum(axis=0)  (x: m×n to z: n,)
```
dz/dx = broadcast(grad_output, x.shape)  // repeat grad_output m times
```

### Transpose: z = x.transpose(axes)
```
dz/dx = grad_output.transpose(inverse_of(axes))
```
The inverse permutation is computed explicitly — don't re-apply the same `axes` unless it's
a true involution (e.g. the default full-reversal case on even-rank tensors, or `{1,0}` on
rank-2). A general permutation like `{2,0,1}` on rank-3 needs its actual inverse `{1,2,0}`.

### Reshape/View: z = x.reshape(new_shape)
```
dz/dx = grad_output.reshape(original_shape)
```
Note this is **not** metadata-only / zero-cost: `Tensor::reshape()` does
`out.storage_ = storage_;` on a `const` `this`, which is a real copy of the underlying
`std::vector<float>`, not a move. Backward correctness is unaffected either way, but don't
describe this (here or elsewhere) as a free operation — it isn't.

### Slice: z = x.slice(slices)
```
dz/dx = zero_fill(x.shape); dz/dx[offset + slice_indices] = grad_output
```
The backward closure zero-fills a tensor of the original shape, then scatters `grad_output`
values into the sliced region. This is implemented with inline odometer-style index
incrementing over `grad_output.shape()`, computing the corresponding flat index in the
original tensor from the stored slice offsets.

### Max/Min: z = x.max() / x.min()
```
dz/dx = zero_fill(x.shape); dz/dx[argmax] = grad_output
dz/dx = zero_fill(x.shape); dz/dx[argmin] = grad_output
```
Forward computes the max/min value only; the backward closure finds the argmax/argmin by
scanning `x.data()` and scatters `grad_output` to that position. For axis-wise reductions
(`max(axis)` / `min(axis)`), the scan is per-output-element along the reduced axis, matching
the iteration order of `Tensor::reduce_axis`. In both cases ties are broken by first-occurrence,
matching `std::ranges::max` / `std::ranges::min` used in the forward pass.

---

## requires_grad Semantics

```cpp
Variable a(1.0f, true);              // tracked
Variable b(2.0f, false);             // NOT tracked
Variable c = torc::add(a, b);        // c.requires_grad() == true (a requires it)
c.backward();                        // a.grad() populated, b.grad() NOT allocated
```

- **Default**: `requires_grad = false` (inference-friendly)
- **Propagation**: If any input requires_grad, output `requires_grad = true`
- **No tape recorded** for non-requires-grad inputs to avoid gradient allocation

---

## Gradient Accumulation

```cpp
loss1.backward();  // W.grad() += dL1/dW
loss2.backward();  // W.grad() += dL2/dW  (accumulates!)
W.zero_grad();      // Reset before next step
```

Matches PyTorch behavior. Call `zero_grad()` at the start of each training step.

---

## Gradient Checking (Testing)

Every differentiable op must have a gradient check test:

```cpp
// Numerical gradient (central difference)
float h = 1e-4f;  // for float32
float numerical_grad = (f(x + h) - f(x - h)) / (2 * h);

// Analytical gradient from autograd
float analytical_grad = x.grad().data()[i];

// Compare
EXPECT_NEAR(numerical_grad, analytical_grad, 1e-2f);   // GRAD_ATOL in test_autograd.cpp
```

Run with `ctest` — catches bugs in backward implementations.

**Tolerance rationale**: float32 has ~7 decimal digits. Central-difference truncation error is
`O(h²) ≈ O(1e-8)` for `h=1e-4` in exact arithmetic, but in practice float32 finite differences
on chained/elementwise ops show errors around `1e-3` to `1e-2` from repeated forward-pass
rounding. The test suite's actual `GRAD_ATOL = 1e-2f` is calibrated against that observed
noise, not against the theoretical `O(h²)` bound alone — using `1e-4` here, as an earlier
draft of this doc suggested, is tighter than the current tests actually run at.

---

## Detaching and Disabling Gradients

### `detach()`

```cpp
Variable d = c.detach();   // new Variable, same data, requires_grad=false, empty tape
```

`detach()` returns a new `Variable` with a copy of the data (Tensor's copy ctor deep-copies
storage_) but with `requires_grad=false` and an empty tape. Backward on `d` is a no-op, and
gradients do not flow back through `d` into `c`'s ancestors. This matches PyTorch's
`Tensor.detach()`.

### `set_grad_enabled(bool)` / `grad_enabled()`

```cpp
Variable::set_grad_enabled(false);
Variable c = torc::add(a, b);   // c.requires_grad() == false, tape is empty
Variable::set_grad_enabled(true);
```

A global flag checked by arithmetic ops at the start of forward. When disabled, those ops still
compute their forward output, but they return plain `Variable`s with `requires_grad=false` and
empty tapes — no `TapeEntry` is recorded. Defaults to `true`. This is useful for inference
sections of a training loop without manually detaching each intermediate.

**Note**: `set_grad_enabled(false)` does *not* change the `requires_grad` flag on existing
`Variable`s. Inputs keep their original `requires_grad` value; only the *new* outputs produced
while disabled are non-tracked.

---

## Extending with New Operations

Ops are **free functions in `namespace torc`**, declared in `autograd.hpp` and defined in
`autograd.cpp` — never `Variable` methods. To add gradient support for a new `Tensor` op:

1. **Add the `Tensor` op first** (if it doesn't exist): `Tensor::my_op(...) const`
2. **Declare the free function** in `autograd.hpp`:
   ```cpp
   Variable my_op(const Variable& a /*, other args */);
   ```
3. **Define it** in `autograd.cpp`, following the existing pattern (see `add`/`mul`/`neg`):
   ```cpp
   Variable my_op(const Variable& a) {
       bool needs_grad = a.requires_grad_;
       Tensor out_data = a.data_.my_op();
       Variable out(std::move(out_data), needs_grad);

       if (needs_grad) {
           TapeEntry entry;
           // const_cast is needed because Variable& params are const but TapeEntry.inputs
           // needs non-const pointers to later call accumulate_grad() on them
           entry.inputs = { const_cast<Variable*>(&a) };

           // Capture whatever the local derivative needs BY VALUE (a copy), the same way
           // mul()/div() capture a_data/b_data — never capture a reference to `a` itself.
           Tensor a_data = a.data();
           entry.backward = [a_data](const Tensor& grad_output, std::vector<Tensor>& input_grads) {
               input_grads[0] = /* local derivative of my_op, applied to grad_output */;
               // Return the gradient at the OUTPUT's shape — do not call
               // reduce_sum_to_shape here, that happens centrally (see "Broadcasting
               // Gradients" above).
           };
           out.tape_.push_back(std::move(entry));
       }
       return out;
   }
   ```
4. **Add a gradient check test** in `test_autograd.cpp`, following the existing `GradCheck*`
   tests (hand-computed example + at least one broadcast case if the op can broadcast).

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Forgetting `reduce_sum_to_shape` for broadcast ops | It's applied centrally in `backward_with_grad` — don't add a second call inside your op's closure, and don't forget the input actually needs it accounted for by the time your closure returns a shape-correct-or-broadcastable grad |
| Calling `reduce_sum_to_shape` inside a backward closure | Almost never needed — it is applied centrally in `backward_with_grad` for elementwise ops. The only current exception is `matmul`, whose backward closure calls it for both inputs because batched matmul produces gradients that may have extra batch dimensions from broadcasting. If you think you need it inside a new op's closure, document the exception explicitly in `docs/AUTOGRAD.md` |
| In-place modification of `Variable::data()` | Don't do it — breaks gradient computation. Use the guarded API (`Variable::fill`) instead; direct `data()`/`operator[]` writes remain unguarded and are the user's responsibility |
| Double `backward()` without `zero_grad()` | Expected accumulation; call `zero_grad()` each step |
| `backward()` on non-scalar | Call `.sum().backward()` or pass an explicit `grad_output` |
| Recursive backward on deep graphs | The backward *walk* uses topological sort (not recursion) — but `build_topo()` itself is still a recursive DFS today, so very deep graphs can still overflow the stack during topo-sort construction, just not during backward execution |
| Circular references with `shared_ptr` | Not applicable — `Variable` owns data directly; tape entries hold raw, non-owning pointers instead. That trades the cycle risk for a *lifetime* risk: every `Variable` in a graph must outlive `backward()` on any of its descendants (see `docs/DESIGN.md`) |
| Assuming `Tensor` is default-constructible (e.g. `std::vector<Tensor> v(n)`) | It isn't — `Tensor` has no default constructor. Construct placeholders with an explicit shape, e.g. `Tensor(std::vector<int>{1})`, as `backward_with_grad` does |
| Assuming `reshape()`/`view()` are "metadata-only" or free | They're not — `Tensor::reshape()` copies `storage_` on every call. Correctness is unaffected, but don't rely on this being zero-cost |

---

## Future Evolution

| Feature | When | Approach |
|---------|------|----------|
| Gradient checkpointing | Milestone 6 | checkpoint(fn, *args) — recompute forward in backward |
| Higher-order derivatives | Later | Nested tapes (Engine per order) |
| Graph optimization/compilation | Not planned | Would need expression-tree |
| Distributed autograd | Not planned | Out of scope |
| Iterative (non-recursive) `build_topo()` | Whenever deep-graph stack safety actually matters | Explicit stack instead of a recursive `std::function` DFS |

---

## References

- docs/DESIGN.md — Architectural decisions
- ROADMAP.md — Milestone 4 checklist
- PyTorch autograd internals: https://pytorch.org/docs/stable/notes/autograd.html
- PyTorch dev mailing list: "Simplified Introduction to PyTorch's Autograd" (zdevito, 2021)
- TinyTorch Module 06 (Harvard): https://mlsysbook.ai/tinytorch/modules/06_autograd.html
- Red Hat Developer: "Optimize PyTorch training with the autograd engine" (2026)
- Micrograd (educational): https://github.com/karpathy/micrograd