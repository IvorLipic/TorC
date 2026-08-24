# torc — Design Notes

This file holds decisions that are expensive to reverse and the rationale behind them,
plus the planned future project structure once ML work (autograd → nn → optim → data)
starts. Update this *before* writing code that implements a new architectural layer, not
after — the point is to force the decision to be explicit.

---

## Decisions made so far

### dtype strategy
Single dtype, hardcoded `float32`. Staying single-dtype is the simplest option and matches
the "naive-but-correct" goal through Milestone 3. Revisit with a dtype enum or templated
storage (`Tensor<T>`) only if multi-precision becomes an actual need — don't add it
speculatively, since it touches every op signature.

### Shape / broadcasting semantics
Elementwise ops use NumPy-style broadcasting via the `broadcast_shape()` helper in
`utils.hpp`. Shapes are aligned from the right; dimensions are compatible if they are equal
or one of them is 1. The result shape is the maximum along each dimension. This was
implemented in Milestone 2 before autograd, because broadcasting changes what a gradient's
"sum over broadcast dims" step needs to look like.

Storage is always row-major and contiguous. `reshape()` copies the shape vector and moves
storage; `view()` delegates to `reshape()`. There is no stride metadata yet, so `transpose()`
and `slice()` explicitly reorder/copy into new contiguous buffers.

### Memory ownership
`storage_` is an owned `std::vector<float>` per `Tensor`, copied fresh on every elementwise
or reduction op. Shape-changing ops (`reshape`/`view`) move `storage_` instead of copying,
so they are O(1) with respect to data size. Fine for a naive reference implementation.
**This gets more nuanced for autograd** — see below.

### Linear algebra — matmul (Milestone 3)

`Tensor::matmul(const Tensor&)` is the first non-elementwise op. Decisions, recorded here so
they are not re-litigated:

- **Operand rank >= 2 required.** A `ShapeError` is thrown for rank 0/1 operands. This is
  deliberately *forward-compatible*: allowing 1D promotion later only *widens* accepted input,
  so it cannot break existing callers, and the shape checks stay the same. Milestone 5's
  `nn::Linear` only ever needs rank >= 2, so 1D/dot-product support is not an early need.
- **Batch broadcasting reuses `broadcast_shape()`.** The last two dims are the matrix
  (contracted on the second of `self` and first of `other`); everything before them is the
  batch prefix and is broadcast with the same helper already used for elementwise ops. This
  honors the AGENTS.md rule against ad-hoc per-op shape coercion. Batch-broadcast failures are
  caught and re-thrown with the **full operand shapes** in the message, since
  `broadcast_shape()` would otherwise only print the prefix subspans.
- **Zero-size dimensions are allowed and defined as zeros.** `(2,0) @ (0,3)` yields a `(2,3)`
  of zeros; `(0,3) @ (3,4)` yields an empty `(0,4)`. The contraction is an empty sum, so the
  loops already produce the right answer — no special-casing, and it matches shapes that are
  legal at construction.
- **Naive triple-loop, `float` accumulator.** Complexity is `m * n * k` with `acc` of type
  `float`. Loop order is `i, j, k` (`k` innermost). Optimizing the loop order for cache
  locality, or adding SIMD/threads, is explicitly deferred to Milestone 6 and must be
  benchmark-gated. The `float` (not `double`) accumulator keeps parity with the
  `cblas_sgemm`-based BLAS backend, so naive and BLAS results agree to the same precision.

**BLAS backend — landed, not deferred.** `src/matmul_blas.cpp` provides a `TORC_USE_BLAS`-gated
replacement for `Tensor::matmul` that calls `cblas_sgemm` per batch element, copying strided
batch slices into contiguous scratch buffers first (CBLAS expects contiguous row-major
operands). `CMakeLists.txt` auto-detects OpenBLAS, falls back to generic `find_package(BLAS)`,
and falls back again to `Accelerate.framework` on macOS; it's meant to fail configure with a
clear message if none provide `cblas_sgemm`.

### Error handling convention
`TorcError` (base, derives `std::runtime_error`) and `ShapeError` (shape mismatches) live in
`utils.hpp`. All error sites use `ShapeError`; existing `catch (std::runtime_error&)` still
works since it's a base class.

---

## Autograd — refined design and implementation plan (Milestone 4)

> **Status note:** live checklist state (what's actually done) lives in ROADMAP.md's
> Milestone 4 section — treat that as the single source of truth. This section explains the
> *why* behind the design and isn't updated per-step, so step numbers below describe scope,
> not completion.

### What "tape-based" actually means here

The term is used in two distinct ways in the literature (see arxiv:1810.11530):

- **Source-transformation tape** (ADIFOR, Tapenade): a global stack that records raw
  intermediate values; the adjoint program reads it. Not relevant to C++ operator-overloading.
- **Operator-overloading tape** (PyTorch, Chainer, Autograd): each `Variable` carries a
  record of the op that produced it. The chain of these records *is* the tape.

Each op's backward closure holds the inputs it needs for its local derivative (the "saved
tensors" pattern, implemented as values captured by the `std::function`). Backward is a walk
over these per-output tape entries — see the next section for how that walk is actually
structured.

### Ownership model — decided

`Variable` **owns** its `Tensor data_` directly (not `shared_ptr`). Backward closures capture
**copies** of any `Tensor` values they need for their local derivative (e.g. `mul`'s backward
closure captures `a_data`/`b_data` by value). This is intentionally naive: it costs O(N) per
saved tensor but keeps the code simple and correct. Revisit when memory pressure actually
matters (Milestone 6+).

This is a narrower guarantee than "the whole graph stays alive" — see the lifetime
consequence called out at the end of the next section, which this alone does not cover.

### Tape / graph structure — as implemented

An earlier draft of this document proposed a design where each `TapeEntry` held no reference
back to its input `Variable`s at all — just a self-contained closure that would recursively
call `input.backward(local_grad)` on its own inputs. **That is not what got built, and should
not be read as describing the current code.** The implemented design (in
`include/torc/autograd.hpp`, detailed in `docs/AUTOGRAD.md`) does an explicit graph traversal
instead:

```cpp
struct TapeEntry {
    std::vector<Variable*> inputs;   // raw, non-owning pointers into the input Variables
    std::function<void(const Tensor& grad_output, std::vector<Tensor>& input_grads)> backward;
};
```

`Variable::backward()`:
1. Builds a full topological order via DFS post-order over `TapeEntry.inputs`, starting from
   the output.
2. Reverses it.
3. Walks the reversed order once, executing each Variable's tape entries and accumulating
   gradients into an `unordered_map<Variable*, Tensor>` keyed by input pointer.
4. Clears `tape_` on completion.

This was chosen over the simpler recursive design specifically because recursion can (a)
overflow the stack on deep graphs and (b) double-visit shared subgraphs (a Variable used as
input to two different ops). See `docs/AUTOGRAD.md` for the full algorithm, rationale, and
sources.

**Consequence for `Variable` lifetime.** Because `TapeEntry.inputs` holds raw pointers into
the *original* `Variable` objects — not copies — **every `Variable` that participates in a
graph must outlive `backward()` on any of that graph's outputs**, not just the final loss
Variable. Returning a `Variable` from a function whose local leaf `Variable`s then go out of
scope, and later calling `.backward()` on the result, is undefined behavior today. This should
be documented prominently for users (README/AGENTS) and/or closed off by a future design
change (e.g. `shared_ptr`-owned graph nodes) — right now it's neither.

### Broadcasting backward — must be explicit

When a broadcast happened during forward (e.g. shape `{3}` broadcast to `{2,3}` in an add),
the backward gradient has shape `{2,3}` but the input's grad must have shape `{3}`. The
intermediate must be **summed over the broadcast dimensions** before being written to the
input's `.grad_`.

Implementation: `reduce_sum_to_shape(grad, target_shape)`, defined in `src/autograd.cpp`. It
sums over leading dimensions introduced by broadcasting via repeated `sum(0)`, then for any
interior dimension where `target_shape[i] == 1` but the gradient's dim is larger, it does
`sum(i)` followed by a `reshape()` that re-inserts the size-1 dimension (`Tensor::sum(axis)`
has no `keepdim` option). This mirrors PyTorch's `_sum_to` in spirit but is written against
the concrete `Tensor` API actually available here.

Critically, **this reduction happens centrally, once**, in `Variable::backward_with_grad`,
comparing each computed `input_grad`'s shape against the input `Variable`'s actual shape —
not inside each op's own backward closure. Individual backward closures (`add`, `mul`, etc.)
return grads at the *output's* shape and never call `reduce_sum_to_shape` themselves. Keep
this pattern for every future op's backward closure; duplicating the reduction inside a
closure would be redundant with, not a substitute for, the centralized step.

**Test requirement**: every broadcast case must be verified with both a correctness test and
a gradient check. A missing unbroadcast is the most common autograd bug.

### View ops backward — implemented (Step 6)

| Op | Forward | Backward rule |
|---|---|---|
| `transpose(axes)` | permutes dims | `grad_input = grad_output.transpose(inverse_of(axes))` — **not** re-applying `axes`; only self-inverse for the default (full-reversal) case or true involutions. A general permutation needs its actual inverse computed. |
| `reshape(new_shape)` / `view(new_shape)` | copies storage to new shape | `grad_input = grad_output.reshape(original_shape)` |
| `slice(slices)` | copies a sub-region | zero-fill `grad_input` at the original shape, scatter `grad_output` into the sliced region — implemented inline in the backward closure using odometer-style index incrementing over `grad_output.shape()`, no new `Tensor` primitive needed. |

`Tensor` currently has no `unsqueeze`/`squeeze`. Since both are just a `reshape()` that
inserts or removes a size-1 dimension, their backward (and forward, if ever needed) can
likely be expressed directly via the existing `reshape()`/`view()` rather than new `Tensor`
primitives — revisit only if a standalone forward API turns out to be needed elsewhere.

### Non-differentiable ops — implemented with argmax tracking

`max()` / `min()` (with or without axis) now compute their backward by scanning the input
tensor data inside the backward closure to find argmax/argmin indices, then scattering
`grad_output` to those positions. Ties are broken by first-occurrence, matching
`std::ranges::max` / `std::ranges::min` used in the forward pass. No argmax indices are
stored on the Variable itself; the scan happens during backward.

**Step 9 — `max`/`min` backward with argmax tracking**
- Whole-tensor `max`/`min`: scan all elements, scatter scalar `grad_output` to the argmax/argmin
- Axis-wise `max(axis)` / `min(axis)`: scan per-output-element along the reduced axis using
  the same stride math as `Tensor::reduce_axis`, scatter `grad_output` values to the
  argmax/argmin positions
- **Test**: hand-computed correctness tests for whole-tensor and axis-wise variants, plus
  gradient checks on a `{2,3}` tensor with `max(0)` and `max(1)` (one-sided forward
  difference, since `max`/`min` are piecewise-linear and central difference gives 0.5 at the
  kink)

### Scalar vs. non-scalar backward

`backward()` on a non-scalar Variable is illegal without an explicit upstream gradient
argument, because there is no canonical `∂loss/∂loss = 1` seed. This is enforced today:
`Variable::backward()` throws `ShapeError` if `data_.numel() != 1`; the overload
`backward(const Tensor& grad_output)` additionally checks
`grad_output.numel() == data_.numel()`. This matches PyTorch's documented behavior and
prevents silent gradient-shape bugs.

### Gradient accumulation semantics

- `grad_` accumulates via `+=` on repeated `backward()` calls — this is intentional and
  required for mini-batch gradient accumulation.
- Leaf Variables (user-created, `requires_grad=true`) start with `has_grad_ == false` and no
  meaningful `grad_`.
- Call `zero_grad()` (resets `has_grad_` and `grad_`) before each training iteration.
- Intermediate (non-leaf) Variables also accumulate into their own `grad_` during the
  backward walk — the implementation does not currently distinguish "leaf" from
  "intermediate" for accumulation or retention purposes; there is no `retain_graph`/leaf-only
  concept yet.

### Gradient checking — methodology, not just a checkbox

For every differentiable op, verify analytical gradients against **central finite differences**:

```
h = 1e-4  (for float32)
grad_numerical[i] = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
max |grad_analytical - grad_numerical| < tol
```

Where to check:
- One hand-picked small example per op (e.g. `{3}` shape for elementwise, `{2,2}` for matmul)
- One broadcast case per op that uses broadcasting
- One batched case for `matmul`

**Tolerance**: `tests/test_autograd.cpp` uses `EPS = 1e-4f` and `GRAD_ATOL = 1e-2f`. Float32
has ~7 decimal digits of precision; central-difference truncation error is
`O(h²) ≈ O(1e-8)` for `h = 1e-4` in exact arithmetic, but in practice float32 finite
differences on chained/elementwise ops show errors around `1e-3` to `1e-2` from repeated
forward-pass rounding. `1e-2` is the tolerance actually calibrated against that observed
noise — it catches real bugs while tolerating float32 precision loss. If precision ever
changes (e.g. a `double` build), re-validate and likely tighten this.

### Incremental implementation steps

Each step = implement → hand-written correctness test → gradient check → commit. Do not
batch steps into a single PR. *(Check ROADMAP.md for actual checkbox state — not duplicated here.)*

**Step 1 — `Variable` scaffold** (`include/torc/autograd.hpp`)
- `data_`, `grad_`, `requires_grad_`, `has_grad_`, `tape_` members
- `Variable(Tensor, bool)`, `Variable(float, bool)` constructors
- `Tensor& data()`, `const Tensor& data() const`
- `bool requires_grad() const`, `bool has_grad() const`
- `void backward()` / `void backward(const Tensor&)` — throws `ShapeError` on non-scalar
  output without an explicit grad
- `void zero_grad()`

**Step 2 — Scalar-only autograd**
- Only rank-0 (`{1}`-shape) Variables, via dedicated `*_scalar` free functions
- Ops: `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar`, `neg_scalar`
- **Grad check**: central difference, per the methodology above, on each scalar op

**Step 3 — Broadcasting backward (elementwise tensor ops)**
- Reuse `Tensor::add/sub/mul/div` for forward pass, via free functions `add`, `sub`, `mul`,
  `div`, `neg`
- `reduce_sum_to_shape` applied centrally in `backward_with_grad`, not per-op (see
  "Broadcasting backward" above)
- **Grad check**: at least one broadcast case per op

**Step 4 — Reduction ops backward**
- `sum()` (no axis): backward = grad broadcast back to input shape
- `mean()`: backward = grad / numel, broadcast back
- `sum(axis)` / `mean(axis)`: backward needs the reduced dimension re-inserted before
  broadcasting back — likely via `reshape()`, mirroring how `reduce_sum_to_shape` already
  re-inserts size-1 dims
- **Test**: correctness on a `{2,3}` tensor, gradient check for whole-tensor and axis variants

**Step 4b — Defer `max`/`min` backward**
- Throw `ShapeError("max/min backward not yet implemented")`
- Document that argmax-tracking is needed before these can be supported

**Step 5 — `matmul` backward**
- Forward: reuse `Tensor::matmul`
- Backward 2D: `dA = dC @ B^T`, `dB = A^T @ dC`
- Backward batched: `Tensor::matmul` already supports batch broadcasting (Milestone 3), so a
  batched matmul's backward needs `reduce_sum_to_shape`-style handling for the *batch*
  dimensions too — new surface area, not a direct reuse of the existing elementwise helper
- **Grad check**: 2D and batched matmul, including at least one batch-broadcast case (e.g.
  `{2,2,3} @ {3,2}`)

**Step 6 — View ops backward**
- `transpose`: permute gradient with the *inverse* permutation (see table above — not simply
  re-applying `axes`); implemented as a dedicated `transpose(a, axes)` overload that computes
  the inverse explicitly
- `reshape`/`view`: reshape gradient back to input shape
- `slice`: zero-fill `grad_input` at the original shape, scatter `grad_output` into the sliced
  region using inline odometer-style indexing inside the backward closure — no new `Tensor`
  primitive was needed
- **Test**: gradient check for each view op, including a non-involutive `transpose` permutation

**Step 7 — `detach()` and `no_grad()` context**
- `Variable detach() const` — new Variable, same data, `requires_grad=false`, empty tape
- `no_grad()` / `set_grad_enabled(bool)` — global flag that ops check; when disabled, ops
  return plain Variables with empty tapes
- **Test**: verify `detach()` breaks graph, gradient doesn't flow back through detached path

**Step 8 — In-place ops: forbid for `requires_grad` Variables**
- Added a guarded in-place API: `Tensor::fill(float)` performs the actual mutation; `Variable::fill(float)` checks `requires_grad_` and throws `TorcError` if the Variable is tracked, otherwise delegates to `data_.fill()`. This is the first guarded in-place entry point in the API.
- Direct writes through `data()` and non-const `operator[]` remain available but unguarded — they bypass the check and are the user's responsibility.
- **Test**: verify `Variable::fill()` throws `TorcError` on a tracked Variable and succeeds on an untracked one; verify `Tensor::fill()` always works.

**Step 9 — `max`/`min` backward with argmax tracking**
- Whole-tensor `max`/`min`: scan all elements, scatter scalar `grad_output` to the argmax/argmin
- Axis-wise `max(axis)` / `min(axis)`: scan per-output-element along the reduced axis using
  the same stride math as `Tensor::reduce_axis`, scatter `grad_output` values to the
  argmax/argmin positions
- **Test**: hand-computed correctness tests for whole-tensor and axis-wise variants, plus
  gradient checks on a `{2,3}` tensor with `max(0)` and `max(1)` (one-sided forward
  difference, since `max`/`min` are piecewise-linear and central difference gives 0.5 at the
  kink)

### Constraints to enforce from Day 1

1. **No in-place modification** of Variables that require grad, outside the (not-yet-built)
   controlled in-place API from Step 8
2. **Tape is consumed** by `backward()` and cleared — a second `backward()` call on the same
   Variable is a silent no-op today (empty tape), not an error; revisit whether that should
   instead throw, per PyTorch's default behavior, when Step 8 lands
3. **`grad_` accumulates** via `+=`, not `=` — this enables gradient accumulation across
   mini-batches
4. **Non-scalar outputs** require an explicit `grad_output` argument to `backward()`
5. **`requires_grad` defaults to `false`** — only parameters should opt in, avoiding wasted
   memory on data/labels in a typical training loop (see TinyTorch Module 06, Q4)
6. **Every `Variable` in a graph must outlive `backward()` calls on its descendants** — see
   "Tape / graph structure" above. Not enforced by the type system today.

### Key risks identified from online sources

1. **Broadcasting backward bugs** (numpygrad docs, PyTorch forums): the single most common
   autograd mistake is forgetting to sum over broadcast dims. Test every broadcast case, not
   just one.
2. **In-place ops corrupting saved tensors** (PyTorch docs): if a user mutates an input after
   the forward pass, a saved-tensor copy in a backward closure could go stale relative to
   other state the graph depends on. Step 8's guarded in-place API is meant to prevent
   user-facing in-place mutation on tracked Variables once it exists.
3. **Float32 numerical noise in gradient checks**: PyTorch's `gradcheck` defaults to double
   precision for this reason. This project's `eps=1e-4` / `atol=1e-2` (see "Gradient
   checking" above) is calibrated for float32 against observed noise, not derived purely from
    truncation-error theory, and should be re-validated if precision ever changes.

---

## Milestone 5 — nn / optim / data (basic ML)

> **Status note:** live checklist state (what's actually done) lives in ROADMAP.md's
> Milestone 5 section — treat that as the single source of truth. This section explains the
> *why* behind the design and isn't updated per-step, so step numbers below describe scope,
> not completion.

### Module design decisions

`nn::Module` follows PyTorch's `nn.Module` pattern but is deliberately minimal:

- **`forward()` is the only required override.** Ops are **free functions** in `namespace torc`,
  not `Variable` methods or `Module` methods. `Module::forward` composes those free functions.
- **Parameter storage is explicit.** `register_parameter(name, param)` stores a named `Variable`
  in an `unordered_map`; `parameters()` flattens it to a `vector<Variable>`. There is no
  `add_module` / named submodule registry yet — `Sequential` owns children via
  `std::unique_ptr<Module>` and collects their parameters recursively.
- **`parameters()` returns by value.** This copies all parameters, which is fine for Milestone 5's
  small models. If optimizer patterns later require it, switch to `vector<Variable&>` or `span`.

### Constraints to enforce

1. **Ops stay free functions.** Do not add op methods to `Variable` or `Module` in Steps 5+ —
   this is already the established pattern from Milestone 4.
2. **Backward is centralized.** Broadcasting reduction (`reduce_sum_to_shape`) is applied once,
   centrally, in `Variable::backward_with_grad`, never inside a module's forward or an op's
   backward closure.
3. **`nn::Module` is stateful but not self-contained.** `parameters()` returns copies; there is
   no `state_dict` / `load_state_dict` yet. Revisit if serialization becomes a need.

### Key risks identified from online sources

1. **Forgetting to register parameters.** PyTorch's `nn.Module` auto-registers `nn.Parameter`
   attributes set in `__init__`; our `Module` requires explicit `register_parameter()` calls.
   Missing one means the optimizer never sees that weight. Tests should verify `parameters()`
   returns exactly what was registered.
2. **`parameters()` copy cost.** Returning `vector<Variable>` by value copies every parameter
   tensor. For Milestone 5 this is negligible, but if a model has thousands of parameters and
   `parameters()` is called every training step, this becomes wasteful. Switch to references
   before that becomes a bottleneck.
3. **Free-function ops inside `forward`.** Because `Module::forward` must compose free functions
   (`torc::mul_scalar`, `torc::add_scalar`, etc.), custom modules need to include
   `torc/autograd.hpp`. This is an extra header dependency compared to PyTorch, where ops are
   tensor methods — acceptable for a minimal library, but worth documenting so new contributors
   don't try to call `x.mul(y)`.

### Incremental implementation steps

Each step = implement → test → update-docs → build → commit. Do not batch steps into a single
PR. *(Check ROADMAP.md for actual checkbox state — not duplicated here.)*

**Step 5.1 — `Tensor::exp()`**
- Elementwise unary transcendental op using `std::ranges::transform` + `std::exp`
- Added `<cmath>` to `src/tensor.cpp` and `tests/test_tensor.cpp`
- **Test**: basic values, shape preservation, negative values

**Step 5.2 — `nn::Module` base class + `nn::Sequential` container**
- `nn::Module` is a lightweight base class with:
  - `virtual Variable forward(const Variable& x) const = 0`
  - `Variable operator()(const Variable& x) const` — calls `forward(x)`
  - `register_parameter(name, param)` — stores named `Variable` in `unordered_map`
  - `named_parameters()` — returns the map (const and non-const overloads)
  - `parameters()` — flattens to `std::vector<Variable>`
- `nn::Sequential` inherits `Module` and:
  - `add(std::unique_ptr<Module>)` appends a module
  - `forward(x)` chains modules sequentially
  - `parameters()` recursively collects from own map + all child modules
- Design decisions:
  - Ops remain **free functions** in `namespace torc`; `Module::forward` composes them
  - `Sequential` owns children via `std::unique_ptr<Module>`; no named submodule registry
    yet (PyTorch's `add_module` is not implemented — revisit if state-dict serialization
    needs it)
  - `parameters()` returns `std::vector<Variable>` by value; this copies all parameters,
    which is fine for Milestone 5's small models but should switch to `vector<Variable&>`
    or `span` if optimizer patterns require it later
  - **Forward lifetime**: `Module` owns a `mutable std::list<Variable> forward_cache_` that
    stores intermediates created during `forward()`. `operator()()` clears this cache before
    calling `forward()`, and subclasses append intermediates via `emplace_back`. Because
    `std::list` never relocates elements, raw `Variable*` pointers stored in tape entries
    remain valid until the next forward pass. This makes `output = module(x); output.backward()`
    safe without any user-side lifetime management.
- **Test**: parameter registration, forward correctness, `Sequential` chaining, empty
  sequential pass-through, own + child parameter collection

**Step 5.3 — `nn::Linear`**
- `Linear(in_features, out_features)` registers `weight` (shape `{out, in}`) and `bias`
  (shape `{out}`), both initialized to `0.01` and `0.0` respectively
- Forward: `x @ W.T + b`, implemented with existing `torc::matmul`, `torc::transpose`,
  and `torc::add` free functions — no new Tensor primitives
- **Test**: construction shape checks, forward correctness (unbatched and batched),
  parameter tracking, and hand-computed gradient checks for both unbatched and batched input
  (loss = sum of output)

**Step 5.3a — Fix `Module::forward` / `Module::operator()` lifetime for autograd**
- **Problem**: local `Variable` intermediates created inside `forward()` are destroyed at
  return, but their tape entries hold raw pointers to them. `backward()` on the returned
  `Variable` therefore dereferences dangling pointers.
- **Solution**: `Module` owns a `mutable std::list<Variable> forward_cache_`. `operator()()`
  clears it before calling `forward()`. Each module's `forward()` appends intermediates via
  `emplace_back` so their addresses are stable for the lifetime of the cache. Because
  `std::list` never relocates elements, raw `Variable*` pointers in tape entries remain valid
  until the next forward pass clears the cache.
- **API impact**: users call `output = module(x); output.backward()` exactly as in Python.
  `Sequential::forward()` calls `module->operator()()` on each child so their caches are
  populated too. The cache is per-module, not per-graph; calling `module(x)` again clears
  the previous cache automatically.
- **Why `std::list`**: `std::vector` can reallocate and move elements, invalidating raw
  pointers stored in tape entries. `std::list` guarantees stable addresses, which is a
  prerequisite for the current tape design. The memory overhead is acceptable for a naive
  reference implementation.
- **Test**: `Linear` gradient checks for unbatched and batched input pass; `Sequential` with
  multiple layers also works because each child's cache keeps its own intermediates alive.

**Step 5.4 — Activation functions**
- `nn::ReLU`, `nn::Sigmoid`, `nn::Softmax` — each is a `Module` with a parameterless
  `forward()` that delegates to the corresponding free function in `namespace torc`
- Free functions added in `src/autograd.cpp`: `torc::relu`, `torc::sigmoid`, `torc::softmax`,
  plus `torc::exp` (needed by sigmoid and softmax internally)
- `Tensor::exp()` and `Tensor::softmax()` added to `include/torc/tensor.hpp` / `src/tensor.cpp`
  as elementwise / whole-tensor primitives
- Backward rules:
  - **ReLU**: `dL/dx = dL/dy * (x > 0 ? 1 : 0)` — mask from forward pass
  - **Sigmoid**: `dL/dx = dL/dy * sigmoid(x) * (1 - sigmoid(x))` — uses captured forward output
  - **Softmax**: `dL/dx = y * (dL/dy - sum(dL/dy * y))` — uses captured forward output
- **Design note**: `Softmax` is currently whole-tensor (flattened). Per-axis softmax can be
  added later if needed for classification heads; the backward implementation is written so
  it operates over the full flattened tensor, which is correct for the 1D case and a
  reasonable MVP for higher-rank inputs.
- **Test**: forward correctness for each activation, backward correctness for ReLU and Sigmoid
  (hand-computed), and numerical gradient check for Softmax (central differences, `h=1e-4`)

**Step 5.5 — Loss functions**
- `nn::MSELoss` and `nn::CrossEntropyLoss` implemented as standalone classes in
  `include/torc/nn/losses.hpp` / `src/nn/losses.cpp` rather than inheriting `Module`
- **Why not inherit `Module`**: losses take *two* inputs (`input`, `target`), not one, so they
  cannot satisfy `Module::forward(const Variable& x)`. They are lightweight callables with
  `forward(input, target)` and `operator()(input, target)` instead.
- `MSELoss`: `mean((input - target)^2)`. Backward = `2 * (input - target) / n` to `input`,
  negated to `target`.
- `CrossEntropyLoss`: per-row softmax + log-softmax + negative log-likelihood. The loss
  averages over the batch. Backward = `(softmax - one_hot(target)) / batch_size` per row.
  **Important**: `Tensor::softmax()` is whole-tensor (flattened), so `CrossEntropyLoss`
  implements its own per-row softmax locally via `row_softmax()` — this is intentional until
  per-axis softmax is promoted to a `Tensor` primitive.
- **New Tensor primitive**: `Tensor::log()` (elementwise, using `std::ranges::transform`) added
  alongside `exp()` as a basic unary op.
- **New autograd free function**: `torc::log(const Variable&)` added in `src/autograd.cpp`.
- **Test**: forward hand-computed values for both losses, backward chain-rule check for MSE,
  and numerical gradient check for CrossEntropyLoss (central differences, `h=1e-4`)

### Optimizer design

Optimizers are deliberately separated from `Module`. They operate on non-owning
`std::vector<Variable*>` returned by `Module::parameters()` and mutate `param->data()` in-place
based on `param->grad()`. This matches PyTorch's separation of `nn.Module` and `torch.optim`.

- **API shape**: each optimizer class (`optim::SGD`, `optim::Adam`, `optim::AdamW`) exposes:
  - `optim::SGD(std::vector<Variable*>& params, float lr, float momentum = 0.0f)` — takes
    pointers, not copies, because `step()` must mutate the original `data_` tensors
  - `void step()` — reads `param->grad()` and updates `param->data()` in-place; skips parameters
    without gradients
  - `void zero_grad()` — clears `has_grad_` and `grad_` on every tracked parameter
- **Why pointers**: `Module::parameters()` previously returned `std::vector<Variable>` by value,
  which copies all parameters. That is fine for inspection but useless for mutation. Step 5.6
  changed `parameters()` to return `std::vector<Variable*>` so optimizers can mutate in-place
  without copying.
- **Gradient accumulation**: optimizers assume `grad_` is already populated by `backward()` and
  that `zero_grad()` was called at the start of the training step. No internal accumulation
  logic lives in the optimizer.
- **Momentum**: velocities are stored per-parameter in a `std::vector<Tensor>` inside the
  optimizer, initialized to zero with the same shape as each parameter's data.

**Step 5.6 — `optim::SGD` with momentum**
- `include/torc/optim.hpp` / `src/optim.cpp`
- Constructor: `SGD(std::vector<Variable*>& params, float lr, float momentum = 0.0f)`
- `step()`: for each parameter with a gradient, either applies vanilla SGD
  (`param = param - lr * grad`) or momentum SGD (`velocity = momentum * velocity + grad;
  param = param - lr * velocity`)
- `zero_grad()`: calls `param->zero_grad()` on every tracked parameter
- **Test**: forward + backward on a small `Linear` model, verify `step()` updates weight and bias
  by the expected amount; verify momentum accumulates velocity across multiple steps; verify
  `zero_grad()` clears gradients; verify `step()` skips parameters without gradients

**Step 5.7 — `optim::Adam`**
- `include/torc/optim.hpp` / `src/optim.cpp`
- Constructor: `Adam(std::vector<Variable*>& params, float lr, float beta1=0.9f, float beta2=0.999f, float eps=1e-8f)`
- Maintains per-parameter moving averages `m_` and `v_` (initialized to zero), plus a `step_` counter
- `step()`:
  - `m = beta1 * m + (1 - beta1) * grad`
  - `v = beta2 * v + (1 - beta2) * grad^2`
  - `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)` (bias correction)
  - `param = param - lr * m_hat / (sqrt(v_hat) + eps)`
- `zero_grad()`: calls `param->zero_grad()` on every tracked parameter
- **New Tensor primitives needed**: `Tensor::sqrt()` (elementwise, uses `std::sqrt`) added alongside
  `exp()`/`log()` as a basic unary op. `torc::sqrt(const Variable&)` autograd free function
  added in `src/autograd.cpp` with backward `grad_input = grad_output / (2 * sqrt(input))`.
- **Test**: forward + backward on a small `Linear` model, verify `step()` updates params;
  verify `zero_grad()` clears gradients; verify `step()` skips params without grads

**Step 5.8 — `optim::AdamW`**
- `include/torc/optim.hpp` / `src/optim.cpp`
- Constructor: `AdamW(std::vector<Variable*>& params, float lr, float weight_decay = 0.01f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)`
- `step()` applies decoupled weight decay: `param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)`
  - Weight decay is added directly to the parameter update rather than being injected into the gradient
  - This matches PyTorch's `torch.optim.AdamW` behavior
- Maintains the same `m_`, `v_`, and bias correction state as `Adam`
- **Test**: forward + backward on a small `Linear` model, verify `step()` with `weight_decay=0.0`
  matches `Adam` behavior; verify `weight_decay > 0` reduces parameters additionally

### Data loader design

Data loading is the thinnest possible wrapper around a dataset, matching PyTorch's
`torch.utils.data.DataLoader` pattern but without the multiprocessing complexity.

- **`data::Dataset`** is an abstract base class with:
  - `virtual size_t len() const = 0`
  - `virtual std::pair<Tensor, Tensor> get(size_t idx) const = 0` — returns `(x, y)` as
    `Tensor`s, not `Variable`s. The training loop wraps them in `Variable`s if `requires_grad`
    is needed.
- **`data::DataLoader`** takes a `Dataset` and produces batches:
  - `DataLoader(const Dataset& ds, size_t batch_size, bool shuffle = false)`
  - `std::vector<std::pair<Tensor, Tensor>> next_batch()` — returns a batch of `(x, y)` pairs
    as `Tensor`s; user is responsible for stacking/reshaping into a batched `Tensor`
- **No collation / padding yet.** Every dataset sample must have the same shape; `DataLoader`
  does not handle ragged inputs. This keeps the first implementation minimal and deferrable.
- **Synthetic loaders first**: Steps 5.9–5.10 should build a synthetic regression dataset and
  a small CSV-backed classification dataset before any real-data loading is attempted.

---

## Planned future project structure

The current layout is intentionally flat because there's only a Tensor class plus autograd.
Once nn/optim/data follow (Milestone 5), flat `src/`/`include/` stops scaling and `main.cpp`
stops being a reasonable place for examples. Proposed layout, to be adopted incrementally as
each milestone actually lands (don't scaffold empty directories ahead of the code that fills
them):

```
torc/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── AGENTS.md
├── ROADMAP.md
├── docs/
│   ├── DESIGN.md
│   └── AUTOGRAD.md
├── include/
│   └── torc/
│       ├── tensor.hpp
│       ├── utils.hpp
│       ├── autograd.hpp     # Variable, TapeEntry, backward()    [Milestone 4 — landed]
│       ├── nn.hpp           # Module base + Sequential           [Milestone 5.2 — landed]
│       ├── nn/
│       │   ├── linear.hpp   # nn::Linear                          [Milestone 5.3]
│       │   ├── activations.hpp # nn::ReLU, nn::Sigmoid, etc.     [Milestone 5.4]
│       │   └── losses.hpp   # nn::MSELoss, nn::CrossEntropyLoss   [Milestone 5.5]
│       ├── optim.hpp        # SGD, Adam, AdamW                    [Milestone 5.6-5.8]
│       └── data.hpp         # Dataset, DataLoader                 [Milestone 5]
├── src/
│   ├── tensor.cpp
│   ├── autograd.cpp                                              [Milestone 4 — landed]
│   ├── nn.cpp                                                   [Milestone 5.2 — landed]
│   ├── nn/
│   │   ├── linear.cpp                                            [Milestone 5.3]
│   │   ├── activations.cpp                                       [Milestone 5.4]
│   │   └── losses.cpp                                             [Milestone 5.5]
│   ├── optim.cpp                                                 [Milestone 5.6-5.7]
│   └── matmul_blas.cpp                                           [Milestone 3 — landed]
├── examples/                # demo binaries move out of src/, main.cpp retired
│   ├── basic_ops.cpp        # today's main.cpp, relocated
│   ├── linear_regression.cpp                                     [Milestone 5]
│   └── mlp_classification.cpp                                    [Milestone 5]
└── tests/
    ├── test_tensor.cpp
    ├── test_autograd.cpp                                         [Milestone 4 — landed]
    ├── test_nn.cpp                                                [Milestone 5.2 — landed]
    ├── test_optim.cpp                                             [Milestone 5]
    └── test_data.cpp                                              [Milestone 5]
```

Rationale for the specific moves:

- **`nn/` as a subdirectory, not a single file, once it has ≥2 layer types.** A single
  `nn.cpp` is fine for `Linear` alone but won't stay readable once activations, losses, and
  more layers land in the same milestone.
- **`nn.hpp` splits into `nn/` sub-headers** (`linear.hpp`, `activations.hpp`) once those
  modules exist, keeping each header focused. The top-level `nn.hpp` remains the public entry
  point and re-exports the sub-headers or just the base `Module`/`Sequential` types.
- **`examples/` instead of a single `main.cpp`.** AGENTS.md already says `main.cpp` should
  stay a thin demo, not a dumping ground — once there are two real end-to-end examples
  (linear regression, MLP classification) that stops being true of a single file. Multiple
  small example binaries keep each one focused and testable in isolation via CMake.
- **`data.hpp`/`data.cpp` are deliberately minimal at first** — a `Dataset` interface plus
  one synthetic or small CSV-backed implementation is enough to unblock the Milestone 5
  end-to-end examples. Don't build a general data pipeline before there's a second dataset
  that needs one.
- **`optim.hpp`/`optim.cpp` mirror PyTorch's `torch.optim` split.** Optimizers take mutable
  references to `Variable` parameters and update `data_` in-place; they do not own parameters
  and do not track graph state. This keeps the optimizer implementation independent from
  `nn::Module` and avoids circular dependencies.
- **`CMakeLists.txt` will need `add_subdirectory` or explicit source lists per target** once
  `nn/` exists as a folder, plus `BUILD_EXAMPLES`/`BUILD_TESTS` options if the example count
  grows enough that not everyone wants to compile all of them by default.

This structure is a target, not a mandate to create now — build it milestone-by-milestone so
each directory only exists once something real lives in it.
