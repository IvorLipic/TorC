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

### View ops backward — trivial but must be implemented

| Op | Forward | Backward rule |
|---|---|---|
| `transpose(axes)` | permutes dims | `grad_input = grad_output.transpose(inverse_of(axes))` — **not** re-applying `axes`; only self-inverse for the default (full-reversal) case or true involutions. A general permutation needs its actual inverse computed. |
| `reshape(new_shape)` / `view(new_shape)` | copies storage to new shape | `grad_input = grad_output.reshape(original_shape)` |
| `slice(slices)` | copies a sub-region | zero-fill `grad_input` at the original shape, scatter `grad_output` into the sliced region — implemented inline in the backward closure using odometer-style index incrementing over `grad_output.shape()`, no new `Tensor` primitive needed. |

`Tensor` currently has no `unsqueeze`/`squeeze`. Since both are just a `reshape()` that
inserts or removes a size-1 dimension, their backward (and forward, if ever needed) can
likely be expressed directly via the existing `reshape()`/`view()` rather than new `Tensor`
primitives — revisit only if a standalone forward API turns out to be needed elsewhere.

### Non-differentiable ops — defer, don't fake

`max()` / `min()` (with or without axis) require **argmax tracking** during forward to
scatter gradients correctly in backward. Do not implement a "gradient goes to the max
element" rule without storing the argmax indices. Defer to a dedicated step (Step 9) after
the core is proven correct.

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
- Forward: store `argmax` indices (per-axis or global) on the Variable
- Backward: scatter gradient to the max positions only
- **Test**: gradient check on a `{2,3}` tensor with `max(0)` and `max(1)`

**Step 10 — Project layout restructure**
- `include/torc/autograd.hpp`, `src/autograd.cpp`, `tests/test_autograd.cpp`, wired into
  `CMakeLists.txt`
- **Already landed** — done alongside Step 1 rather than deferred to the end; kept here only
  for historical scoping reference.

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
│       ├── nn.hpp           # Module base + Linear, activations  [Milestone 5]
│       ├── optim.hpp        # SGD, Adam                          [Milestone 5]
│       └── data.hpp         # Dataset, DataLoader                [Milestone 5]
├── src/
│   ├── tensor.cpp
│   ├── autograd.cpp                                              [Milestone 4 — landed]
│   ├── matmul_blas.cpp                                           [Milestone 3 — landed]
│   ├── nn/
│   │   ├── linear.cpp                                            [Milestone 5]
│   │   └── activations.cpp                                       [Milestone 5]
│   ├── optim.cpp                                                 [Milestone 5]
│   └── data.cpp                                                  [Milestone 5]
├── examples/                # demo binaries move out of src/, main.cpp retired
│   ├── basic_ops.cpp        # today's main.cpp, relocated
│   ├── linear_regression.cpp                                     [Milestone 5]
│   └── mlp_classification.cpp                                    [Milestone 5]
└── tests/
    ├── test_tensor.cpp
    ├── test_autograd.cpp                                         [Milestone 4 — landed]
    ├── test_nn.cpp                                                [Milestone 5]
    └── test_optim.cpp                                             [Milestone 5]
```

Rationale for the specific moves:

- **`nn/` as a subdirectory, not a single file, once it has ≥2 layer types.** A single
  `nn.cpp` is fine for `Linear` alone but won't stay readable once activations, losses, and
  more layers land in the same milestone.
- **`examples/` instead of a single `main.cpp`.** AGENTS.md already says `main.cpp` should
  stay a thin demo, not a dumping ground — once there are two real end-to-end examples
  (linear regression, MLP classification) that stops being true of a single file. Multiple
  small example binaries keep each one focused and testable in isolation via CMake.
- **`data.hpp`/`data.cpp` are deliberately minimal at first** — a `Dataset` interface plus
  one synthetic or small CSV-backed implementation is enough to unblock the Milestone 5
  end-to-end examples. Don't build a general data pipeline before there's a second dataset
  that needs one.
- **`CMakeLists.txt` will need `add_subdirectory` or explicit source lists per target** once
  `nn/` exists as a folder, plus `BUILD_EXAMPLES`/`BUILD_TESTS` options if the example count
  grows enough that not everyone wants to compile all of them by default.

This structure is a target, not a mandate to create now — build it milestone-by-milestone so
each directory only exists once something real lives in it.
