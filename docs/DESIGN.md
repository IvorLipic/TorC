# torc — Design Notes

This file holds decisions that are expensive to reverse and the rationale behind them,
plus the planned future project structure once ML work (autograd → nn → optim → data)
starts. Update this *before* writing code that implements a new architectural layer, not
after — the point is to force the decision to be explicit.

---

## Review-derived design constraints and known gaps

This section records issues found after the initial milestone implementation. It guides agents
working on the hardening backlog in `ROADMAP.md`; it is not a claim that these behaviors are fixed.

### Softmax and cross-entropy semantics

`Tensor::softmax()` remains a flattened legacy primitive for compatibility and is only suitable for
one-dimensional inputs. The axis-aware overload `Tensor::softmax(axis)` computes a stable softmax
independently along the selected dimension; `torc::softmax(a, axis)` uses the matching per-axis
Jacobian-vector product. `nn::Softmax(axis = -1)` defaults to the last axis, so conventional
`{batch, classes}` inputs normalize each row without mixing examples. Batched forward and backward
tests enforce this contract. New code should pass an explicit axis (or use the module default)
rather than rely on flattened behavior.

`CrossEntropyLoss` validates rank-2, non-empty logits and rank-1 targets whose length equals the
batch size before indexing. Targets must be finite integer-valued class indices in
`[0, classes)`, and logits must be finite; malformed input throws `ShapeError` rather than risking
an out-of-bounds read or divide by zero. Forward computes each row's log-sum-exp directly, with
double-precision accumulation, instead of taking `log(softmax(logits))`. Backward reuses the
validated, stably computed per-row probabilities. Extreme finite logits therefore remain usable;
non-finite logits are rejected explicitly.

### Numerical domains and empty tensors

`NumericalError` is the `TorcError` subclass for invalid numeric inputs. Domain-sensitive tensor
operations and reductions reject NaN or infinity rather than silently propagating them. Ordinary
add/subtract/multiply/negate remain IEEE-style elementwise operations (so callers can explicitly
choose propagation without paying a validation scan). Division rejects both scalar zero divisors
and tensor divisors containing zero. `log` requires strictly positive finite inputs; `sqrt` requires
non-negative finite inputs. `exp`, softmax, and matmul require finite inputs; softmax retains its
separate empty-tensor `ShapeError` for both flattened and axis-aware forms.

Empty reductions follow explicit identities: whole-tensor `sum()` returns `0`, while `mean()`,
`max()`, `min()`, and every axis reduction throw `ShapeError` because no finite result is defined.
Elementwise operations on empty tensors preserve their shape when their inputs satisfy the numeric
contract. These checks happen before SIMD kernels so CPU-specific paths cannot bypass validation.

### Graph ownership, mutability, and gradient mode

The tape stores raw pointers to input Variables to preserve the value-style API, but each pointer is
paired with a weak lifetime token captured when the tape entry is created. Tracked inputs also carry
a recorded `requires_grad` flag; untracked constants are not traversed and remain safe to discard
because backward closures capture the tensor values they need. Backward validates the metadata and
token before every graph traversal or pointer dereference, throwing `TorcError` if a tracked input
has expired. This converts the former use-after-free UB into a deterministic error, but does not
retain graph nodes: every tracked Variable must still outlive `backward()` for a graph that uses it.
Variable copy construction creates a fresh token, while moves transfer the token, so ordinary
value-return and container moves do not invalidate live graph edges.

`Variable` data, gradients, flags, and tape are currently public. This was convenient for
incremental tests but prevents invariant enforcement; make them private before adding version
counters, saved-tensor checks, or graph reuse.

The `fill()` guard is incomplete while mutable `data()` and indexing remain available: callers can
modify a tracked input after forward and invalidate saved derivatives. Either remove those mutable
access paths, add version checking, or explicitly detect unsupported mutation.

`set_grad_enabled()` is a process-global boolean. It must become thread-local and scoped by an RAII
guard so exceptions and nested inference regions restore the previous state reliably.

### Memory, shape, and portability constraints

Tensor operations intentionally copy contiguous storage, and backward closures often capture full
Tensor copies. This is useful for a reference implementation, but memory scales with graph size.
Any stride/view or saved-tensor optimization must preserve row-major semantics and update ownership
rules together.

Broadcast kernels currently reconstruct an index vector per output element, and shape products and
element counts use `int`. Replace allocation-heavy indexing and add checked arithmetic before
accepting large shapes.

The default build enables CPU-specific optimization (`-march=native` or AVX2 under MSVC). Keep a
portable baseline path and make ISA-specific dispatch opt-in or runtime-selected; otherwise a
binary built on one machine may fail on another.

### Verification and product boundary

The aggregate GoogleTest target does not replace sanitizer, fuzz/property, malformed-input, or
clean-build coverage. New hardening work should add those checks and keep the documented test
command reproducible from a fresh build directory.

The project currently has no install/package metadata, CI, serialization/state-dict API, or
train/eval mode. These are scope decisions, not accidental PyTorch compatibility: documentation
and examples should call torc a small CPU reference/educational library until those capabilities
are designed.

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

Storage is always row-major and contiguous. `reshape()` copies the shape vector and copies
storage (the method is `const`, so `storage_` cannot be moved); `view()` delegates to `reshape()`.
There is no stride metadata yet, so `transpose()` and `slice()` explicitly reorder/copy into new
contiguous buffers.

### Memory ownership
`storage_` is an owned `std::vector<float>` per `Tensor`, copied fresh on every elementwise
  or reduction op. Shape-changing ops (`reshape`/`view`) copy `storage_` (the methods are `const`,
  so it cannot be moved), so they are O(N) with respect to data size. Fine for a naive reference
  implementation.
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
- **Cache-blocked tiling, `float` accumulator.** Complexity is `m * n * k` with `acc` of type
  `float`. The implementation uses tile sizes of 32×32×32 with an `i, k, j` loop order and a
  sparsity early-exit inside the innermost loop. This was shipped ahead of Milestone 6 because
  it is correctness-preserving for dense inputs and the naive triple-loop was the clear
  bottleneck for ML workloads. The sparsity early-exit is retained for potential sparse use
  cases but is a branch in the hot path for dense data — revisit if dense throughput becomes a
  priority.

### Error handling convention
`TorcError` (base, derives `std::runtime_error`), `ShapeError` (shape mismatches), and
`NumericalError` (invalid numeric domains or non-finite values) live in `utils.hpp`. Tensor shape
failures should use `ShapeError`; numeric-domain failures should use `NumericalError`. The current
data-loader code still throws standard `invalid_argument`, `out_of_range`, and `runtime_error`
exceptions in several paths; unifying those errors is part of the review-derived hardening backlog.
Existing `catch (std::runtime_error&)` sites remain compatible because the torc base error derives
from it.

### Documentation consistency

Performance and milestone notes are historical design records, but they must still identify the
current implementation accurately. In particular, the matmul paragraph above describes a retained
sparsity early-exit while the current implementation has removed that branch, and the roadmap has
an unchecked AVX2 vectorization item despite the implementation containing an AVX2 path. When a
performance change lands, update both the checklist and this rationale in the same change so agents
do not optimize against obsolete assumptions.

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
- `Variable detach() const` — new Variable, copy of the data (Tensor copy ctor deep-copies
  storage_), `requires_grad=false`, empty tape
- `no_grad()` / `set_grad_enabled(bool)` — global flag that most arithmetic ops check; when
  disabled, those ops return plain Variables with empty tapes. Note: activation and transcendental
  ops did not originally check this flag (a bug fixed in Step 5.12 follow-up)
- **Test**: verify `detach()` breaks graph, gradient doesn't flow back through detached path;
  verify `set_grad_enabled(false)` blocks tape-building for arithmetic, activation, and
  transcendental ops

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
  - **Sequential implementation**: `Sequential::forward()` calls each child module's
    `operator()()` directly, which clears each child's `forward_cache_` before calling
    `forward()`. This prevents unbounded cache growth across repeated forward passes while
    keeping tape-entry pointers valid for the duration of backward.
- **Test**: parameter registration, forward correctness, `Sequential` chaining, empty
  sequential pass-through, own + child parameter collection

**Step 5.3 — `nn::Linear`**
- `Linear(in_features, out_features)` registers `weight` (shape `{out, in}`) and `bias`
  (shape `{out}`), both initialized to `0.01` and `0.0` respectively
- `Linear(in_features, out_features, init_std)` overload: if `init_std <= 0`, weight is
  initialized with Kaiming normal (`std::sqrt(2 / fan_in)`); otherwise samples weight from
  `N(0, init_std)`. Bias remains `0.0`.
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
   `Sequential::forward()` calls each child's `operator()()` directly, which clears each
   child's `forward_cache_` before calling `forward()`. This prevents unbounded cache growth
   across repeated forward passes while keeping tape-entry pointers valid for the duration of
   backward.
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
- **Design note**: The no-argument `Softmax` primitive preserves whole-tensor (flattened)
  semantics for compatibility. Axis-aware softmax is now the preferred path: `nn::Softmax`
  defaults to the last axis and the free function accepts an explicit axis, with backward using
  the same axis.
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
- `CrossEntropyLoss`: validates rank/shape/domain constraints, computes per-row probabilities
  with a stable max-shifted softmax, and evaluates negative log-likelihood from log-sum-exp.
  The loss averages over the batch. Backward = `(softmax - one_hot(target)) / batch_size` per row.
  The local `row_softmax()` remains intentionally separate so the loss can share the validated
  row layout while avoiding an intermediate `log(softmax(...))` computation.
- **New Tensor primitive**: `Tensor::log()` (elementwise, using `std::ranges::transform`) added
  alongside `exp()` as a basic unary op.
- **New autograd free function**: `torc::log(const Variable&)` added in `src/autograd.cpp`.
- **Test**: forward hand-computed values for both losses, backward chain-rule check for MSE,
  numerical gradient check for CrossEntropyLoss (central differences, `h=1e-4`), and malformed
  input plus extreme-logit coverage for CrossEntropyLoss.

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

**Step 5.9 — `data::Dataset` + `data::DataLoader`**
- `include/torc/data.hpp` / `src/data.cpp`
- `Dataset` is an abstract base class with `virtual size_t len() const = 0` and
  `virtual std::pair<Tensor, Tensor> get(size_t idx) const = 0` — returns `(x, y)` as
  `Tensor`s, not `Variable`s
- `TensorDataset(Tensor xs, Tensor ys)` stores `xs` and `ys` by value; `get(idx)` slices the
  first dimension and reshapes to remove the sample axis
- `DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle = false)`:
  - `std::pair<Tensor, Tensor> next_batch()` — stacks individual samples into batched tensors
    with the batch dimension prepended; the last batch may be smaller
  - `bool has_next() const` — true if the current epoch has more batches
  - `void reset()` — starts a new epoch; reshuffles indices if `shuffle` was set
- **PyTorch pattern followed**: map-style Dataset with `__len__` / `__getitem__` semantics,
  `DataLoader` as the sampling/batching layer, shuffle applied at epoch start
- **No multiprocessing / collate_fn yet** — kept minimal for Milestone 5
- **Test**: `TensorDataset` construction, indexing, out-of-range error, mismatched sample count;
  `DataLoader` iteration, batch shape, last-batch size, shuffle permutation, reset, empty dataset

**Step 5.10 — Toy dataset loaders**
- `SyntheticRegression` in `include/torc/data.hpp` / `src/data.cpp` — generates `(x, y)` pairs
  in-memory with configurable `num_samples`, `num_features`, `weight`, `bias`, `noise_std`,
  and `seed`; follows PyTorch's `sklearn.datasets.make_regression` pattern
- `CSVDataset` in `include/torc/data.hpp` / `src/data.cpp` — loads tabular data from a file:
  - `CSVDataset::Options` controls `has_header`, `delimiter`, `feature_cols`, `target_col`
  - Reads all rows into `Tensor` storage at construction; `get(idx)` slices like `TensorDataset`
  - Validates consistent column counts and parseable floats
- **Extensibility for future datasets**: the `Dataset` abstraction is format-agnostic. Future
  `TextDataset` / `SequenceDataset` (token vocab + padded sequences) will subclass `Dataset`
  directly without changing `DataLoader`
- **Test**: `SyntheticRegression` shape, values, reproducibility; `CSVDataset` loading,
  header skipping, custom delimiter, malformed-line error, missing-file error, inconsistent
  columns error; `MNISTDataset` loading, length, shape, pixel normalization, inconsistent
  columns error

**Step 5.11 — End-to-end linear regression example**
- `examples/linear_regression/linear_regression.cpp` — trains `nn::Linear(1, 1)` on `data::SyntheticRegression`
  using `optim::SGD` and `nn::MSELoss`
- Training loop uses `DataLoader` for batching and shuffling
- Prints epoch loss and learned parameters to stdout
- Saves `loss_history.csv` (epoch, loss) and `predictions.csv` (x, y_true, y_pred) for
  external visualization
- **Visualization approach**: simplest path is a tiny optional Python helper script
  (`examples/linear_regression/plot_results.py`) using matplotlib. It reads the two CSVs and produces
  `loss_curve.png` and `predictions.png`. The C++ binary has zero plotting dependencies;
  users without Python can open the CSVs in Excel/Google Sheets/LibreOffice Calc.

**Step 5.12 — End-to-end MLP on MNIST**
- `examples/mnist_mlp/mnist_mlp.cpp` — trains a 3-layer MLP (`Linear(784, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 10)`)
  on `data::MNISTDataset` using `optim::AdamW` and `nn::CrossEntropyLoss`
- `MNISTDataset` in `include/torc/data.hpp` / `src/data.cpp` — loads MNIST-format CSV files
  (label + 784 pixels per row), normalizes pixels to `[0, 1]` by dividing by 255
- Training loop uses `DataLoader` for batching and shuffling
- Accepts optional `max_samples` CLI arg to limit training set size for quick experiments
- Prints per-epoch loss and accuracy to stdout; writes `loss_history.csv` and `per_class_accuracy.csv`
- Evaluation reuses the same `MNISTDataset` / `DataLoader` objects rather than reloading CSVs each epoch

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
  - `std::pair<Tensor, Tensor> next_batch()` — returns `(x_batch, y_batch)` with the batch
    dimension prepended (shape `{batch_size, *sample_shape}`); the last batch may be smaller
    if `len(ds) % batch_size != 0`
  - `bool has_next() const` — true if the current epoch has more batches
  - `void reset()` — starts a new epoch; reshuffles indices if `shuffle` was set
- **No collation / padding yet.** Every dataset sample must have the same shape; `DataLoader`
  does not handle ragged inputs. This keeps the first implementation minimal and deferrable.
- **Synthetic loaders first**: Steps 5.9–5.10 should build a synthetic regression dataset and
  a small CSV-backed classification dataset before any real-data loading is attempted.

---

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
