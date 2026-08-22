# torc — Design Notes

This file holds decisions that are expensive to reverse and the rationale behind them,
plus the planned future project structure once ML work (autograd to nn to optim to data)
starts. Update this *before* writing code that implements a new architectural layer, not
after — the point is to force the decision to be explicit.

---

## Decisions made so far

### dtype strategy
Single dtype, hardcoded loat32. Staying single-dtype is the simplest option and matches
the " naive-but-correct\ goal through Milestone 3. Revisit with a dtype enum or templated
storage (Tensor<T>) only if multi-precision becomes an actual need — don't add it
speculatively, since it touches every op signature.

### Shape / broadcasting semantics
Elementwise ops use NumPy-style broadcasting via the roadcast_shape() helper in
utils.hpp. Shapes are aligned from the right; dimensions are compatible if they are equal
or one of them is 1. The result shape is the maximum along each dimension. This was
implemented in Milestone 2 before autograd, because broadcasting changes what a gradient's
\sum over broadcast dims\ step needs to look like.

Storage is always row-major and contiguous. eshape() copies the shape vector and moves
storage; iew() delegates to eshape(). There is no stride metadata yet, so ranspose()
and slice() explicitly reorder/copy into new contiguous buffers.

### Memory ownership
storage_ is an owned std::vector<float> per Tensor, copied fresh on every elementwise or
reduction op. Shape-changing ops (eshape/iew) move storage_ instead of copying, so they
are O(1) with respect to data size. Fine for a naive reference implementation. **This will need
to change for autograd** — see below.

### Linear algebra — matmul (Milestone 3)

Tensor::matmul(const Tensor&) is the first non-elementwise op. Decisions, recorded here so
they are not re-litigated:

- **Operand rank >= 2 required.** A ShapeError is thrown for rank 0/1 operands. This is
 deliberately *forward-compatible*: allowing 1D promotion later only *widens* accepted input, so
 it cannot break existing callers, and the shape checks stay the same. Milestone 5's 
n::Linear
 only ever needs rank >= 2, so 1D/dot-product support is not an early need.
- **Batch broadcasting reuses roadcast_shape().** The last two dims are the matrix (contracted
 on the second of self and first of other); everything before them is the batch prefix and is
 broadcast with the same helper already used for elementwise ops. This honors the AGENTS.md rule
 against ad-hoc per-op shape coercion. Batch-broadcast failures are caught and re-thrown with the
 **full operand shapes** in the message, since roadcast_shape() would otherwise only print the
 prefix subspans.
- **Zero-size dimensions are allowed and defined as zeros.** (2,0) @ (0,3) yields a (2,3) of
 zeros; (0,3) @ (3,4) yields an empty (0,4). The contraction is an empty sum, so the loops
 already produce the right answer — no special-casing, and it matches shapes that are legal at
 construction.
- **Naive triple-loop, loat accumulator.** Identity: m * n * k with cc of type loat.
 Loop order is i, j, k. Optimizing to i, k, j (better cache locality) or adding SIMD/threads
 is explicitly deferred to Milestone 6 and must be benchmark-gated. The loat (not double)
 accumulator keeps parity with a future sgemm-based BLAS backend tight.

**Deferred to a separate plan:** the basic BLAS backend behind a CMake flag (the last Milestone 3
item). It is intentionally sequenced *after* this naive implementation is proven correct by tests,
per the roadmap. When it lands it owns the first real backend seam (likely a src/matmul.cpp
split and a TORC_USE_BLAS option), not this milestone.

### Error handling convention
TorcError (base, derives std::runtime_error) and ShapeError (shape mismatches) live in
utils.hpp. All error sites use ShapeError; existing catch (std::runtime_error&) still
works since it's a base class.

---

## Autograd — refined design and implementation plan (Milestone 4)

### What "tape-based" actually means here

The term is used in two distinct ways in the literature (see arxiv:1810.11530):

- **Source-transformation tape** (ADIFOR, Tapenade): a global stack that records raw
  intermediate values; the adjoint program reads it. Not relevant to C++ operator-overloading.
- **Operator-overloading tape** (PyTorch, Chainer, Autograd): each `Variable` carries a
  `grad_fn` pointer to the op that produced it. The chain of `grad_fn` pointers *is* the tape —
  a linked list walked in reverse during `backward()`. This is what we implement.

Each op's `grad_fn` holds the inputs it needs for its local derivative (the "saved tensors"
pattern). Backward is a linear walk over this per-output tape, not a global data structure.

### Ownership model — decided

`Variable` **owns** its `Tensor data_` directly (not `shared_ptr`). The `grad_fn` closures
capture **copies** of any tensors they need for backward computation. This is intentionally
naive: it means intermediate tensors stay alive as long as the output Variable that carries
their `grad_fn` is alive. The user (or training loop) must keep the loss Variable alive until
`backward()` completes — the same constraint PyTorch documents with "the graph keeps
intermediates alive."

**Rationale**: `shared_ptr` adds cycle-prone complexity for no benefit at this scale. Copies
cost O(N) per saved tensor but the code stays simple and correct. Revisit when memory pressure
actually matters (Milestone 6+).

### Tape / graph structure

```cpp
// Forward-declared; defined in autograd.cpp
struct Variable;

struct TapeEntry {
    std::function<void(const Tensor& grad_output)> backward;
    // backward reads saved tensors from its own closure; no raw pointers needed here
};

class Variable {
public:
    Tensor data_;
    Tensor grad_;          // None (empty Tensor) until first backward populates it
    bool requires_grad_;
    std::vector<TapeEntry> tape_;
};
```

Each op's static `apply(inputs...)` method:
1. Calls the `Tensor` forward to get `data_out`.
2. Creates `Variable out(data_out, requires_grad_of_any_input)`.
3. If `out.requires_grad_`, appends a `TapeEntry` to `out.tape_` whose closure captures
   copies of the inputs it needs, and whose `backward` function:
   - Receives `grad_output` (the upstream gradient w.r.t. `out.data_`)
   - Computes per-input gradients using the chain rule
   - Calls `input.grad_.accumulate(local_grad)` for each input that requires grad
   - Calls `input.backward(local_grad)` to propagate further upstream

`backward()` on a Variable:
- Seeds with `Tensor::ones` if the output is scalar and no explicit gradient was given
  (`?loss/?loss = 1`); throws `ShapeError` for non-scalar outputs with no explicit gradient
- Walks `tape_` in reverse, calling each entry's `backward(upstream_grad)`
- Clears `tape_` after completion (graph is consumed; no re-backwarding)

### Broadcasting backward — must be explicit

When a broadcast happened during forward (e.g. shape `{3}` broadcast to `{2,3}` in an add),
the backward gradient has shape `{2,3}` but the input's grad must have shape `{3}`. The
intermediate must be **summed over the broadcast dimensions** before being written to the
input's `.grad_`.

Implementation: `reduce_sum_to_shape(grad, target_shape)` — sums over any leading dimensions
that were introduced by broadcasting. This mirrors PyTorch's `_sum_to` (see
`aten/src/ATen/native/Utils.h` in the PyTorch source) and the `Function.unbroadcast()` static
method in `ml-by-hand`.

**Test requirement**: every broadcast case must be verified with both a correctness test and a
gradient check. A missing unbroadcast is the most common autograd bug.

### View ops backward — trivial but must be implemented

These have simple but non-obvious backward rules that must be coded explicitly:

| Op | Forward | Backward rule |
|---|---|---|
| `transpose(axes)` | permutes dims | `grad_input = grad_output.transpose(inv(axes))` |
| `reshape(new_shape)` | copies storage to new shape | `grad_input = grad_output.reshape(original_shape)` |
| `slice(slices)` | copies a sub-region | zero-fill `grad_input`, scatter `grad_output` into sliced region |
| `unsqueeze(axis)` | inserts dim of size 1 | `grad_input = grad_output.squeeze(axis)` |

### Non-differentiable ops — defer, don't fake

`max()` / `min()` (with or without axis) require **argmax tracking** during forward to
scatter gradients correctly in backward. Do not implement a "gradient goes to the max element"
rule without storing the argmax indices. Defer to a dedicated step after the core is proven
correct.

### Scalar vs. non-scalar backward

`backward()` on a non-scalar Variable is illegal without an explicit upstream gradient
argument, because there is no canonical `?loss/?loss = 1` seed. Enforce this: throw
`ShapeError` if `output.numel() != 1` and no `grad_output` was provided. This matches
PyTorch's documented behavior and prevents silent gradient-shape bugs.

### Gradient accumulation semantics

- `grad_` accumulates via `+=` on repeated `backward()` calls — this is intentional and
  required for mini-batch gradient accumulation
- Leaf Variables (user-created, `requires_grad=true`) start with `grad_` empty
- Call `zero_grad()` (sets `grad_` to empty Tensor) before each training iteration
- Intermediate Variables compute their grad, propagate it, then discard it

### Gradient checking — methodology, not just a checkbox

For every differentiable op, verify analytical gradients against **central finite differences**:

```
h = 1e-4  (for float32; use 1e-6 for double)
grad_numerical[i] = (f(x + h*e_i) - f(x - h*e_i)) / (2*h)
max |grad_analytical - grad_numerical| < 1e-4
```

Where to check:
- One hand-picked small example per op (e.g. `{3}` shape for elementwise, `{2,2}` for matmul)
- One broadcast case per op that uses broadcasting
- One batched case for `matmul`

Tolerance rationale: float32 has ~7 decimal digits. Central difference error is O(h²) = O(1e-8)
for h=1e-4, well within float32 precision. The 1e-4 tolerance gives a comfortable margin while
catching actual bugs.

### Incremental implementation steps

Each step = implement ? hand-written correctness test ? gradient check ? commit. Do not
batch steps into a single PR. A non-SOTA model can execute each step independently.

**Step 1 — `Variable` scaffold** (`include/torc/autograd.hpp`)
- `data_`, `grad_`, `requires_grad_`, `tape_` members
- `Variable(Tensor, bool)`, `Variable(float, bool)` constructors
- `Tensor& data()`, `const Tensor& data() const`
- `bool requires_grad() const`
- `void backward()` — walks tape in reverse, clears it
- `void zero_grad()` — clears `grad_`
- **Test**: construction, `requires_grad` flag, empty tape behavior

**Step 2 — Scalar-only autograd**
- Only rank-0 Variables
- Ops: `add`, `sub`, `mul`, `div`, `neg` (unary `-`), scalar variants
- Each backward closure: `input.grad_ += output.grad_ * local_grad`
- **Test**: hand-computed gradient for `x=3, y=4` through `x*y + x - y`
- **Grad check**: central difference, tolerance 1e-4, on each scalar op

**Step 3 — Broadcasting backward (elementwise tensor ops)**
- Reuse `Tensor::add/sub/mul/div` for forward pass
- Implement `reduce_sum_to_shape(grad, target_shape)` helper in `autograd.cpp`
- Each backward closure calls `reduce_sum_to_shape` on the per-input gradient before
  accumulating
- **Test**: dim-1 broadcast `{2,2} + {2,1}`, multi-dim `{3,1} + {1,3}`, higher-rank
- **Grad check**: at least one broadcast case per op

**Step 4 — Reduction ops backward**
- `sum()` (no axis): backward = `ones_like(input)` — gradient must broadcast back to
  input shape
- `mean()`: backward = `ones/nel` broadcasted back
- `sum(axis)` / `mean(axis)`: backward = `unsqueeze(axis) then ones broadcast`
- **Test**: correctness on `{2,3}` tensor, gradient check for whole-tensor and axis variants

**Step 4b — Defer `max`/`min` backward**
- Mark as `ShapeError("max/min backward not yet implemented")`
- Document that argmax-tracking is needed before these can be supported

**Step 5 — `matmul` backward**
- Forward: reuse `Tensor::matmul`
- Backward 2D: `dA = dC @ B^T`, `dB = A^T @ dC`
- Backward batched: same formula per batch element; use `transpose_last_two()`
- **Test**: hand-computed example `{2,3} @ {3,2}`, identity `I @ A = A`
- **Grad check**: 2D and batched matmul

**Step 6 — View ops backward**
- `transpose`: permute gradient with inverse axes
- `reshape`/`view`: reshape gradient back to input shape
- `slice`: zero-fill then scatter gradient into sliced region
- **Test**: gradient check for each view op

**Step 7 — `detach()` and `no_grad()` context**
- `Variable detach() const` — new Variable, same data, `requires_grad=false`, empty tape
- `no_grad()` / `set_grad_enabled(bool)` — global flag that ops check; when disabled,
  ops return plain Variables with empty tapes
- **Test**: verify `detach()` breaks graph, gradient doesn't flow back through detached path

**Step 8 — In-place ops: forbid for `requires_grad` Variables**
- `Tensor::operator[]` is in-place; `Variable` must not expose mutable `data()` when
  `requires_grad=true`, OR
- Add `check_inplace_allowed()` that throws `TorcError` if `requires_grad`
- **Test**: verify in-place write throws on tracked Variable

**Step 9 — `max`/`min` backward with argmax tracking**
- Forward: store `argmax` indices (per-axis or global) on the Variable
- Backward: scatter gradient to the max positions only
- **Test**: gradient check on `{2,3}` tensor with `max(0)` and `max(1)`

**Step 10 — Project layout restructure**
- `include/torc/autograd.hpp`, `src/autograd.cpp`, `tests/test_autograd.cpp`
- Update `CMakeLists.txt` to register the new source and test target

### Constraints to enforce from Day 1

1. **No in-place modification** of Variables that require grad (outside controlled backward
   closures)
2. **Tape is consumed** by `backward()` and cleared — calling `backward()` twice on the same
   Variable is a `TorcError`
3. **Leaf `.grad_` accumulates** via `+=`, not `=` — this enables gradient accumulation
   across mini-batches
4. **Non-scalar outputs** require explicit `grad_output` argument to `backward()`
5. **`requires_grad` defaults to `false`** — most tensors (data, labels) don't need tracking;
   only parameters opt in. This avoids wasting memory on ~90% of tensors in a typical
   training loop (see TinyTorch Module 06, Q4)

### Key risks identified from online sources

1. **Broadcasting backward bugs** (numpygrad docs, PyTorch forums): the single most common
   autograd mistake is forgetting to sum over broadcast dims. This must be tested at every
   broadcast case, not just one.

2. **In-place ops corrupting saved tensors** (PyTorch docs): if a user mutates an input after
   the forward pass, the saved-tensor copy in the backward closure is stale. We prevent this
   by forbidding in-place ops on tracked Variables (Step 8).

3. **Float32 numerical noise in gradient checks**: PyTorch's `gradcheck` defaults to double
   precision for this reason. Our `eps=1e-4` / `atol=1e-4` is calibrated for float32 and
   should be re-validated if precision ever changes.

---

## Planned future project structure

The current layout is intentionally flat because there's only a Tensor class. Once
autograd lands (Milestone 4) and nn/optim/data follow (Milestone 5), flat src//include/
stops scaling and main.cpp stops being a reasonable place for examples. Proposed layout,
to be adopted incrementally as each milestone actually lands (don't scaffold empty
directories ahead of the code that fills them):

`
torc/
+-- CMakeLists.txt
+-- LICENSE
+-- README.md
+-- AGENTS.md
+-- ROADMAP.md
+-- docs/
¦ +-- DESIGN.md
+-- include/
¦ +-- torc/
¦ +-- tensor.hpp
¦ +-- utils.hpp
¦ +-- autograd.hpp # Variable, Function/Node, backward() [Milestone 4]
¦ +-- nn.hpp # Module base + Linear, activations [Milestone 5]
¦ +-- optim.hpp # SGD, Adam [Milestone 5]
¦ +-- data.hpp # Dataset, DataLoader [Milestone 5]
+-- src/
¦ +-- tensor.cpp
¦ +-- autograd.cpp [Milestone 4]
¦ +-- nn/
¦ ¦ +-- linear.cpp [Milestone 5]
¦ ¦ +-- activations.cpp [Milestone 5]
¦ +-- optim.cpp [Milestone 5]
¦ +-- data.cpp [Milestone 5]
+-- examples/ # demo binaries move out of src/, main.cpp retired
¦ +-- basic_ops.cpp # today's main.cpp, relocated
¦ +-- linear_regression.cpp [Milestone 5]
¦ +-- mlp_classification.cpp [Milestone 5]
+-- tests/
 +-- test_tensor.cpp
 +-- test_autograd.cpp [Milestone 4]
 +-- test_nn.cpp [Milestone 5]
 +-- test_optim.cpp [Milestone 5]
`

Rationale for the specific moves:

- **
n/ as a subdirectory, not a single file, once it has =2 layer types.** A single
 
n.cpp is fine for Linear alone but won't stay readable once activations, losses, and
 more layers land in the same milestone.
- **examples/ instead of a single main.cpp.** AGENTS.md already says main.cpp should
 stay a thin demo, not a dumping ground — once there are two real end-to-end examples
 (linear regression, MLP classification) that stops being true of a single file. Multiple
 small example binaries keep each one focused and testable in isolation via CMake.
- **data.hpp/data.cpp are deliberately minimal at first** — a Dataset interface plus
 one synthetic or small CSV-backed implementation is enough to unblock the Milestone 5
 end-to-end examples. Don't build a general data pipeline before there's a second dataset
 that needs one.
- **CMakeLists.txt will need dd_subdirectory or explicit source lists per target** once
 
n/ exists as a folder, plus BUILD_EXAMPLES/BUILD_TESTS options if the example count
 grows enough that not everyone wants to compile all of them by default.

This structure is a target, not a mandate to create now — build it milestone-by-milestone so
each directory only exists once something real lives in it.
