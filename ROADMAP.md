# torc — Roadmap

Milestone checklist. Update this file as work lands — it's expected to churn. For *why*
decisions were made this way, see [docs/DESIGN.md](docs/DESIGN.md); for the autograd
algorithm specifically, see [docs/AUTOGRAD.md](docs/AUTOGRAD.md). For stable orientation,
see [AGENTS.md](AGENTS.md).

---

## Immediate next steps

1. Implement Milestone 4 **Step 4** (reduction ops backward — `sum`/`mean`, whole-tensor and
   axis-wise). Steps 1–3 are done; see the incremental step list in the Milestone 4 section
   below.
2. Each subsequent step must pass gradient checks (central finite difference, eps=1e-4,
   atol=1e-2 — see `docs/DESIGN.md`'s "Gradient checking" section for why atol is 1e-2, not
   1e-4) before the next step begins.

---

## Milestone 1 — Core tensor op completeness
- [x] `sub`, `div`
- [x] Unary negation (`operator-()`)
- [x] Scalar ops (`Tensor + float`, etc.)
- [x] `operator==`, `operator<<` (pretty-printing) for debugging/tests
- [x] Reductions: `sum()`, `mean()`, `max()`, `min()` (whole-tensor first, axis-wise later)
- [x] `reshape()` / `view()` — no-copy where possible
- [x] Copy vs. move semantics review (rule of 5, or confirm defaults are sufficient)

## Milestone 2 — Shape flexibility
- [x] NumPy-style broadcasting for elementwise ops
- [x] Multi-dimensional indexing helper (`operator[](i, j, ...)`)
- [x] `transpose()` / axis permutation
- [x] Slicing (basic contiguous-range slices)

## Milestone 3 — Linear algebra
- [x] `matmul()` for 2D tensors (naive triple loop first, correctness over speed)
- [x] Batched matmul for higher-rank tensors
- [x] Basic BLAS backend option (behind a CMake flag) once naive matmul is correct

## Milestone 4 — Autograd (incremental, tested each step)

Each step must pass both correctness tests and numerical gradient checks before proceeding.
Do not implement multiple steps in one PR.

- [x] Decide tape structure, ownership model, and broadcasting backward approach — see
      `docs/DESIGN.md` (rationale) and `docs/AUTOGRAD.md` (the actual algorithm: topological
      sort + raw-pointer graph, not per-Variable recursion)
- [x] **Step 1**: `Variable` scaffold (`data_`, `grad_`, `requires_grad_`, `has_grad_`,
      `tape_`) with `backward()`, `zero_grad()`
- [x] **Step 2**: Scalar-only autograd (`add`, `sub`, `mul`, `div`, `neg`) — hand tests + grad
      check (eps=1e-4, atol=1e-2)
- [x] **Step 3**: Tensor-tensor elementwise + broadcasting backward (`reduce_sum_to_shape`
      helper, applied centrally in `backward_with_grad`) — grad check on broadcast cases
- [ ] **Step 4**: Reduction ops backward (`sum`, `mean` whole-tensor and axis-wise) — grad check
- [ ] **Step 4b**: Defer `max`/`min` backward (argmax-tracking not yet implemented — throw
      `ShapeError`)
- [ ] **Step 5**: `matmul` backward (2D + batched, **including batch-broadcast cases** — these
      need their own reduce-over-batch-dims handling, not a direct reuse of
      `reduce_sum_to_shape`) — hand test + grad check
- [ ] **Step 6**: View ops backward (`transpose` — true inverse permutation, not re-applying
      `axes`; `reshape`/`view`; `slice` — needs a new zero-fill-and-scatter `Tensor`
      primitive that doesn't exist yet) — grad check
- [ ] **Step 7**: `detach()` and `no_grad()` / `set_grad_enabled(bool)` context
- [ ] **Step 8**: In-place ops forbidden for `requires_grad` Variables (throw `TorcError`) —
      needs a new guarded in-place API first, since direct `data()`/`operator[]` mutation
      can't be intercepted as-is
- [ ] **Step 9**: `max`/`min` backward with argmax tracking — grad check

## Milestone 5 — nn / optim / data (basic ML)
- [ ] `nn::Module` base class + `Linear`, common activations (ReLU, Sigmoid, Softmax)
  - Needs elementwise `exp` on `Tensor` first — no unary transcendental op exists today
    (only `add`/`sub`/`mul`/`div`/`neg`, reductions, and `matmul`); Softmax and Sigmoid both
    depend on it
- [ ] Loss functions (MSE, cross-entropy)
- [ ] `optim::SGD`, `optim::Adam`
- [ ] `data::Dataset` / `data::DataLoader` abstractions
- [ ] A toy dataset loader (start with a synthetic or small CSV dataset)
- [ ] End-to-end example: linear regression on a synthetic dataset
- [ ] End-to-end example: small MLP on a toy classification dataset (e.g. MNIST-subset)
- [ ] LayerNorm — needs per-row variance, which doesn't exist yet either (only
      `sum`/`mean`/`max`/`min`); can be derived as `mean(x^2) - mean(x)^2` once `mul` and
      `mean(axis)` are available, which they already are
- [ ] Embedding lookup / gather-by-index — `Tensor::slice()` is contiguous-range only today,
      no arbitrary-index gather
- [ ] End-to-end example: minimal transformer block (multi-head self-attention + feed-forward)
      on a toy sequence task — depends on the `exp`, LayerNorm, and embedding-lookup items
      above, plus `Softmax`/`matmul`/broadcasting already in place

## Milestone 6 — Performance & ergonomics
- [ ] Cache `numel()` at construction/reshape instead of recomputing
- [ ] SIMD or threaded elementwise ops (behind a flag, benchmark-gated)
- [ ] Optional CUDA/Metal backend exploration (stretch goal, only after CPU path is solid)

## Milestone 7 — Bindings & packaging
- [ ] Python bindings (pybind11) once the C++ API has stabilized
- [ ] Versioned releases / install target (`cmake --install`)
- [ ] CI (GitHub Actions: build + `ctest` on push)