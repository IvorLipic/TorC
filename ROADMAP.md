# torc — Roadmap

Milestone checklist. Update this file as work lands — it's expected to churn. For *why*
decisions were made this way, see [docs/DESIGN.md](docs/DESIGN.md); for the autograd
algorithm specifically, see [docs/AUTOGRAD.md](docs/AUTOGRAD.md). For stable orientation,
see [AGENTS.md](AGENTS.md).

---

## Immediate next steps

1. Milestone 4 is complete. See the Milestone 5 section below for next steps.
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
- [x] `matmul()` for 2D tensors (cache-blocked tiling with 32×32×32 tiles, `i, k, j` loop order)
- [x] Batched matmul for higher-rank tensors

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
- [x] **Step 4**: Reduction ops backward (`sum`, `mean` whole-tensor and axis-wise) — grad check
- [x] **Step 4b**: Defer `max`/`min` backward (argmax-tracking not yet implemented — throw
      `ShapeError`)
- [x] **Step 5**: `matmul` backward (2D + batched, **including batch-broadcast cases** — these
      need their own reduce-over-batch-dims handling, not a direct reuse of
      `reduce_sum_to_shape`) — hand test + grad check
- [x] **Step 6**: View ops backward (`transpose` — true inverse permutation, not re-applying
      `axes`; `reshape`/`view`; `slice` — zero-fill-and-scatter into original shape) — grad check
- [x] **Step 7**: `detach()` and `no_grad()` / `set_grad_enabled(bool)` context
- [x] **Step 8**: In-place ops forbidden for `requires_grad` Variables (throw `TorcError`) —
      new guarded in-place API (`Tensor::fill`, `Variable::fill`) added; direct `data()`/`operator[]`
      mutation remains unguarded
- [x] **Step 9**: `max`/`min` backward with argmax tracking — grad check
## Milestone 5 — nn / optim / data (basic ML)

Each step must pass tests before proceeding.

- [x] **Step 5.1**: `Tensor::exp()` — elementwise unary transcendental op; needed before any
      activation that uses it (Sigmoid, Softmax)
- [x] **Step 5.2**: `nn::Module` base class — parameter storage, `forward()` hook, and
      `torch::nn::Sequential`-style container
- [x] **Step 5.3**: `nn::Linear` — `Linear(in, out)` with weight/bias parameters and a forward
  that does `x @ W.T + b`. `Linear(in, out, init_std)` overload: when `init_std <= 0`,
  weight is initialized with Kaiming normal (`std::sqrt(2 / fan_in)`); otherwise samples
  weight from `N(0, init_std)`. Bias is always initialized to `0.0`.
- [x] **Step 5.3a**: Fix `Module::forward` / `Module::operator()` API so intermediate `Variable`s
       created inside `forward()` stay alive until after `backward()` completes. Solution:
       `Module` owns a `mutable std::list<Variable> forward_cache_`; `operator()()` clears it
       before calling `forward()`, and each module's `forward()` appends intermediates via
       `emplace_back` so their addresses are stable for tape-entry raw pointers.
       `Sequential::forward()` calls each child's `operator()()` to prevent unbounded cache
       growth.
- [x] **Step 5.4**: Activation functions — `nn::ReLU`, `nn::Sigmoid`, `nn::Softmax` (each as a
      `Module` with differentiable forward, gradient checks included)
- [x] **Step 5.5**: Loss functions — `nn::MSELoss`, `nn::CrossEntropyLoss` (implemented as
      standalone classes in `nn/losses.hpp` / `nn/losses.cpp` with `forward(input, target)`
      signature and custom backward closures)
- [x] **Step 5.6**: Optimizer — `optim::SGD` with momentum (`optim::SGD` takes
      `std::vector<Variable*>&` and mutates `param->data()` in-place via `step()`; `zero_grad()`
      clears all parameter gradients)
- [x] **Step 5.7**: Optimizer — `optim::Adam` (with bias correction, `beta1=0.9`, `beta2=0.999`)
- [x] **Step 5.8**: Optimizer — `optim::AdamW`
- [x] **Step 5.9**: `data::Dataset` base + `data::DataLoader` (batching, shuffling)
- [x] **Step 5.10**: Toy dataset loaders — synthetic regression dataset, then a small CSV-backed
      classification dataset
- [x] **Step 5.11**: End-to-end example — linear regression on a synthetic dataset
- [x] **Step 5.12**: End-to-end example — small MLP on MNIST
- [ ] **Step 5.13**: `nn::LayerNorm` — per-row variance (`mean(x^2) - mean(x)^2`) using
      existing `mul`/`mean(axis)`; test on `{batch, features}` tensors
- [ ] **Step 5.14**: Embedding lookup / gather-by-index — `Tensor::slice()` is contiguous-range
      only today, so this needs either an arbitrary-index gather primitive or a one-hot +
      `matmul` workaround
- [ ] **Step 5.15**: End-to-end example — minimal transformer block (multi-head self-attention +
      feed-forward) on a toy sequence task; depends on `exp`, LayerNorm, embedding-lookup,
      Softmax, `matmul`, and broadcasting all being in place

## Milestone 6 — Performance & ergonomics

- [x] Benchmark infrastructure (Google Benchmark, `BUILD_BENCHMARKS=OFF` default)
- [x] Cache `numel()` at construction/reshape instead of recomputing
- [x] SIMD or threaded elementwise ops (behind a flag, benchmark-gated)
- [x] `matmul` cache-blocked tiling (32×32×32 tiles, `i, k, j` loop order) shipped in Milestone 3
- [x] Contiguous same-shape fast paths for elementwise binary ops (dispatch to `simd::add`/`sub`/`mul`/`div` directly, bypassing index reconstruction)
- [ ] Vectorize matmul inner loop with AVX2/FMA for dense throughput
- [x] Remove sparsity early-exit from matmul inner loop (dense throughput prioritized)
- [ ] Replace `std::vector<int>` index reconstruction in hot loops with stack-allocated stride iteration
- [ ] Optional CUDA/Metal backend exploration (stretch goal, only after CPU path is solid)

## Milestone 7 — Bindings & packaging
- [ ] Python bindings (pybind11) once the C++ API has stabilized
- [ ] Versioned releases / install target (`cmake --install`)
- [ ] CI (GitHub Actions: build + `ctest` on push)

## Review-derived hardening backlog

Before treating the library as production-ready, the backlog below must be addressed. These items
are separate from feature milestones: adding more layers on top of unsafe or numerically fragile
primitives would make the eventual fixes more expensive.

### Correctness and API safety

- [x] **Make softmax axis-aware.** Added `Tensor::softmax(axis)` and `torc::softmax(a, axis)` with
      stable per-axis forward/backward rules. `nn::Softmax(axis = -1)` now defaults to the last
      axis, so `{batch, classes}` inputs normalize each row. The no-argument primitive remains
      the documented flattened legacy behavior for compatibility. Batched forward and gradient
      tests cover the new contract.
- [x] **Harden `CrossEntropyLoss`.** The loss now requires rank-2, non-empty logits and rank-1
      targets whose length matches the batch. It rejects non-finite logits, non-integral/NaN,
      negative, and out-of-range targets before indexing. Forward uses per-row stable log-sum-exp
      (with double-precision accumulation), and malformed-input/extreme-logit tests are covered.
- [x] **Define numerical-domain and empty-tensor behavior.** `NumericalError` now rejects
      non-finite values, non-positive `log` inputs, negative `sqrt` inputs, and division by zero.
      Empty `sum()` returns its additive identity (`0`); `mean`/`max`/`min`, axis reductions, and
      softmax reject empty tensors with `ShapeError`. Regression tests cover each contract.
- [ ] **Remove raw-pointer graph lifetime UB.** Replace or wrap `TapeEntry.inputs`' raw,
      non-owning `Variable*` pointers with an ownership-safe graph representation, or introduce
      an API that makes graph ownership/lifetime impossible to misuse. Add returned-graph and
      destroyed-intermediate tests.
- [ ] **Encapsulate `Variable` state.** Make `data_`, `grad_`, `requires_grad_`, `has_grad_`, and
      `tape_` private and expose only invariant-preserving operations.
- [ ] **Close tracked-mutation loopholes.** Prevent or version-check mutable `data()` and indexing
      on tracked Variables; the current `fill()` guard does not protect direct writes after
      forward. Add mutation-after-forward tests.
- [ ] **Make gradient mode scoped and thread-local.** Add an RAII `no_grad` guard and thread-local
      state so nested scopes, exceptions, and concurrent inference cannot leak gradient state.
- [ ] **Validate optimizers and parameter ownership.** Reject invalid learning rates, momentum,
      beta, epsilon, and decay values; detect null/stale/shape-changing parameter pointers; and
      define behavior when a module parameter map changes after optimizer construction.
- [ ] **Validate dataset contracts.** Check sample-shape consistency in generic batching and guard
      all `size_t`-to-`int` conversions and shape-product arithmetic against overflow. Align data
      loader errors with the project's TorcError hierarchy.

### Performance and portability

- [ ] Replace per-output-element index-vector allocation in broadcast elementwise kernels with
      stride/odometer iteration.
- [ ] Reduce avoidable full-buffer copies from `reshape`/`view`, `transpose`, `slice`, optimizer
      updates, and autograd saved tensors; introduce strides or an explicitly documented memory
      budget before scaling model sizes.
- [ ] Add a portable SIMD dispatch/build mode. `-march=native` and unconditional AVX2 options
      currently make binaries dependent on the build CPU; retain optimized paths without making
      the default artifact non-portable.
- [ ] Add checked shape-product/element-count arithmetic so large shapes fail predictably instead
      of overflowing `int` and producing invalid allocations or indexing.

### Verification, packaging, and scope

- [ ] Add Debug/Release sanitizer jobs (ASan/UBSan where supported), a clean compiler matrix, and
      CI that configures and rebuilds from scratch rather than relying on a pre-existing build
      directory. Keep at least one test target per logical suite in addition to the aggregate test.
- [ ] Add property/fuzz tests for broadcasting, matmul batch promotion, malformed losses/data, and
      empty/zero-sized shapes; the current 252 example-oriented tests do not cover these hazards.
- [ ] Add an install target, package/version metadata, and a dependency strategy that does not
      require an implicit network fetch for every clean build.
- [ ] Reconcile roadmap/design status with the implementation after each performance change;
      the matmul AVX2 and sparsity items currently contain stale or contradictory checklist text.
- [ ] Decide and document the intended scope for train/eval modes, serialization/state dicts,
      operator ergonomics, higher-order gradients, and device backends. Until then, describe torc
      as a small educational CPU reference library rather than a general PyTorch replacement.
