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
- [x] `matmul()` for 2D tensors (naive triple loop first, correctness over speed)
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
  weight is initialized with Kaiming normal (`std::sqrt(2 / fan_in)`); otherwise uses
  `init_std` as the fixed standard deviation. Bias is always initialized to `0.0`.
- [x] **Step 5.3a**: Fix `Module::forward` / `Module::operator()` API so intermediate `Variable`s
      created inside `forward()` stay alive until after `backward()` completes. Solution:
      `Module` owns a `mutable std::list<Variable> forward_cache_`; `operator()()` clears it
      before calling `forward()`, and each module's `forward()` appends intermediates via
      `emplace_back` so their addresses are stable for tape-entry raw pointers.
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
- [ ] Remove sparsity early-exit from matmul inner loop if dense performance is prioritized
- [ ] Replace `std::vector<int>` index reconstruction in hot loops with stack-allocated stride iteration
- [ ] Optional CUDA/Metal backend exploration (stretch goal, only after CPU path is solid)

## Milestone 7 — Bindings & packaging
- [ ] Python bindings (pybind11) once the C++ API has stabilized
- [ ] Versioned releases / install target (`cmake --install`)
- [ ] CI (GitHub Actions: build + `ctest` on push)