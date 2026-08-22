# torc — Roadmap

Milestone checklist. Update this file as work lands — it's expected to churn. For *why*
decisions were made this way, see [docs/DESIGN.md](docs/DESIGN.md). For stable orientation,
see [AGENTS.md](AGENTS.md).

---

## Immediate next steps

1. Implement Milestone 4 Step 1 (`Variable` scaffold) — this is the foundation for everything
   that follows. See the incremental step list in the Milestone 4 section below.
2. Each subsequent step must pass gradient checks (central finite difference, eps=1e-4,
   atol=1e-4) before the next step begins.

---

## Milestone 0 — Hygiene
- [x] Dedupe the `enable_testing()` block in `CMakeLists.txt`
- [x] Fix `torc_tests` to link `torc` only, not recompile `tensor.cpp`
- [x] Pick and wire in a test framework (GoogleTest via `FetchContent`)
- [x] Add a top-level `README.md` (quickstart, build instructions)
- [x] Add `.gitignore` and `LICENSE` (MIT)
- [x] Fix stale license reference in `README.md`
- [x] Decide fate of `utils.hpp` helpers — integrated into `tensor.cpp`
- [x] Split `AGENTS.md` into `AGENTS.md` / `ROADMAP.md` / `docs/DESIGN.md`

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

- [x] Decide tape structure, ownership model, and broadcasting backward approach — see `docs/DESIGN.md`
- [x] **Step 1**: `Variable` scaffold (`data_`, `grad_`, `requires_grad_`, `tape_`) with `backward()`, `zero_grad()`
- [x] **Step 2**: Scalar-only autograd (`add`, `sub`, `mul`, `div`, `neg`) — hand tests + grad check (eps=1e-4, atol=1e-4)
- [x] **Step 3**: Tensor-tensor elementwise + broadcasting backward (`reduce_sum_to_shape` helper) — grad check on broadcast cases
- [ ] **Step 4**: Reduction ops backward (`sum`, `mean` whole-tensor and axis-wise) — grad check
- [ ] **Step 4b**: Defer `max`/`min` backward (argmax-tracking not yet implemented — throw `ShapeError`)
- [ ] **Step 5**: `matmul` backward (2D + batched) — hand test + grad check
- [ ] **Step 6**: View ops backward (`transpose`, `reshape`/`view`, `slice`) — grad check
- [ ] **Step 7**: `detach()` and `no_grad()` / `set_grad_enabled(bool)` context
- [ ] **Step 8**: In-place ops forbidden for `requires_grad` Variables (throw `TorcError`)
- [ ] **Step 9**: `max`/`min` backward with argmax tracking — grad check
- [ ] **Step 10**: Restructure to `include/torc/autograd.hpp`, `src/autograd.cpp`, `tests/test_autograd.cpp`

## Milestone 5 — nn / optim / data (basic ML)
- [ ] `nn::Module` base class + `Linear`, common activations (ReLU, Sigmoid, Softmax)
- [ ] Loss functions (MSE, cross-entropy)
- [ ] `optim::SGD`, `optim::Adam`
- [ ] `data::Dataset` / `data::DataLoader` abstractions
- [ ] A toy dataset loader (start with a synthetic or small CSV dataset)
- [ ] End-to-end example: linear regression on a synthetic dataset
- [ ] End-to-end example: small MLP on a toy classification dataset (e.g. MNIST-subset)

## Milestone 6 — Performance & ergonomics
- [ ] Cache `numel()` at construction/reshape instead of recomputing
- [ ] SIMD or threaded elementwise ops (behind a flag, benchmark-gated)
- [ ] Optional CUDA/Metal backend exploration (stretch goal, only after CPU path is solid)

## Milestone 7 — Bindings & packaging
- [ ] Python bindings (pybind11) once the C++ API has stabilized
- [ ] Versioned releases / install target (`cmake --install`)
- [ ] CI (GitHub Actions: build + `ctest` on push)
