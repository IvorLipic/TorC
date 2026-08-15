# torc — Roadmap

Milestone checklist. Update this file as work lands — it's expected to churn. For *why*
decisions were made this way, see [docs/DESIGN.md](docs/DESIGN.md). For stable orientation,
see [AGENTS.md](AGENTS.md).

---

## Immediate next steps

1. Pick reductions API shape (`sum()`/`mean()`/`max()`/`min()` whole-tensor first) and land
   it with tests.
2. Start the Milestone 4 design conversation (tape vs. expression-tree for autograd) in
   `docs/DESIGN.md` *before* writing `Variable`/`Function` code — this is the biggest
   architectural fork in the project so far.

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
- [ ] `matmul()` for 2D tensors (naive triple loop first, correctness over speed)
- [ ] Batched matmul for higher-rank tensors
- [ ] Basic BLAS backend option (behind a CMake flag) once naive matmul is correct

## Milestone 4 — Autograd
- [ ] Decide on graph representation (tape-based vs. expression-tree) — see `docs/DESIGN.md`
- [ ] `Variable` wrapping `Tensor` + grad + backward closure
- [ ] `requires_grad` flag + `backward()` for the ops already implemented
- [ ] Gradient checking tests (numerical vs. analytical) added alongside each op
- [ ] Restructure `include/torc/` and `src/` per the future-layout plan in `docs/DESIGN.md`

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
