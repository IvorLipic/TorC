# torc — Agent Guide

`torc` is a minimal, from-scratch C++ tensor library (naive `float32`-only storage) with a
CMake build, a small demo binary, and a unit test target. This document orients an agent
(human or AI) picking up the codebase: what exists today, what's missing, known issues,
and a concrete path forward.

---

## 1. Project layout

```
torc/
├── CMakeLists.txt
├── LICENSE                 # MIT
├── README.md
├── .gitignore
├── include/
│   └── torc/
│       ├── tensor.hpp      # Tensor class declaration
│       └── utils.hpp       # shape_product / shape_to_string helpers (not yet used by Tensor)
├── src/
│   ├── tensor.cpp          # Tensor implementation
│   └── main.cpp            # demo executable
└── tests/
    └── test_tensor.cpp     # GoogleTest suite, fetched via CMake FetchContent
```

Three build targets:
- **`torc`** — static/object library built from `src/tensor.cpp`.
- **`torc_demo`** — executable (`src/main.cpp`) linked against `torc`.
- **`torc_tests`** — GoogleTest executable (`tests/test_tensor.cpp`), links `torc`,
  registered via `enable_testing()` / `add_test(NAME TorcTests ...)`.

C++17. GoogleTest (`v1.14.0`) is pulled in for tests via `FetchContent`; no other external
dependencies.

---

## 2. Current functionality

`torc::Tensor` (see `include/torc/tensor.hpp`, `src/tensor.cpp`):

- Construct from an explicit shape (zero-filled): `Tensor(std::vector<int> shape)`
- Construct from a flat initializer list + shape, with a size-mismatch check that throws
  `ShapeError`
- `data()` / `const data()` — raw pointer access to underlying `std::vector<float> storage_`
- `shape()` — returns `const std::vector<int>&`
- `numel()` — product of shape dims (via `shape_product()`), recomputed each call (not cached)
- `add(other)`, `sub(other)`, `mul(other)`, `div(other)` — elementwise tensor-tensor,
  require identical shape, throw `ShapeError` on mismatch
- `add(scalar)`, `sub(scalar)`, `mul(scalar)`, `div(scalar)` — scalar overloads (each
  element op applied with a scalar)
- `operator==` — value + shape equality comparison
- `operator<<` (via `include/torc/utils.hpp`) — pretty-prints shape and data

`include/torc/utils.hpp` provides `TorcError` / `ShapeError` exception hierarchy,
`shape_product()`, and `shape_to_string()`, all used by `tensor.cpp`.

`src/main.cpp` demonstrates constructing two rank-1 tensors and running `add`/`mul`.

`tests/test_tensor.cpp` covers: zero-filled construction, initializer-list construction
(matching and mismatched sizes), shape accessors, mutable/const `data()` access,
`add`/`sub`/`mul`/`div` correctness, shape-mismatch throws for all four ops, scalar overloads,
`operator==`, `operator<<` output, custom exception types (`ShapeError`), and a multi-dim
(`{2,2}`) elementwise case.

That's the entire surface area today. Everything else (subtraction, division, matmul,
reshaping, broadcasting, reductions, printing, autograd, device support) does not exist yet.

---

## 3. Known issues / cleanup items

1. **`utils.hpp` helpers now integrated.** `tensor.cpp` uses `shape_product()` for
   `numel()` and constructors; `shape_to_string()` is used in error messages and
   `operator<<`.
2. **`numel()` recomputes on every call** instead of being cached at construction —
   fine for now, but will matter once tensors get large or `numel()` is called in hot loops.
3. **`README.md` now references the MIT LICENSE file.**
4. **No CI configured.** Build + `ctest` only run locally; no GitHub Actions workflow yet
   (tracked in Milestone 6 below).
5. **Error handling convention:** a small custom exception hierarchy is introduced —
   `TorcError` (base, derives from `std::runtime_error`) and `ShapeError` (for shape
   mismatches). All existing `std::runtime_error` throws are now `ShapeError`; since
   `ShapeError` derives from `std::runtime_error`, existing code catching
   `std::runtime_error` continues to work.

*(Previously tracked here: a duplicated `enable_testing()` block in `CMakeLists.txt`, and
`torc_tests` double-compiling `tensor.cpp` alongside linking `torc`. Both are resolved —
`CMakeLists.txt` now has a single `enable_testing()` call, and `torc_tests` only compiles
`tests/test_tensor.cpp` and links the `torc` library. GoogleTest is also now wired in via
`FetchContent`, and `.gitignore`/`README.md`/`LICENSE` all exist at the repo root.)*

---

## 4. Build & test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure   # runs TorcTests
./build/torc_demo
```

---

## 5. Design gaps to resolve early

Decisions worth making explicitly before the API surface grows, since they're expensive
to change later:

- **dtype strategy**: currently hardcoded `float32` (single-dtype only). Stay single-dtype
  (simplest) for now; revisit with a `dtype` enum / templated storage only if multi-precision
  becomes a need.
- **Shape/broadcasting semantics**: **Decision: no broadcasting for now.** All elementwise
  ops (`add`/`sub`/`mul`/`div`) require identical shapes and share a single `check_same_shape()`
  guard. NumPy-style broadcasting is deferred to Milestone 2, at which point the shape-check
  signature will be generalized to a `broadcast_shape()` helper.
- **Memory ownership**: `storage_` is a `std::vector<float>` owned per-`Tensor`, copied on
  every op (`Tensor out(shape_)` allocates fresh storage each time). Fine for a naive
  reference implementation; revisit if performance or a computation graph (autograd) is a
  goal, since autograd typically needs shared/ref-counted storage.
- **Error handling convention**: see §3 item 5 above. `TorcError` (base, `std::runtime_error`)
  and `ShapeError` (shape mismatches) are now in `utils.hpp`; all error sites use `ShapeError`.

---

## 6. Roadmap / milestones

### Milestone 0 — Hygiene
- [x] Dedupe the `enable_testing()` block in `CMakeLists.txt`
- [x] Fix `torc_tests` to link `torc` only, not recompile `tensor.cpp`
- [x] Pick and wire in a test framework (GoogleTest via `FetchContent`)
- [x] Add a top-level `README.md` (quickstart, build instructions)
- [x] Add `.gitignore` and `LICENSE` (MIT)
- [x] Fix stale license reference in `README.md` (§3 item 3)
- [x] Decide fate of `utils.hpp` helpers — integrated (§3 item 1)

### Milestone 1 — Core tensor op completeness
- [x] `sub`, `div`, unary negation
- [x] Scalar ops (`Tensor + float`, etc.)
- [x] `operator==`, `operator<<` (pretty-printing) for debugging/tests
- [ ] Reductions: `sum()`, `mean()`, `max()`, `min()` (whole-tensor first, axis-wise later)
- [ ] `reshape()` / `view()` — no-copy where possible
- [ ] Copy vs. move semantics review (rule of 5, or confirm defaults are sufficient)

### Milestone 2 — Shape flexibility
- [ ] NumPy-style broadcasting for elementwise ops
- [ ] Multi-dimensional indexing helper (`operator()(i, j, ...)`)
- [ ] `transpose()` / axis permutation
- [ ] Slicing (basic contiguous-range slices)

### Milestone 3 — Linear algebra
- [ ] `matmul()` for 2D tensors (naive triple loop first, correctness over speed)
- [ ] Batched matmul for higher-rank tensors
- [ ] Basic BLAS backend option (behind a CMake flag) once naive matmul is correct

### Milestone 4 — Autograd
- [ ] Decide on graph representation (tape-based vs. expression-tree)
- [ ] `requires_grad` flag + `backward()` for the ops already implemented
- [ ] Gradient checking tests (numerical vs. analytical) added alongside each op

### Milestone 5 — Performance & ergonomics
- [ ] Cache `numel()` at construction/reshape instead of recomputing
- [ ] SIMD or threaded elementwise ops (behind a flag, benchmark-gated)
- [ ] Optional CUDA/Metal backend exploration (stretch goal, only after CPU path is solid)

### Milestone 6 — Bindings & packaging
- [ ] Python bindings (pybind11) once the C++ API has stabilized
- [ ] Versioned releases / install target (`cmake --install`)
- [ ] CI (GitHub Actions: build + `ctest` on push)

---

## 7. Immediate next steps (this session or next)

1. ~~Fix the stale license line in `README.md`~~ — done, README now references MIT LICENSE.
2. ~~Adopt `utils.hpp`'s `shape_product`/`shape_to_string` in `tensor.cpp`~~ — done.
   `numel()` now calls `shape_product()`, constructors use it, and error messages/`operator<<`
   use `shape_to_string()`.
3. ~~Add `sub()` and scalar-op overloads~~ — done, plus `div()` for symmetry.
4. ~~Add `operator<<` for `Tensor`~~ — done, uses `shape_to_string()`.
5. **Broadcasting decision:** no broadcasting for now. All elementwise ops require identical
   shapes via a shared `check_same_shape()` guard. NumPy-style broadcasting deferred to
   Milestone 2.

---

## 8. Notes for future agents

- Keep `Tensor` naive-but-correct until Milestone 3; don't prematurely optimize storage or
  add a device abstraction before basic ops and autograd are in place.
- Every new op should ship with: a shape-mismatch test, a correctness test against a
  hand-computed example, and (once autograd lands) a gradient check.
- Prefer extending `tensor.hpp`/`tensor.cpp` over growing `main.cpp` — `main.cpp` should
  stay a thin demo, not become a dumping ground for feature examples.
