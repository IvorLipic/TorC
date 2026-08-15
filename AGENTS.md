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
├── include/
│   └── torc/
│       ├── tensor.hpp      # Tensor class declaration
│       └── utils.hpp       # empty stub — no content yet
├── src/
│   ├── tensor.cpp          # Tensor implementation
│   └── main.cpp            # demo executable
└── tests/
    └── test_tensor.cpp     # referenced by CMake, not reviewed in this pass
```

Two build targets:
- **`torc`** — static/object library built from `src/tensor.cpp`.
- **`torc_demo`** — executable (`src/main.cpp`) linked against `torc`.
- **`torc_tests`** — test executable, links `torc`, registered via `enable_testing()` / `add_test(NAME TorcTests ...)`.

C++17, no external dependencies declared (no fetched GoogleTest, no BLAS, nothing).

---

## 2. Current functionality

`torc::Tensor` (see `include/torc/tensor.hpp`, `src/tensor.cpp`):

- Construct from an explicit shape (zero-filled): `Tensor(std::vector<int> shape)`
- Construct from a flat initializer list + shape, with a size-mismatch check that throws
  `std::runtime_error`
- `data()` / `const data()` — raw pointer access to underlying `std::vector<float> storage_`
- `shape()` — returns `const std::vector<int>&`
- `numel()` — product of shape dims, recomputed each call (not cached)
- `add(other)`, `mul(other)` — elementwise, **require identical shape** (no broadcasting),
  throw on mismatch

`src/main.cpp` demonstrates constructing two rank-1 tensors and running `add`/`mul`.

That's the entire surface area today. Everything else (subtraction, division, matmul,
reshaping, broadcasting, reductions, printing, autograd, device support) does not exist yet.

---

## 3. Known issues / cleanup items

These are small but worth fixing before building further on top:

1. **`CMakeLists.txt` has a duplicated block.** `enable_testing()` and its comment header
   appear twice in a row (lines ~15–21). Harmless (idempotent call) but should be
   deduplicated.
2. **`torc_tests` recompiles `src/tensor.cpp` directly** instead of linking the `torc`
   library target only — it does both (`src/tensor.cpp` in sources *and*
   `target_link_libraries(torc_tests PRIVATE torc)`), which double-compiles the same
   translation unit into the test binary. Pick one (prefer linking `torc`, drop the direct
   source).
3. **`include/torc/utils.hpp` is an empty file.** Either populate it (e.g. shape utilities,
   errors, printing helpers) or remove it until there's real content, to avoid confusion.
4. **No test framework wired in.** `tests/test_tensor.cpp` exists per `CMakeLists.txt` but
   there's no GoogleTest/Catch2/doctest dependency fetched — worth confirming what
   framework (if any) it currently uses, or standardizing on one (Catch2 header-only or
   GoogleTest via `FetchContent` are both good low-friction choices).
5. **`numel()` recomputes on every call** instead of being cached at construction —
   fine for now, but will matter once tensors get large or `numel()` is called in hot loops.
6. **No `.gitignore` / no README** observed — worth adding a top-level `README.md`
   distinct from this agent guide, for human-facing quickstart instructions.

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

- **dtype strategy**: currently hardcoded `float32`. Decide now whether to stay
  single-dtype (simplest) or plan for a `dtype` enum / templated storage before more ops
  are added on top of the float-only assumption.
- **Shape/broadcasting semantics**: `add`/`mul` currently require exact shape equality.
  NumPy-style broadcasting is a common expectation — decide early whether to support it,
  since it changes the shape-checking code path used by every future op.
- **Memory ownership**: `storage_` is a `std::vector<float>` owned per-`Tensor`, copied on
  every op (`Tensor out(shape_)` allocates fresh storage each time). Fine for a naive
  reference implementation; revisit if performance or a computation graph (autograd) is a
  goal, since autograd typically needs shared/ref-counted storage.
- **Error handling convention**: currently `std::runtime_error` for both shape mismatch
  cases. Worth deciding on a small custom exception hierarchy (`ShapeError`, `TorcError`)
  once error sites multiply.

---

## 6. Roadmap / milestones

### Milestone 0 — Hygiene (small, do first)
- [ ] Dedupe the `enable_testing()` block in `CMakeLists.txt`
- [ ] Fix `torc_tests` to link `torc` only, not recompile `tensor.cpp`
- [ ] Pick and wire in a test framework (Catch2 or GoogleTest via `FetchContent`)
- [ ] Add a top-level `README.md` (quickstart, build instructions, license)
- [ ] Decide fate of `utils.hpp` (populate or delete)

### Milestone 1 — Core tensor op completeness
- [ ] `sub`, `div`, unary negation
- [ ] Scalar ops (`Tensor + float`, etc.)
- [ ] `operator==`, `operator<<` (pretty-printing) for debugging/tests
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

1. Fix the two `CMakeLists.txt` issues in §3 (items 1–2) — quick, unblocks trusting `ctest`.
2. Review `tests/test_tensor.cpp` content directly (not included in this pass) and confirm
   what's actually covered vs. what Milestone 0/1 items still need test coverage.
3. Add `sub()` and scalar-op overloads to `Tensor` — smallest next increment that keeps the
   API internally consistent with `add`/`mul`.
4. Add an `operator<<` for `Tensor` — makes every future debugging session faster and is a
   prerequisite for writing readable test assertions.
5. Write down the broadcasting decision (§5) before Milestone 2 work starts, since it
   affects the shape-check signature used everywhere.

---

## 8. Notes for future agents

- Keep `Tensor` naive-but-correct until Milestone 3; don't prematurely optimize storage or
  add a device abstraction before basic ops and autograd are in place.
- Every new op should ship with: a shape-mismatch test, a correctness test against a
  hand-computed example, and (once autograd lands) a gradient check.
- Prefer extending `tensor.hpp`/`tensor.cpp` over growing `main.cpp` — `main.cpp` should
  stay a thin demo, not become a dumping ground for feature examples.