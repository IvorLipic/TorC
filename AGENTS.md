# torc — Agent Guide

`torc` is a minimal, from-scratch C++ tensor library (naive `float32`-only storage) with a
CMake build, a small demo binary, and a unit test target.

This file is the stable orientation doc: what exists, how to build/test it, and the
conventions to follow. It should change rarely.

- For what's next and progress tracking, see **[ROADMAP.md](ROADMAP.md)**.
- For *why* things are built the way they are (and where the project is headed
  architecturally, e.g. autograd/nn/optim), see **[docs/DESIGN.md](docs/DESIGN.md)**.

---

## 1. Project layout

```
torc/
├── CMakeLists.txt
├── LICENSE                 # MIT
├── README.md
├── AGENTS.md                # this file
├── ROADMAP.md                # milestone checklist, updated frequently
├── .gitignore
├── docs/
│   └── DESIGN.md             # design decisions + planned future structure
├── include/
│   └── torc/
│       ├── tensor.hpp      # Tensor class declaration
│       └── utils.hpp       # shape_product / shape_to_string / error types, used by Tensor
├── src/
│   ├── tensor.cpp          # Tensor implementation
│   └── main.cpp            # demo executable
└── tests/
    └── test_tensor.cpp     # GoogleTest suite, fetched via CMake FetchContent
```

Three build targets:
- **`torc`** — library built from `src/tensor.cpp`.
- **`torc_demo`** — executable (`src/main.cpp`) linked against `torc`.
- **`torc_tests`** — GoogleTest executable (`tests/test_tensor.cpp`), links `torc`,
  registered via `enable_testing()` / `add_test(NAME TorcTests ...)`.

C++17. GoogleTest (`v1.14.0`) is pulled in for tests via `FetchContent`; no other external
dependencies.

---

## 2. Build & test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure   # runs TorcTests
./build/torc_demo
```

---

## 3. Current functionality

`torc::Tensor` (see `include/torc/tensor.hpp`, `src/tensor.cpp`):

- Construct from an explicit shape (zero-filled): `Tensor(std::vector<int> shape)`
- Construct from a flat initializer list + shape, with a size-mismatch check that throws
  `ShapeError`
- `data()` / `const data()` — raw pointer access to underlying `std::vector<float> storage_`
- `shape()` — returns `const std::vector<int>&`
- `numel()` — product of shape dims (via `shape_product()`), recomputed each call (not cached)
- `add(other)`, `sub(other)`, `mul(other)`, `div(other)` — elementwise tensor-tensor with
  NumPy-style broadcasting; throws `ShapeError` if shapes are incompatible
- `add(scalar)`, `sub(scalar)`, `mul(scalar)`, `div(scalar)` — scalar overloads
- `operator-()` — unary negation (elementwise)
- `operator==` — value + shape equality comparison
- `operator<<` (via `include/torc/utils.hpp`) — pretty-prints shape and data
- `operator()(int, int, ...)` — multi-dimensional indexing with bounds checking
- `transpose(std::vector<int> axes)` — permute dimensions; default reverses axes
- `slice(std::vector<Slice>)` — basic contiguous-range slicing
- `sum()` / `mean()` / `max()` / `min()` — whole-tensor return `float`; axis-wise overloads
  (`int axis`) return `Tensor` with reduced shape
- `reshape()` / `view()` — change shape without copying data (moves flat `std::vector` storage)
- Rule of 5 is explicitly defaulted (copy/move ctor/assign + destructor), since the class only
  contains `std::vector` members; declared for clarity and to prevent accidental deletion if
  members change later.

`src/main.cpp` demonstrates constructing two rank-1 tensors and running the elementwise ops.

`tests/test_tensor.cpp` covers: zero-filled construction, initializer-list construction
(matching and mismatched sizes), shape accessors, mutable/const `data()` access,
`add`/`sub`/`mul`/`div` correctness, shape-mismatch throws for all four ops, scalar overloads,
`operator==`, `operator<<` output, custom exception types (`ShapeError`), a multi-dim
(`{2,2}`) elementwise case, unary negation, and reductions (`sum`/`mean`/`max`/`min`
whole-tensor and axis-wise), broadcasting (identical, dim-1, multi-dim, incompatible throws),
multi-dimensional indexing (`operator()`, bounds/rank checks, read/write), `transpose`
(default and custom axes, invalid throws), and `slice` (first/last/inner dim, combined,
out-of-range throws).

That's the entire surface area today. Everything else (matmul, autograd, nn layers, optimizers,
data loading, device support) does not exist yet — see ROADMAP.md.

---

## 4. Conventions

- **Single dtype**: `float32` only, hardcoded. Don't introduce a `dtype` enum or templated
  storage without an explicit design decision in `docs/DESIGN.md`.
- **Broadcasting**: elementwise ops use NumPy-style broadcasting via `broadcast_shape()`.
  Follow the existing pattern — don't add ad-hoc shape coercion to individual ops.
- **Error handling**: throw `ShapeError` (or a new `TorcError` subclass) for library errors,
  never a bare `std::runtime_error`. `ShapeError` derives from `TorcError` derives from
  `std::runtime_error`, so existing `catch (std::runtime_error&)` sites keep working.
- **Ownership**: `Tensor::storage_` is an owned `std::vector<float>`, copied fresh on every
  op. This is intentionally naive; don't add shared/ref-counted storage without updating
  `docs/DESIGN.md` first (autograd will likely force this decision — see there).
- **Keep `main.cpp`/`examples/` thin**: demos should showcase the API, not be where new
  features get exercised for the first time. Every new op ships with tests in `tests/`.
- **Every new op needs**: a shape-mismatch test (if applicable), a correctness test against
  a hand-computed example, and — once autograd lands — a gradient check.

---

## 5. Known issues

1. `numel()` recomputes on every call instead of being cached at construction — fine for now,
   will matter once tensors get large or `numel()` is called in hot loops.
2. No CI configured — build + `ctest` only run locally, no GitHub Actions workflow yet.
