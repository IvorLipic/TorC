# torc — Agent Guide

`torc` is a minimal, from-scratch C++ tensor library (naive `float32`-only storage) with a
CMake build, a small demo binary, a tape-based autograd system (Milestone 4, in progress),
and a unit test target.

This file is the stable orientation doc: what exists, how to build/test it, and the
conventions to follow. It should change rarely.

- For what's next and progress tracking, see **[ROADMAP.md](ROADMAP.md)**.
- For *why* things are built the way they are (and where the project is headed
  architecturally, e.g. nn/optim/data), see **[docs/DESIGN.md](docs/DESIGN.md)**.
- For the autograd algorithm specifically (tape structure, topological-sort backward,
  broadcasting), see **[docs/AUTOGRAD.md](docs/AUTOGRAD.md)**.

---

## 1. Project layout

```
torc/
├── CMakeLists.txt
├── LICENSE                    # MIT
├── README.md
├── AGENTS.md                   # this file
├── ROADMAP.md                   # milestone checklist, updated frequently
├── .gitignore
├── docs/
│   ├── DESIGN.md                # design decisions + planned future structure
│   └── AUTOGRAD.md              # autograd algorithm details
├── include/
│   └── torc/
│       ├── tensor.hpp          # Tensor class declaration
│       ├── utils.hpp           # shape_product / shape_to_string / error types, used by Tensor
│       └── autograd.hpp        # Variable class + free-function ops (add, mul, ...)
├── src/
│   ├── tensor.cpp              # Tensor implementation
│   ├── autograd.cpp            # Variable op implementations (backward closures)
│   ├── matmul_blas.cpp         # CBLAS-backed matmul, compiled only when TORC_USE_BLAS=ON
│   └── main.cpp                # demo executable
└── tests/
    ├── test_tensor.cpp         # GoogleTest suite for Tensor
    └── test_autograd.cpp       # GoogleTest suite for Variable / autograd
```

Three build targets:
- **`torc`** — library built from `src/tensor.cpp` and `src/autograd.cpp` (plus
  `src/matmul_blas.cpp` when configured with `-DTORC_USE_BLAS=ON`).
- **`torc_demo`** — executable (`src/main.cpp`) linked against `torc`.
- **`torc_tests`** — GoogleTest executable (`tests/test_tensor.cpp` +
  `tests/test_autograd.cpp`), links `torc`, registered via `enable_testing()` /
  `add_test(NAME TorcTests ...)`.

C++23. GoogleTest (`v1.14.0`) is pulled in for tests via `FetchContent`; the only other
dependency is an optional CBLAS library (OpenBLAS / MKL / Accelerate), needed only when
`TORC_USE_BLAS=ON`.

---

## 2. Build & test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure   # runs TorcTests (Tensor + Variable suites)
./build/torc_demo
```

For the optional BLAS-backed `matmul`, see README.md's "With CBLAS backend" section
(`-DTORC_USE_BLAS=ON`).

---

## 3. Current functionality

### `torc::Tensor` (see `include/torc/tensor.hpp`, `src/tensor.cpp`)

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
- `operator[](int, int, ...)` — multi-dimensional indexing with bounds checking
- `transpose(std::vector<int> axes)` — permute dimensions; default reverses axes
- `slice(std::vector<Slice>)` — basic contiguous-range slicing (no arbitrary-index gather yet)
- `matmul(other)` — matrix multiplication; handles 2D and batched (higher-rank) tensors with
  NumPy-style broadcasting of leading batch dims; operands must be rank >= 2, throws `ShapeError`
  on rank < 2, inner-dimension mismatch, or incompatible batch dims. Naive triple-loop by
  default; CBLAS-backed (`cblas_sgemm` per batch element) when built with `-DTORC_USE_BLAS=ON`
- `sum()` / `mean()` / `max()` / `min()` — whole-tensor return `float`; axis-wise overloads
  (`int axis`) return `Tensor` with reduced shape. No `keepdim` option, no variance/std yet
- `reshape()` / `view()` — change shape without copying data (moves flat `std::vector` storage)
- No elementwise unary transcendental ops yet (`exp`, `log`, etc.) — needed before Milestone 5
  activations (Sigmoid, Softmax) can land, see ROADMAP.md
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
multi-dimensional indexing (`operator[]`, bounds/rank checks, read/write), `transpose`
(default and custom axes, invalid throws), `slice` (first/last/inner dim, combined,
out-of-range throws), and `matmul` (2D correctness, identity, inner-dim/rank mismatch throws,
transpose/associative properties, batched with distinct matrices, batch broadcasting via rank
promotion and size-1 dims, incompatible batch throws, and zero-size dimensions).

### `torc::Variable` (see `include/torc/autograd.hpp`, `src/autograd.cpp`) — Milestone 4, in progress

- Wraps a `Tensor` with `requires_grad`/`has_grad` flags and a `tape_` of `TapeEntry` recording
  the op that produced it (see `docs/AUTOGRAD.md` for the full tape structure and why backward
  uses an explicit topological sort rather than per-Variable recursion)
- `Variable(Tensor, bool)`, `Variable(float, bool)` constructors
- `data()`, `grad()`, `requires_grad()`, `has_grad()`, `backward()`, `backward(grad_output)`,
  `zero_grad()`
- Ops are **free functions in `namespace torc`**, not `Variable` methods — e.g.
  `torc::add(a, b)`, `torc::mul_scalar(a, b)`, not `a.add(b)`. Follow this pattern for any new
  op; do not add op methods onto `Variable` itself
- Implemented so far: scalar-only `add_scalar`/`sub_scalar`/`mul_scalar`/`div_scalar`/
  `neg_scalar` (Step 2), tensor-tensor `add`/`sub`/`mul`/`div`/`neg` with broadcasting
  (Step 3), reduction ops backward (`sum`/`mean`, whole-tensor and axis-wise) with grad
  checks (Step 4), `max`/`min` backward with argmax tracking (whole-tensor and axis-wise,
  Step 9), `matmul` backward (2D + batched with batch-broadcast cases, Step 5), view-op
  backward (`transpose`, `reshape`/`view`, `slice`, Step 6), `detach()`/`no_grad()` /
  `set_grad_enabled(bool)` (Step 7), and guarded in-place ops (`Tensor::fill`,
  `Variable::fill` throws `TorcError` on tracked Variables, Step 8). Broadcasting
  reduction (`reduce_sum_to_shape`) is applied **centrally** in
  `Variable::backward_with_grad`, not inside individual backward closures — don't duplicate it
  in a new op's closure
- **Not yet implemented**: Milestone 5 (`nn`/`optim`/`data`) — see ROADMAP.md for exact
  status
- **Lifetime constraint (unenforced by the type system):** `TapeEntry.inputs` holds raw,
  non-owning `Variable*` pointers into the actual input `Variable`s, not copies. Every
  `Variable` participating in a graph must outlive `backward()` on any of that graph's
  outputs — not just the final loss Variable. See §5 and `docs/DESIGN.md`.

`tests/test_autograd.cpp` covers: `Variable` scaffold (construction from `Tensor`/scalar,
`requires_grad`/`has_grad` flags, `zero_grad`, `backward()` no-op when untracked, non-scalar
`backward()` throws, `grad_output` shape-mismatch throws, empty-tape backward), scalar autograd
(`add`/`sub`/`mul`/`div`/`neg` hand-computed gradients, chained ops, gradient accumulation and
`zero_grad` reset, non-tracked inputs get no grad, tape is consumed after `backward()`, hand-
computed polynomial checks), and tensor-tensor elementwise autograd (identical-shape and
broadcast cases for `add`/`sub`/`mul`/`div`/`neg`, non-tracked-input handling, and numerical
gradient checks against central finite differences for the broadcast cases).

Everything else — `nn` layers, optimizers, data loading, device support — does not exist yet.
See ROADMAP.md.

---

## 4. Conventions

- **Single dtype**: `float32` only, hardcoded. Don't introduce a `dtype` enum or templated
  storage without an explicit design decision in `docs/DESIGN.md`.
- **Broadcasting**: elementwise ops use NumPy-style broadcasting via `broadcast_shape()`.
  Follow the existing pattern — don't add ad-hoc shape coercion to individual ops.
- **Error handling**: throw `ShapeError` (or a new `TorcError` subclass) for library errors,
  never a bare `std::runtime_error`. `ShapeError` derives from `TorcError` derives from
  `std::runtime_error`, so existing `catch (std::runtime_error&)` sites keep working.
- **Ownership (`Tensor`)**: `Tensor::storage_` is an owned `std::vector<float>`, copied fresh
  on every op. This is intentionally naive; don't add shared/ref-counted storage without
  updating `docs/DESIGN.md` first.
- **Ownership (`Variable`)**: `Variable` owns its `Tensor data_` directly; backward closures
  capture **copies** of any `Tensor` values they need. The graph itself, however, is raw
  pointers (`TapeEntry.inputs`) — see the lifetime constraint in §3 above before writing code
  that constructs and discards intermediate `Variable`s.
- **Autograd ops are free functions**, not `Variable` methods (see §3). This is a deliberate,
  already-established pattern — don't introduce member-method ops for new ops in Steps 4+.
- **Broadcasting reduction in backward is centralized**: `reduce_sum_to_shape` is called once,
  in `Variable::backward_with_grad`, after a backward closure returns — never inside the
  closure itself.
- **Keep `main.cpp`/`examples/` thin**: demos should showcase the API, not be where new
  features get exercised for the first time. Every new op ships with tests in `tests/`.
- **Every new op needs**: a shape-mismatch test (if applicable), a correctness test against
  a hand-computed example, and — for autograd ops — a gradient check against central finite
  differences (see `docs/DESIGN.md`'s "Gradient checking" section for the calibrated
  tolerance, `atol=1e-2`, not the more obvious-looking `1e-4`).

---

## 5. Known issues

1. `numel()` recomputes on every call instead of being cached at construction — fine for now,
   will matter once tensors get large or `numel()` is called in hot loops.
2. No CI configured — build + `ctest` only run locally, no GitHub Actions workflow yet.
3. `Variable`'s tape (`TapeEntry.inputs`) holds raw, non-owning `Variable*` pointers with no
   lifetime safety — using a `Variable` after any of its graph ancestors have been destroyed is
   undefined behavior, and nothing catches this at compile or run time. See `docs/DESIGN.md`'s
   "Tape / graph structure" section.
4. `Variable`'s data members (`data_`, `grad_`, `requires_grad_`, `has_grad_`, `tape_`) are all
   `public`, despite the trailing-underscore naming that elsewhere in this codebase
   (`Tensor::shape_`, `storage_`) signals "private." Tests reach directly into `tape_`. This is
   inconsistent with `Tensor`'s own convention; not urgent to fix, but don't copy the pattern
   for new types without deciding it's intentional.