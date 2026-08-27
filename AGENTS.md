# torc — Agent Guide

`torc` is a minimal, from-scratch C++ tensor library (naive `float32`-only storage) with a
CMake build, a small demo binary, a tape-based autograd system (Milestone 4, complete),
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
│       ├── autograd.hpp        # Variable class + free-function ops (add, mul, ...)
│       ├── nn.hpp              # nn::Module base + nn::Sequential container
│       ├── optim.hpp           # optim::SGD, optim::Adam, optim::AdamW declarations
│       └── nn/
│           ├── linear.hpp      # nn::Linear declaration
│           ├── activations.hpp # nn::ReLU, nn::Sigmoid, nn::Softmax declarations
│           └── losses.hpp      # nn::MSELoss, nn::CrossEntropyLoss declarations
├── src/
│   ├── tensor.cpp              # Tensor implementation
│   ├── simd_ops.hpp            # AVX2-accelerated elementwise/unary ops
│   ├── autograd.cpp            # Variable op implementations (backward closures)
│   ├── nn.cpp                  # Module / Sequential method definitions
│   ├── nn/
│   │   ├── linear.cpp          # nn::Linear forward
│   │   └── activations.cpp     # activation forward functions
│   │   └── losses.cpp          # loss function forward/backward
│   ├── optim.cpp               # SGD/Adam/AdamW step/zero_grad
│   ├── data.cpp                # Dataset / DataLoader implementation
│   └── main.cpp                # demo executable
├── benchmarks/
│   └── bench_tensor.cpp        # Google Benchmark suite (optional)
├── examples/
│   ├── linear_regression/
│   │   ├── linear_regression.cpp   # end-to-end training script (Step 5.11)
│   │   └── plot_results.py         # optional matplotlib visualization
│   └── mnist_mlp/
│       ├── mnist_mlp.cpp           # 3-layer MLP on MNIST (Step 5.12)
│       └── plot_results.py         # optional matplotlib visualization
└── tests/
    ├── test_tensor.cpp         # GoogleTest suite for Tensor
    ├── test_autograd.cpp       # GoogleTest suite for Variable / autograd
    ├── test_nn.cpp             # GoogleTest suite for nn::Module / nn::Sequential / activations
    └── test_data.cpp           # GoogleTest suite for data::Dataset / DataLoader
```

Five default build targets:
- **`torc`** — library built from `src/tensor.cpp`, `src/autograd.cpp`, `src/nn.cpp`,
  `src/nn/linear.cpp`, `src/nn/activations.cpp`, `src/nn/losses.cpp`, `src/optim.cpp`,
  `src/data.cpp`
- **`torc_demo`** — executable (`src/main.cpp`) linked against `torc`.
- **`linear_regression_example`** — executable (`examples/linear_regression/linear_regression.cpp`) linked against `torc`.
- **`mnist_mlp_example`** — executable (`examples/mnist_mlp/mnist_mlp.cpp`) linked against `torc`.
- **`torc_tests`** — GoogleTest executable (`tests/test_tensor.cpp` +
  `tests/test_autograd.cpp` + `tests/test_nn.cpp` + `tests/test_data.cpp`), links `torc`, registered via `enable_testing()` /
  `add_test(NAME TorcTests ...)`.

When `BUILD_BENCHMARKS=ON`, CMake adds the optional `torc_benchmarks` target from
`benchmarks/bench_tensor.cpp`.

C++23. GoogleTest (`v1.14.0`) is pulled in for tests via `FetchContent`; Google Benchmark
(`v1.8.0`) is pulled in for benchmarks via `FetchContent` when `BUILD_BENCHMARKS=ON`. There are
no other dependencies.

---

## 2. Build & test

```bash
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release   # runs TorcTests (Tensor + Variable + nn + data suites)
./build/torc_demo
```

---

## 3. Current functionality

### `torc::Tensor` (see `include/torc/tensor.hpp`, `src/tensor.cpp`)

See `README.md` for the full Tensor feature list. Key highlights:
- Explicit-shape and initializer-list construction
- Elementwise `add`/`sub`/`mul`/`div` with NumPy-style broadcasting and scalar overloads
- Reductions: `sum()`, `mean()`, `max()`, `min()` (whole-tensor and axis-wise)
- `matmul()` for 2D and batched matrix multiplication with batch broadcasting
- `transpose()`, `slice()`, `reshape()`/`view()` (copying), `exp()`, `softmax()` and
  `softmax(axis)`, `log()`, `sqrt()`

`tests/test_tensor.cpp` covers all of the above plus broadcasting, indexing, and error cases.

### `torc::Variable` (see `include/torc/autograd.hpp`, `src/autograd.cpp`) — Milestone 4, complete

- Wraps a `Tensor` with `requires_grad`/`has_grad` flags and a `tape_` of `TapeEntry` recording
  the op that produced it (see `docs/AUTOGRAD.md` for the full tape structure and why backward
  uses an explicit topological sort rather than per-Variable recursion)
- `Variable(Tensor, bool)`, `Variable(float, bool)` constructors
- `data()`, `grad()`, `requires_grad()`, `has_grad()`, `backward()`, `backward(grad_output)`,
  `zero_grad()`
- Ops are **free functions in `namespace torc`**, not `Variable` methods — e.g.
  `torc::add(a, b)`, `torc::mul_scalar(a, b)`, not `a.add(b)`. Follow this pattern for any new
  op; do not add op methods onto `Variable` itself
- **Milestone 5 in progress**: `nn::Module` base class and `nn::Sequential` container (Step 5.2),
  `nn::Linear` (Step 5.3), Module forward-lifetime fix (Step 5.3a), activation functions
  (`nn::ReLU`, `nn::Sigmoid`, `nn::Softmax`, Step 5.4), loss functions (`nn::MSELoss`,
   `nn::CrossEntropyLoss`, Step 5.5), optimizers (`optim::SGD` with momentum, Step 5.6;
   `optim::Adam`, Step 5.7; `optim::AdamW`, Step 5.8), data loaders (`data::Dataset`,
   `data::TensorDataset`, `data::SyntheticRegression`, `data::CSVDataset`, `data::DataLoader`,
   Step 5.9–5.10), and an end-to-end linear regression example (`examples/linear_regression/linear_regression.cpp`,
   Step 5.11) are implemented — see ROADMAP.md for exact status
- **Lifetime constraint (runtime-checked, not ownership-retaining):** `TapeEntry.inputs` holds
  raw, non-owning `Variable*` pointers paired with weak lifetime tokens. Backward checks tokens
  before traversing tracked inputs and throws `TorcError` if an ancestor was destroyed. Every
  tracked `Variable` should still outlive `backward()`; untracked constants may be temporary
  because their tensor values are captured by closures. See §5 and `docs/DESIGN.md`.

`tests/test_autograd.cpp` covers: `Variable` scaffold, scalar and tensor-tensor autograd,
broadcast backward, reductions, view ops, `detach()`/`set_grad_enabled(bool)`, and guarded in-place ops —
each with gradient checks against central finite differences where applicable. See
`README.md` for the full feature list.

### `torc::nn::Module` and `torc::nn::Sequential`

- `nn::Module` is a lightweight base class with `forward()`, `operator()()`, parameter
  registration, and `parameters()` collection. `forward()` must keep intermediates alive in
  `forward_cache_` so backward is safe.
- `nn::Sequential` chains `Module`s in order; `forward(x)` passes input through each module.
- Tests cover parameter registration, forward/backward for `nn::Linear`, activation modules
  (`ReLU`, `Sigmoid`, `Softmax`, including axis-aware behavior) with gradient checks, loss functions (`MSELoss`,
  `CrossEntropyLoss`) with forward correctness and numerical gradient checks, and optimizers
  (`optim::SGD` with momentum, `optim::Adam`, `optim::AdamW`) with step/zero_grad behavior.
- `tests/test_data.cpp` covers `data::TensorDataset` construction and indexing, `data::SyntheticRegression`
  and `data::CSVDataset` loading, `data::MNISTDataset` loading and pixel normalization, and
  `data::DataLoader` batching, shuffling, and epoch reset.
- For full API details and design rationale, see `docs/DESIGN.md`'s Milestone 5 section.

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

1. No CI configured — build + `ctest` only run locally, no GitHub Actions workflow yet.
2. `Variable`'s tape (`TapeEntry.inputs`) retains raw, non-owning `Variable*` pointers. Weak
   lifetime tokens now detect destroyed tracked ancestors and make backward throw `TorcError`,
   but the graph is still not ownership-retaining; callers must keep tracked ancestors alive.
   See `docs/DESIGN.md`'s "Tape / graph structure" section.
3. `Variable`'s data members (`data_`, `grad_`, `requires_grad_`, `has_grad_`, `tape_`) are all
   `public`, despite the trailing-underscore naming that elsewhere in this codebase
   (`Tensor::shape_`, `storage_`) signals "private." Tests reach directly into `tape_`. This is
   inconsistent with `Tensor`'s own convention; not urgent to fix, but don't copy the pattern
   for new types without deciding it's intentional.
