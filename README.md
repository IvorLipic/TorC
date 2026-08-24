# torc

A minimal, from-scratch C++ tensor library. Naive `float32`-only storage with elementwise
operations and a tape-based autograd system, built on CMake with GoogleTest-based unit tests.

## Documentation

- **[AGENTS.md](AGENTS.md)** — orientation: project layout, build targets, conventions, known issues
- **[ROADMAP.md](ROADMAP.md)** — milestone checklist, what's done and what's next
- **[docs/DESIGN.md](docs/DESIGN.md)** — architecture decisions and the rationale behind them
- **[docs/AUTOGRAD.md](docs/AUTOGRAD.md)** — the autograd tape structure and backward algorithm

## Features

### Tensor
- `torc::Tensor` with explicit-shape construction and initializer-list construction
- Elementwise `add`, `sub`, `mul`, `div` with NumPy-style broadcasting
- Scalar overloads for all four elementwise ops
- Unary negation (`operator-()`)
- Elementwise exponential (`exp()`)
- `operator==` for value + shape equality
- Pretty-printing (`operator<<`) for shape and data
- Raw pointer data access
- Shape inspection and element count
- Reductions: `sum()`, `mean()`, `max()`, `min()` (whole-tensor and axis-wise)
- `reshape()` / `view()` for changing shape — note this currently copies the underlying
  storage on every call (not a zero-cost stride trick); see `docs/AUTOGRAD.md`'s Common
  Pitfalls for why that matters if you're implementing backward for either
- Multi-dimensional indexing (`operator[](i, j, ...)`)
- `transpose()` for permuting dimensions
- Slicing (`slice()`) for contiguous-range sub-tensors (no arbitrary-index gather yet)
- `matmul()` for 2D and batched matrix multiplication with batch broadcasting
- Optional CBLAS-backed `matmul` for performance (via `TORC_USE_BLAS`)

### Autograd
- `torc::Variable`: tape-based autograd wrapping `Tensor`, with `requires_grad`/`has_grad`
  tracking
- Backward pass via topological sort over the tape (not recursion) — see
  [docs/AUTOGRAD.md](docs/AUTOGRAD.md) for the full algorithm and why
- Ops are free functions (`torc::add(a, b)`, etc.), never `Variable` methods
- Implemented so far: scalar ops (`add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar`,
  `neg_scalar`), tensor-tensor elementwise ops with broadcasting backward (`add`, `sub`,
  `mul`, `div`, `neg`), reduction ops backward (`sum`, `mean`, whole-tensor and axis-wise),
  `max`/`min` backward with argmax tracking (whole-tensor and axis-wise), `matmul` backward
  (2D + batched, including batch-broadcast cases), view-op backward (`transpose`,
  `reshape`/`view`, `slice`), `detach()`/`no_grad()` / `set_grad_enabled(bool)`, and
  guarded in-place ops (`Tensor::fill` / `Variable::fill` throws `TorcError` on tracked
  Variables) — each with gradient checks against central finite differences where applicable
- **Known constraint:** the tape holds raw, non-owning pointers to the `Variable`s in a graph.
  Every `Variable` participating in a graph must outlive `backward()` on any of that graph's
  outputs — not just the final loss `Variable`. Not enforced by the compiler; see
  `docs/DESIGN.md`.

### Milestone 5 - Basic ML (in progress)
- `nn::Module` base class with `forward()` hook and parameter registration
- `nn::Sequential` container for chaining modules
- `nn::Linear` layer with weight/bias parameters and autograd support
- Activation modules: `nn::ReLU`, `nn::Sigmoid`, `nn::Softmax`
- Loss functions: `nn::MSELoss`, `nn::CrossEntropyLoss`
- Optimizer: `optim::SGD` with momentum
- Data loaders are not yet implemented — see
  [ROADMAP.md](ROADMAP.md) for exact status

## Requirements

- CMake 3.16+
- A C++23 compiler (GCC, Clang, or MSVC)
- Git (for fetching GoogleTest during the build)
- **Optional**: CBLAS library (OpenBLAS, Intel MKL, or Accelerate on macOS) for accelerated `matmul`

## Build

### Default (naive matmul)

```bash
cmake -S . -B build
cmake --build build
```

### With CBLAS backend (accelerated matmul)

```bash
# Install a BLAS library first:
#   Ubuntu/Debian: sudo apt install libopenblas-dev
#   Fedora:        sudo dnf install openblas-devel
#   macOS:         brew install openblas  (or use Accelerate.framework)
#   Windows:       Install OpenBLAS or Intel MKL, ensure it's in library path

cmake -S . -B build_blas -DTORC_USE_BLAS=ON
cmake --build build_blas
```

Supported BLAS providers (auto-detected):
- **OpenBLAS** (cross-platform, `find_package(OpenBLAS)`)
- **Intel MKL** (via `find_package(BLAS)` when MKL is in path)
- **Accelerate.framework** (macOS only, automatic fallback)

When `TORC_USE_BLAS=ON`, the library requires `cblas_sgemm` (CBLAS C interface). If no CBLAS
is found, CMake fails configure with a clear error message.

## Run the demo

```bash
./build/torc_demo
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

Tests use [GoogleTest](https://github.com/google/googletest), fetched automatically via CMake
`FetchContent`. The `TorcTests` target covers `tests/test_tensor.cpp` (Tensor),
`tests/test_autograd.cpp` (Variable / autograd), and `tests/test_nn.cpp` (`nn::Module` /
`nn::Sequential`).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.