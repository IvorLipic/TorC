# torc

A minimal, from-scratch C++ tensor library. Naive `float32`-only storage with elementwise operations, built on CMake with GoogleTest-based unit tests.

## Features

- `torc::Tensor` with explicit-shape construction and initializer-list construction
- Elementwise `add`, `sub`, `mul`, `div` with NumPy-style broadcasting
- Scalar overloads for all four elementwise ops
- Unary negation (`operator-()`)
- `operator==` for value + shape equality
- Pretty-printing (`operator<<`) for shape and data
- Raw pointer data access
- Shape inspection and element count
- Reductions: `sum()`, `mean()`, `max()`, `min()` (whole-tensor and axis-wise)
- `reshape()` / `view()` for changing shape without copying data
- Multi-dimensional indexing (`operator[](i, j, ...)`)
- `transpose()` for permuting dimensions
- Slicing (`slice()`) for contiguous-range sub-tensors
- `matmul()` for 2D and batched matrix multiplication with batch broadcasting
- Optional CBLAS-backed `matmul` for performance (via `TORC_USE_BLAS`)

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

When `TORC_USE_BLAS=ON`, the library requires `cblas_sgemm` (CBLAS C interface). If no CBLAS is found, CMake will fail with a clear error message.

## Run the demo

```bash
./build/torc_demo
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

Tests use [GoogleTest](https://github.com/google/googletest), fetched automatically via CMake `FetchContent`.

## Project layout

```
torc/
├── CMakeLists.txt
├── include/torc/
│   ├── tensor.hpp
│   └── utils.hpp
├── src/
│   ├── tensor.cpp
│   ├── matmul_blas.cpp   # CBLAS matmul (when TORC_USE_BLAS=ON)
│   └── main.cpp
└── tests/
    └── test_tensor.cpp
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
