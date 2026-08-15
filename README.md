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
- Multi-dimensional indexing (`operator()(i, j, ...)`)
- `transpose()` for permuting dimensions
- Slicing (`slice()`) for contiguous-range sub-tensors

## Requirements

- CMake 3.16+
- A C++23 compiler (GCC, Clang, or MSVC)
- Git (for fetching GoogleTest during the build)

## Build

```bash
cmake -S . -B build
cmake --build build
```

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
│   └── main.cpp
└── tests/
    └── test_tensor.cpp
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
