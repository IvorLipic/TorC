# TorC

TorC is a small C++23 library for tensors and automatic differentiation. With it, you can build and
train small neural networks on a CPU. It is for learners and developers who want to read how a deep
learning library works, in code that is short enough to study.

> **Learn with TorC Learn:** [TorC Learn](https://ivorlipic.github.io/TorC/) is an interactive
> course that uses this library. Its source is on the
> [`web`](https://github.com/IvorLipic/TorC/tree/web) branch.

In code and in CMake, the library has the name `torc`. TorC is a reference implementation for
learning, not a replacement for PyTorch. It uses `float32` only, runs on the CPU, and copies tensor
data in most operations.

## Features

| Area | What it contains |
| --- | --- |
| Tensors (`torc::Tensor`) | `float32` tensors. `add`, `sub`, `mul` and `div` with NumPy-style broadcasting. `sum`, `mean`, `max` and `min` over the whole tensor or one axis. `matmul` (2D and batched), `conv2d`, `transpose`, `slice`, `reshape`/`view`, `exp`, `log`, `sqrt` and `softmax`. Scalar overloads, unary `-`, multi-dimensional indexing `t[i, j]`, `operator==` and printing with `operator<<`. |
| Autograd (`torc::Variable`) | Reverse-mode automatic differentiation. Each result keeps a tape, which is a record of the operations that made it. Operations are free functions such as `torc::mul(a, b)`. `detach()`, `Variable::set_grad_enabled()` and `Variable::NoGradGuard` stop the recording. |
| Layers (`torc::nn`) | `Module`, `Sequential`, `Linear`, `Conv2d`, `Flatten`, `ReLU`, `Sigmoid`, `Softmax` |
| Losses (`torc::nn`) | `MSELoss`, `CrossEntropyLoss` |
| Optimizers (`torc::optim`) | `SGD` with momentum, `Adam`, `AdamW` |
| Data (`torc::data`) | `TensorDataset`, `SyntheticRegression`, `CSVDataset`, `MNISTDataset`, and a `DataLoader` that makes batches and can shuffle them |
| Errors | `TorcError` and its subclasses `ShapeError`, `NumericalError` and `IndexError` |

The tests check the gradients against hand-computed values. For most operations, they also compare
the gradients with numerical estimates from finite differences.

### Rules to know before you start

- A tracked `Variable` is a `Variable` that records gradients. Keep every tracked `Variable` alive
  until `backward()` returns. The tape keeps only pointers to them, so `backward()` throws
  `TorcError` if one is gone. Write `Variable prod = torc::mul(a, b); Variable y = torc::sum(prod);`.
  Do not write `torc::sum(torc::mul(a, b))`. A module keeps its own intermediate values alive until
  the next call of the same module. Call `backward()` before you call that module again.
- Call `backward()` on a result with one element, or give it an explicit output gradient.
- `nn::MSELoss` throws `ShapeError` when the input shape and the target shape are different. This
  is a change on `main` after version 0.1.0. Version 0.1.0 broadcast the two shapes.
- `reshape`, `view`, `transpose` and `slice` copy the data. They are not free views.
- `slice` takes one contiguous range per axis. TorC has no gather by a list of indices.

## A short example

This program fits a straight line to four points. It uses a `Linear` layer, the mean squared error
loss and gradient descent.

```cpp
#include <iostream>
#include <torc/nn/linear.hpp>
#include <torc/nn/losses.hpp>
#include <torc/optim.hpp>

int main() {
    using torc::Tensor;
    using torc::Variable;

    // Four points on the line y = 2x + 1, one feature per row.
    Variable x(Tensor({0.0f, 1.0f, 2.0f, 3.0f}, {4, 1}), false);
    Variable y(Tensor({1.0f, 3.0f, 5.0f, 7.0f}, {4, 1}), false);

    torc::nn::Linear model(1, 1, 0.0f, /*seed=*/42);  // pred = x @ W.T + b
    torc::nn::MSELoss loss_fn;
    torc::optim::SGD optimizer(model.parameters(), /*lr=*/0.05f);

    for (int step = 0; step < 500; ++step) {
        Variable pred = model(x);           // shape (4, 1), same as y
        Variable loss = loss_fn(pred, y);   // mean squared error
        optimizer.zero_grad();
        loss.backward();                    // gradients of W and b
        optimizer.step();                   // W -= lr * dW, b -= lr * db
    }

    const auto& params = model.named_parameters();
    std::cout << "weight = " << params.at("weight").data().data()[0]
              << ", bias = " << params.at("bias").data().data()[0] << "\n";
}
```

To build it, use one of the CMake projects in
[Use TorC from another CMake project](#use-torc-from-another-cmake-project). The program prints a
weight close to 2 and a bias close to 1. Our clang-18 Release build printed
`weight = 2, bias = 0.999998`.

For longer programs (linear regression, an MNIST MLP and an MNIST CNN), see
[EXAMPLES.md](EXAMPLES.md).

## Requirements

- CMake 3.20 or later.
- A C++23 compiler: GCC 14 or later, Clang 18 or later, or MSVC 19.36 or later. The code uses
  deducing `this` and `std::format`. GCC 13 cannot compile it.
- Git and network access to build the tests or the benchmarks. CMake downloads GoogleTest and
  Google Benchmark with `FetchContent`.
- Optional: OpenMP. The option `-DUSE_OPENMP=ON` runs some large elementwise and reduction loops
  in parallel.

The CI tests Clang 18 and GCC 14 on Ubuntu 24.04. No CI job tests MSVC or macOS.

## Build and test

```bash
git clone https://github.com/IvorLipic/TorC.git
cd TorC
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Replace `clang++-18` with your C++23 compiler, for example `g++-14`. If your default compiler
supports C++23, you can remove that option.

The build makes the static library, the example programs and one GoogleTest program,
`torc_tests`. CTest runs that program as the test `TorcTests`. It contains the suites in
`tests/test_tensor.cpp`, `tests/test_autograd.cpp`, `tests/test_nn.cpp` and `tests/test_data.cpp`.

With a multi-configuration generator such as Visual Studio, add `--config Release` to the build
command and `-C Release` to the `ctest` command.

### CMake options

| Option | Default | Effect |
| --- | --- | --- |
| `TORC_BUILD_TESTS` | ON in a top-level build, else OFF | Build `torc_tests`. Only this option downloads GoogleTest. |
| `TORC_BUILD_EXAMPLES` | ON in a top-level build, else OFF | Build the example programs. |
| `TORC_INSTALL` | ON in a top-level build, else OFF | Make the `cmake --install` and package-config rules. |
| `TORC_NATIVE_ARCH` | ON in a top-level build, else OFF | Compile Release builds of the library with `-march=native` (`/arch:AVX2` on MSVC). |
| `BUILD_BENCHMARKS` | OFF | Build `torc_benchmarks`. This downloads Google Benchmark. |
| `USE_OPENMP` | OFF | Run some large elementwise and reduction loops in parallel with OpenMP. |

A top-level build is a build of this repository itself. A build through `add_subdirectory` or
`FetchContent` is not a top-level build.

`TORC_NATIVE_ARCH` turns on the AVX2 code in `src/simd_ops.hpp` and in the `matmul` inner loop.
With `-DTORC_NATIVE_ARCH=OFF`, the library uses scalar loops. The binary then runs on any CPU that
the compiler targets by default, but `matmul` and the elementwise operations are slower. Use
`-DTORC_NATIVE_ARCH=OFF` when you install a build that other computers will use.

## Install

```bash
cmake -S . -B build-install -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release \
  -DTORC_BUILD_TESTS=OFF -DTORC_BUILD_EXAMPLES=OFF -DTORC_NATIVE_ARCH=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/torc-install"
cmake --build build-install -j4
cmake --install build-install
```

With a multi-configuration generator such as Visual Studio, add `--config Release` to both the
`cmake --build` and the `cmake --install` command, so that they use the same configuration.

The install step copies the static library, the public headers in `include/torc/`, the files
`torcConfig.cmake` and `torcConfigVersion.cmake`, and `LICENSE` into the prefix.

## Use TorC from another CMake project

The target `torc::torc` gives your program the include directories and the C++23 requirement.
Complete projects for both methods below are in
[`examples/consume/find_package/`](examples/consume/find_package/) and
[`examples/consume/fetchcontent/`](examples/consume/fetchcontent/).

### With an installed copy and `find_package`

Put this `CMakeLists.txt` next to your `main.cpp`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(torc 0.1 CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE torc::torc)
```

Then tell CMake where you installed TorC:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/torc-install"
cmake --build build
./build/my_app
```

With a multi-configuration generator, build with `--config Release` and run `./build/Release/my_app`.

`find_package(torc 0.1)` accepts versions 0.1.x only, because a new 0.x minor version can change
the API.

### With `FetchContent`

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(torc
    GIT_REPOSITORY https://github.com/IvorLipic/TorC.git
    GIT_TAG        v0.1.0)   # a release tag or a commit SHA
FetchContent_MakeAvailable(torc)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE torc::torc)
```

Build it with the same `cmake` commands as above, without `CMAKE_PREFIX_PATH`. In this mode,
TorC does not build its tests or examples, so CMake does not download GoogleTest. `TORC_NATIVE_ARCH`
is also off. Add `set(TORC_NATIVE_ARCH ON)` before `FetchContent_MakeAvailable`, or pass
`-DTORC_NATIVE_ARCH=ON`, to use the AVX2 code.

Configure your project with `-DCMAKE_BUILD_TYPE=Release`. The library uses its optimization flags
(`-O3`, `-funroll-loops`) only in a Release build. The tag `v0.1.0` does not contain the changes
under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).

## Performance

TorC runs on one thread by default and uses simple algorithms. The table compares it with PyTorch
on one CPU thread. The two libraries ran on the same machine with Release builds.

| Operation | Shape | torc (µs) | torch 2.11 (µs) | torc / torch |
|-----------|-------|-----------|-----------------|--------------|
| Elementwise add | [16384] | 6.6 | 9.3 | 0.7× |
| Matmul 2D | [64, 64, 64] | 64.5 | 19.7 | 3.3× |
| Matmul 2D | [128, 128, 128] | 512 | 99.3 | 5.2× |
| Matmul 2D | [256, 256, 256] | 4283 | 869 | 4.9× |
| Batched matmul | [4, 64, 64, 64] | 261 | 74.1 | 3.5× |
| Batched matmul | [8, 128, 128, 128] | 4142 | 1069 | 3.9× |
| Softmax | [16384] | 265 | 39.8 | 6.7× |
| Transpose 2D | [256, 256] | 424 | 4.4 | 96× |

At this size, elementwise add is faster than in PyTorch. `matmul` is 3.3 to 5.2 times slower.
`transpose` shows the largest difference, because `torch.t()` makes a view and TorC copies the data.
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) has the full tables and the method.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on each push to `main` and on each pull
request. Each job configures and builds from an empty build directory. The jobs use a read-only token and no secrets.

- **Build and test:** Clang 18 Release, Clang 18 Release with `-DTORC_NATIVE_ARCH=OFF`, and GCC 14
  Release. Each job then runs `ctest --output-on-failure`.
- **Consumer test:** the job installs TorC to a prefix. Then it builds and runs the
  `find_package` and `FetchContent` projects in `examples/consume/` and checks their output.
- **Sanitizers:** Clang 18 Debug with AddressSanitizer and UndefinedBehaviorSanitizer. A sanitizer
  report stops the test.

## Learn more

- [TorC Learn](https://ivorlipic.github.io/TorC/): the interactive course that uses this
  library.
- [EXAMPLES.md](EXAMPLES.md): how to run the training examples, and how to get the MNIST data.
- [docs/DESIGN.md](docs/DESIGN.md): the design decisions and the reasons for them.
- [docs/AUTOGRAD.md](docs/AUTOGRAD.md): the tape and the backward algorithm. Its section
  "Common Pitfalls" lists the mistakes that are easy to make.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): the benchmark results and how to run them.
- [ROADMAP.md](ROADMAP.md): the milestones, with their finished and open work.
- [CHANGELOG.md](CHANGELOG.md): the changes in each version.

## Contributing

Read [AGENTS.md](AGENTS.md) before you change the code. It gives the project layout, the build
targets, the conventions and the known issues. Each new operation needs tests in `tests/`,
including a gradient check for each differentiable operation.

## License

TorC uses the MIT License. See [LICENSE](LICENSE).
