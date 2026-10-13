# Changelog

All notable changes to `torc` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow semantic versioning
(0.x releases may still change the API).

## [Unreleased]

### Added
- GitHub Actions CI (`.github/workflows/ci.yml`): clang-18/gcc-14 build + `ctest` matrix, install and
  consumer smoke test, and an ASan + UBSan job.

### Fixed
- `nn::MSELoss` backward: when `input` and `target` have different shapes that broadcast (for
  example `(N, 1)` against `(N)`), the gradient was scaled by `2 / input.numel()` although the
  forward loss is the mean over the broadcast result, so it was too large by the ratio of the
  broadcast and input element counts (3x for a `(3, 1)` input against a `(3)` target). Backward
  now scales by `2 / N_broadcast` and the standard broadcast reduction sums to each operand's
  shape. Equal-shape results are unchanged, and the forward semantics (mean over the broadcast
  result) are now documented in `nn/losses.hpp`.

## [0.1.0] - 2026-10-13

First packaged version of the library (tag `v0.1.0`).

### Library
- `torc::Tensor`: `float32` tensors with NumPy-style broadcasting elementwise ops, reductions
  (whole-tensor and axis-wise, negative axes supported), `matmul` (2D and batched), `transpose`,
  `slice`, `reshape`/`view`, `exp`/`log`/`sqrt`, and axis-aware `softmax`.
- `torc::Variable`: tape-based reverse-mode autograd (topological-sort backward, broadcasting
  reduction, `detach`, thread-local grad mode with `NoGradGuard`, guarded in-place ops) with
  finite-difference-checked gradients.
- `torc::nn`: `Module`, `Sequential`, `Linear`, `Conv2d`, `Flatten`, `ReLU`, `Sigmoid`, `Softmax`,
  `MSELoss`, `CrossEntropyLoss`.
- `torc::optim`: `SGD` (with momentum), `Adam`, `AdamW`.
- `torc::data`: `Dataset`, `TensorDataset`, `SyntheticRegression`, `CSVDataset`, `MNISTDataset`,
  `DataLoader` (batching, shuffling).
- Fixed before this release: slice backward scattering with the wrong strides, and negative `axis`
  in the autograd `sum`/`mean`/`max`/`min` wrappers and in `Tensor::mean`.

### Packaging
- `cmake --install` installs the `torc` static library, the public headers under
  `include/torc/` (including `nn/`), and `torcConfig.cmake` / `torcConfigVersion.cmake`
  (`SameMinorVersion`: `find_package(torc 0.1)` accepts 0.1.x only, since 0.x minors may break); consume with `find_package(torc CONFIG)` and link `torc::torc`.
- `torc::torc` is also available when torc is added with `add_subdirectory`/FetchContent.
- New CMake options: `TORC_BUILD_TESTS`, `TORC_BUILD_EXAMPLES`, `TORC_INSTALL` (all default ON only
  for a top-level build) and `TORC_NATIVE_ARCH` (default ON for a top-level build, OFF when
  consumed via `add_subdirectory`/FetchContent; OFF drops `-march=native` and the AVX2 paths that
  depend on it, giving portable, reproducible consumer builds).
- GoogleTest is fetched only when `TORC_BUILD_TESTS=ON`.
- Public usage requirement `cxx_std_23`; minimum CMake raised from 3.16 to 3.20 (the first
  version that knows `cxx_std_23`; 3.16 could not select C++23 at all).
- Consumer examples under `examples/consume/`.

### Known limitations
- Requires a C++23 compiler (Clang 18+, GCC 14+, MSVC 19.36+).
- Tracked `Variable`s must outlive `backward()`; the tape holds non-owning pointers.
- No CI yet; developer builds still fetch GoogleTest over the network.
