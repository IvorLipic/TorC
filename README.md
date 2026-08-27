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
- Elementwise natural logarithm (`log()`)
- Elementwise square root (`sqrt()`)
- Whole-tensor softmax (`softmax()`) — used by `nn::Softmax` and `nn::CrossEntropyLoss`
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
  `reshape`/`view`, `slice`), `detach()`/`no_grad()` / `set_grad_enabled(bool)`, unary
  transcendental ops (`exp`, `log`, `sqrt`) with backward and gradient checks, and guarded
  in-place ops (`Tensor::fill` / `Variable::fill` throws `TorcError` on tracked
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
- Optimizers: `optim::SGD` with momentum, `optim::Adam`, `optim::AdamW`
- Data loaders: `data::TensorDataset`, `data::SyntheticRegression`, `data::CSVDataset`, `data::MNISTDataset`, and `data::DataLoader` with batching and shuffling
- End-to-end examples: linear regression (`examples/linear_regression/linear_regression.cpp`) and MNIST MLP (`examples/mnist_mlp/mnist_mlp.cpp`)

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

Tests use [GoogleTest](https://github.com/google/googletest), fetched automatically via CMake
`FetchContent`. The `TorcTests` target covers `tests/test_tensor.cpp` (Tensor),
`tests/test_autograd.cpp` (Variable / autograd), `tests/test_nn.cpp` (`nn::Module` /
`nn::Sequential`), and `tests/test_data.cpp` (`data::TensorDataset` / `data::DataLoader`).

## Run the linear regression example

```bash
./build/linear_regression_example
```

This trains `nn::Linear` on `SyntheticRegression` for 100 epochs and writes two CSV files
to `examples/linear_regression/`: `loss_history.csv` and `predictions.csv`.

## Run the MNIST MLP example

```bash
./build/mnist_mlp_example
```

Trains a 3-layer MLP (`Linear(784, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 10)`) on the bundled MNIST
dataset (`datasets/mnist/mnist_train.csv`) for 10 epochs using `AdamW` and `CrossEntropyLoss`.
Evaluates on both train and test splits each epoch and prints accuracy. Accepts an optional
command-line argument for the maximum number of training samples to use (e.g.
`./build/mnist_mlp_example 100`). The dataset is already provided as CSV files in
`datasets/mnist/`; no conversion step is needed.

## Visualize results

The simplest path is the optional Python helper script using matplotlib:

```bash
python examples/linear_regression/plot_results.py
```

It reads the CSVs and saves `loss_curve.png` and `predictions.png`. If you don't have
matplotlib, open the CSVs in Excel, Google Sheets, or LibreOffice Calc instead.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.