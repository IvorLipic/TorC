# Lesson `modules` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`nn::Module` (`include/torc/nn.hpp`, `src/nn.cpp`): `register_parameter` stores into an `std::unordered_map`, so `parameters()`
returns pointers in the map's iteration order; `operator()` clears `forward_cache_` and calls `forward`. `nn::Sequential::add`
(throws on null), `forward` (chains modules, caching each output), `parameters()` (its own, then each child's). `nn::ReLU`
(no parameters). `nn::Linear` registers `weight` (out, in) and `bias` (out).

## Observed (tools/trace/trace_modules.cpp → site/data/modules.json)
- Parameter counts: `Linear(784, 32), ReLU, Linear(32, 10)` = 25450 in 4 tensors; the MNIST MLP = 26506 in 6; without ReLU 25450.
  `Sequential::parameters()` gives the same totals.
- `Linear(784, 32), ReLU, Linear(64, 10)`: the constructors accept it; the forward pass throws at the third module,
  `ShapeError: matmul inner dimensions must match, got (8, 32) and (64, 10)`.
- Calling the model a second time before `L1.backward()` makes it throw `TorcError: autograd graph input expired before
  backward(); keep all graph Variables alive`. Backward before the next forward, or backward of the latest loss, works.
- On this toolchain `named_parameters()` iterates `bias` before `weight` (the lesson says only "no fixed order").

## Expected (mathematics)
`in·out + out` parameters per `Linear`.

## Suspected discrepancy
None. (The parameter order is by design of the container, not a bug.)

## Observed, added for the v3 appendix rewrite (`parameter_order`, `custom_module`)
- `Sequential::parameters()` lists the modules in `add()` order. Inside one `Linear`, the order comes from the `std::unordered_map`
  (`register_parameter` uses `emplace`): on this toolchain the recorded `Linear(3, 2)` lists `bias` before `weight`, and the 784-32-10
  model lists shapes `(32)`, `(32, 784)`, `(10)`, `(10, 32)`. The C++ standard does not fix the order, so the page says "no fixed
  order" and shows the recorded run as one observation (D7).
- `Sequential::named_parameters()` is its own (empty) map: count 0, while the `Linear` it holds has 2. `Sequential` has no accessor
  for its modules, so the page keeps a pointer to the layer before `add()`.
- The forward-cache rule (`include/torc/nn.hpp`, `src/nn/linear.cpp`): `Module::operator()` clears `forward_cache_` and calls
  `forward`. `Linear::forward` keeps the transposed weight, the product and the sum in the cache. `ReLU::forward` keeps nothing.
  The returned `Variable` is a copy. A second call of the same module destroys the in-between Variables of the first call, so
  `L1.backward()` throws "autograd graph input expired before backward(); keep all graph Variables alive". The loss of the latest
  call still works.
- A module of our own with two operations: with a local in-between Variable, `backward()` throws the same message; with
  `forward_cache_.emplace_back(...)` it works (both Variables get a gradient).

## Lesson status
Rewritten 2026-11 as an appendix page (`template: 3`, part `using-torc`; brief §6 appendix and §7, D7, D17): Module, Sequential
(model viewer with three guided tasks), named parameters and the order of `parameters()`, a predict quiz for each rule, the
forward-cache rule with the call-order widget, a module of our own, and two checks. Widgets: `module-widgets.js` (unchanged).
Tests: `site/test/modules.test.mjs`. Not done: human think-aloud, screen reader.
