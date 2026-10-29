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

## Lesson status
Published 2026-10-16 as "Modules" (the course map said "The abstraction: Module"; a colon in the heading breaks the writing rules).
Widgets: `module-widgets.js` (model viewer, call order). Tests: `site/test/modules.test.mjs`. Both code blocks were compiled and
run against v0.1.0. Checked: Node tests, `check_site`, `tools/prose/check.sh`, Chromium at both widths, axe-core clean.
Not done: independent review, screen reader.
