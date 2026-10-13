# TorC autograd — implementation contract

Status: audit by a read-only agent (code reading + compiled repros), with the lead's re-verification of the
numbers via `visualizations/examples/autograd_trace.cpp` (finite-difference checked). Anything not listed as
**Observed** is a hypothesis. Line numbers refer to the tree at the time of audit.

Build note: `include/torc/tensor.hpp:46` uses C++23 deducing-this. **g++ 13 cannot compile it**; clang++ 18
(used here) or GCC ≥ 14 are needed. Existing tests under clang 18: 3 optimizer tests
(`SGD/Adam/AdamW.SkipsParamsWithoutGrad`, `tests/test_nn.cpp`) fail because they index
`Module::parameters()`, which iterates an `unordered_map` (unspecified order) — unrelated to autograd.

## 1. Observed behavior

| Topic | Fact | Reference |
|---|---|---|
| Representation | `Variable` owns `Tensor data_`, `grad_`, flags and a `tape_` (`vector<TapeEntry>`); ops are free functions | `include/torc/autograd.hpp`, `src/autograd.cpp` |
| Graph edges | Each op appends one `TapeEntry` (raw `Variable*` inputs + weak lifetime tokens + backward closure) to its **output**. No global tape | `autograd.hpp` `TapeEntry`, e.g. `mul` in `autograd.cpp` |
| Recording | Only if an operand `requires_grad` **and** grad mode is on | op bodies, `Variable::grad_enabled()` |
| Forward | Eager: value computed immediately via `Tensor` method | `autograd.cpp` |
| Backward | `backward()` needs `requires_grad`, `numel()==1`; seeds 1.0; recursive DFS post-order topo sort over tracked inputs, reversed; each node's closure runs once all consumers have contributed | `Variable::backward`, `backward_with_grad`, `build_topo` |
| Accumulation | `grad_map[input] += g` and `input->accumulate_grad(g)`; a reused variable gets the **sum** of contributions | `backward_with_grad` |
| Broadcast | `reduce_sum_to_shape` applied centrally after the closure (matmul does it inside its closure) | `autograd.hpp` |
| Intermediate grads | Populated (`m.grad()` etc.) for all tracked non-root nodes | repro |
| Root grad | **The root never gets `.grad()`**; the seed 1.0 is internal | repro: `y.has_grad()==false` |
| Reset | `zero_grad()` clears `grad_`, `has_grad=false`. Grads accumulate across *different* roots until reset | `autograd.cpp` |
| Tape consumption | After `backward()` only the **root's** tape is cleared; a second `backward()` on the same root is a silent no-op | `autograd.hpp` end of `backward_with_grad` |
| ReLU at 0 | gradient 0 | repro |
| max/min ties | gradient to first argmax/argmin only | repro |
| Lifetime | Tracked ancestors must outlive `backward()`; destroyed/moved-from ones make backward throw `TorcError` | `build_topo`, `GraphLifetime.*` tests |
| `nn::Module` | Calling a module again clears its `forward_cache_`, invalidating the previous forward's graph | `include/torc/nn.hpp` |
| Grad mode | `set_grad_enabled` / `NoGradGuard` flag is `thread_local` (docs say "process-global") | `autograd.hpp` |

## 2. Mathematical expectations (independently derived)
`add`: g,g · `sub`: g,−g · `mul`: g·b, g·a · `div`: g/b, −g·a/b² · `relu`: g·[x>0] · `sum`: broadcast g ·
`mean`: g/N · `matmul`: dA = g·Bᵀ, dB = Aᵀ·g · `sigmoid`: g·s(1−s). All matched code and central finite
differences (h = 1e-2, tolerance 1e-2 per `docs/DESIGN.md`) for: exp, log, sqrt, relu, sigmoid, softmax (all
variants), neg, mul, div, sub, sum/mean (axis ≥ 0), max/min, transpose, reshape, matmul, broadcast add/mul,
MSELoss, CrossEntropyLoss.

## 3. Suspected / confirmed defects (not fixed; core untouched)
| ID | Severity | Summary | Evidence |
|---|---|---|---|
| D1 | Medium, reproduced | `slice` backward computes strides from `grad_output.shape()` instead of the input shape → wrong scatter whenever a trailing sliced dim is narrower than the input and the leading extent is > 1 (a plain 2-D crop, no offset needed; confirmed by an independent reviewer). `a=[[1,2,3],[4,5,6]]`, `slice(a,{{0,2},{1,3}})`, `sum().backward()` gives `[0 1 1 1 1 0]`, expected `[0 1 1 0 1 1]`; `slice(a,{{0,2},{0,2}})` gives `[1 1 1 1 0 0]`, expected `[1 1 0 1 1 0]` | `autograd.cpp` ~743–750 |
| D2 | Low–medium, reproduced | Negative `axis` in autograd `sum/mean/max/min(x, axis)` is not normalised (forward is). `sum(x,-1)` → all-zero grad; `max(x,-1)` → out-of-bounds read | `autograd.cpp` ~373, 414, 495, 532 |
| D3 | Low, design quirk | `backward()` on an intermediate permanently cuts the graph above it for later losses (no `retain_graph`) | repro |
| D4 | Low | Root `.grad()` never set | repro |
| D5 | Doc/minor | `-0` shows in grads; `parameters()` order unspecified; "process-global" vs `thread_local`; recursive `build_topo` | various |

The visualizations avoid D1/D2 and say nothing that depends on them. They are reported to the user for decision.

## 4. Confirmed limitations
No differentiable `pow`, `abs`, `tanh`, `clamp`, concat, indexing. float32 only. Graph is non-owning (lifetime rules above).

## 5. Unanswered / not verified
Conv2d and data-loader paths beyond existing tests; batched-matmul broadcast re-run; thread-safety;
deep-graph recursion limits; whether GCC 14 reproduces the 3 optimizer test failures.
