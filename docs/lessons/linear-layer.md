# Lesson note: `linear-layer` (internal, not shipped)

Library = TorC `v0.1.0` (`4c1eabe`) in `build/trace/_deps/torc-src/`; paths relative to it. Data: `site/data/linear.json`
(producer `tools/trace/trace_linear.cpp`; schema in `docs/trace-schema.md`). **V** = verified by running the real library (or the independent
NumPy replica, said where used), **NV** = not verified. No lesson prose here.

## 1. Library symbols

| Symbol | Where | Behaviour |
|---|---|---|
| `nn::Linear(in, out, init_std = 0, seed = random_device)` | `include/torc/nn/linear.hpp`, ctor `src/nn/linear.cpp:9-33` | throws `TorcError("Linear features must be positive")` if `in <= 0` or `out <= 0` (V). Weight `{out, in}`, bias `{out}` all zeros, both registered as `"weight"` / `"bias"` with `requires_grad=true`. `init_std <= 0`: `fill_kaiming_normal(..., fan_in = in, seed)` = `mt19937(seed)`, `normal_distribution<float>(0, sqrt(2/in))` (`include/torc/utils.hpp:65-75`); else `normal_distribution<float>(0, init_std)` (V by statistics and by construction) |
| `Linear::forward` | `src/nn/linear.cpp:35-50` | `add(matmul(x, transpose(weight)), bias)`; the three intermediates are parked in `Module::forward_cache_` (a `std::deque`, `include/torc/nn.hpp`) so the returned Variable's graph stays alive until the next call of the module (`Module::operator()` clears the cache first). The intermediates are private: the site's hand-built graph is therefore checked for exact equality (below) |
| `torc::transpose`, `matmul`, `add` | `src/autograd.cpp:689`, `:583`, `:159` | `transpose(a)` copies (no strides in TorC) with backward = transpose of the grad; `matmul` backward `dA = G B^T`, `dB = A^T G` (each reduced by `reduce_sum_to_shape`, `autograd.cpp:115`); `add` broadcasts `b{out}` over the batch, its gradient is summed back over the batch axis |
| shape errors | `Tensor::matmul` `src/tensor.cpp:206-221` | texts in section 2 |
| `nn::MSELoss` | `src/nn/losses.cpp:12-45` | fused `mean((y-t)^2)`; see `sgd-step.md` for its broadcast caveat |
| `Sequential::parameters()` | `src/nn.cpp:45-53` | own parameters (none) + each child's, in layer order; within a layer the order is an `unordered_map` order (unspecified) |
| `nn::Conv2d` | `src/nn/conv.cpp:8-45` | weight `{out, in, k, k}`, bias `{out}`; Kaiming with `fan_in = in*k*k` (used here only for the CNN parameter count) |
| Tests that already pin this | `tests/test_nn.cpp:162` `Linear.ConstructionCreatesCorrectShapes`, `:179` `ForwardMatchesHandComputed`, `:234` `GradCheckForwardAndBackward`, `:680` `SeedProducesReproducibleWeights` | |

## 2. The numbers (`linear.json`)

**Example (V; library equals the hand-built graph bit for bit, and equals NumPy float64 within `1e-5`).** `x = [[1,2,3],[-1,0,2]]` (2x3), `W = [[0.5,-1,2],[1,0,-0.5]]` (2x3), `b = [0.1,-0.2]`, `t = [[4,0],[3,-2]]`.
`Wt = [[0.5,1],[-1,0],[2,-0.5]]`; `xWt = [[4.5,-0.5],[3.5,-2]]`; `y = [[4.6,-0.7],[3.6,-2.2]]` (float32: `4.5999999, -0.699999988, 3.5999999, -2.20000005`); `errors = y - t = [[0.6,-0.7],[0.6,-0.2]]`; `L = 0.3125` (float32 `0.31249994`).
Backward: `y.grad = dY = errors/2 = [[0.3,-0.35],[0.3,-0.1]]` (because `N = 4` elements: `2 e / 4`); `W.grad = dY^T x = [[0, 0.6, 1.5],[-0.25, -0.7, -1.25]]`; `b.grad = [0.6, -0.45]` (column sums of `dY`). Plan 3.3: **same** in every number.
Loss `sum(y)` with `x` trainable (scene `linear-dx`): `S = 5.3`; every `y.grad = 1`; `x.grad = [[1.5,-1,1.5],[1.5,-1,1.5]]` = column sums of `W` in every row (V); `W.grad = [[0,2,5],[0,2,5]]` (= column sums of `x`: `1-1 = 0`, `2+0 = 2`, `3+2 = 5`, each output neuron); `b.grad = [2,2]` (the batch size). Finite-difference checks: all `ok` (max error `4.8e-5`).
**Parity (V).** `nn::Linear(3,2)` with these weights: output, loss, `W.grad`, `b.grad` (and `x.grad` for the second scene) are bit-identical to the hand-built graph (`parity.*_equal = true`, `max_abs_diff = 0`).

**Shape errors (V, exact `what()`).**
| call | exception | text |
|---|---|---|
| `Linear(3,2)` applied to `x (2,2)` | `ShapeError` | `matmul inner dimensions must match, got (2, 2) and (3, 2)` |
| `Linear(784,32)` applied to `x (8,10)` | `ShapeError` | `matmul inner dimensions must match, got (8, 10) and (784, 32)` |
| `Linear(3,2)` applied to `x (4,4)` | `ShapeError` | `matmul inner dimensions must match, got (4, 4) and (3, 2)` |
| `Linear(3,2)` applied to `x (3)` (no batch axis) | `ShapeError` | `matmul requires rank >= 2 operands, got (3) and (3, 2)` |
| `Linear(0,2)` | `TorcError` | `Linear features must be positive` |
| `MSELoss(y (2,2), t (3,2))` | `ShapeError` | `Cannot broadcast shapes (2, 2) and (3, 2)` |
| `Linear(3,2)` applied to `x (4,3)` | none | `y` has shape `(4, 2)` |

The second operand in the message is the **transposed** weight `(in, out)`, not the `{out,in}` weight the learner declared: say this on the page (V; plan: same). Rank-3 input `x (2,2,3)` into `Linear(3,2)`: observed no error and output shape `(2,2,2)` (forward only, V by a scratch run); gradient correctness for rank 3 is **NV**, so do not claim anything about it (plan: same).

**Initialisation statistics (seed fixed per row; stdlib-dependent, see conventions in `docs/trace-schema.md`).**

| layer | seed | numel | expected std | observed std | observed / expected | mean |
|---|---|---|---|---|---|---|
| Linear(3,2) | 7 | 6 | 0.8165 | 0.6811 | 0.834 | 0.286 |
| Linear(32,32) | 1 | 1024 | 0.25 | 0.25271 | 1.011 | 0.00072 |
| Linear(784,32) | 1 | 25088 | 0.050508 | 0.050662 | 1.003 | 0.00038 |
| Linear(256,256) | 1 | 65536 | 0.088388 | 0.088790 | 1.005 | 0.00012 |
| Linear(784,32, init_std 0.01) | 1 | 25088 | 0.01 | 0.010031 | 1.003 | 0.00008 |
| Linear(784,32, init_std 1.0) | 1 | 25088 | 1 | 1.00306 | 1.003 | 0.0076 |

Plan 3.3 quoted "0.0509 for 784x32 vs 0.0505; 0.2613 vs 0.25 for 32x32" for unspecified seeds; with seed 1 the library gives **0.05066** and **0.25271**. Those plan numbers are not reproduced and must not be used; the JSON numbers are the truth for the seeds shipped. Expected sampling error of a std estimated from `n` Gaussian draws is about `1/sqrt(2n)`: 2.2 % for n = 1024, 0.45 % for n = 25088; observed errors (1.1 %, 0.3 %) are within that. A std test with 5 % tolerance for `n >= 1000` passes (checked inside the trace program). All biases are exactly zero (`bias_all_zero:true`, V). `Linear(3,2)` is shipped with its six actual weights (`weights.values`) and `Linear(32,32)` with all 1024; the 25088 and 65536 are summarised by `histogram` (40 bins over +-4 expected std, counts sum to `numel - under - over`; `under`/`over` are 2/2 for 784x32).

**Signal through six layers (V; `Linear(256,256)` + `ReLU` x 6 on a 64x256 N(0,1) input, std of the ReLU output over all values, population std).**

| init | layer 1..6, primary run (seeds 200+layer) | mean over 10 other seeds, layer 1..6 |
|---|---|---|
| Kaiming (library default, std 0.0884) | 0.830, 0.872, 0.921, 0.970, 0.967, 1.036 | 0.823, 0.819, 0.810, 0.799, 0.779, 0.758 (min 0.692, max 0.890 overall) |
| `init_std` 0.01 | 0.0939, 0.0112, 0.00133, 1.59e-4, 1.79e-5, 2.17e-6 | 0.0932, 0.0105, 1.17e-3, 1.31e-4, 1.44e-5, 1.59e-6 |
| `init_std` 1.0 | 9.39, 112, 1.33e3, 1.59e4, 1.79e5, 2.17e6 | 9.32, 105, 1.17e3, 1.31e4, 1.44e5, 1.59e6 |

The plan's numbers (Kaiming 0.83, 0.83, 0.84, 0.91, 0.85, 0.82; 0.01: 0.094 ... 1.7e-6; 1.0: 9.4 ... 1.7e6) came from an unspecified seed. The first-layer values and the orders of magnitude agree (V); the later layers differ in the second digit, as expected from random weights. **The JSON (primary run plus `across_seeds`) wins**; when quoting "stays near 0.8" use `across_seeds` (all Kaiming values over 10 seeds lie in 0.69-0.89). About 46-51 % of ReLU outputs are exactly zero in every layer for every init (`frac_zero`, V).

**Parameter counts (V, built for real).** `Linear(784,32)`: weight `[32,784]` = 25088 + bias `[32]` = 32 -> **25120**, 2 tensors. MNIST MLP (`examples/mnist_mlp/mnist_mlp.cpp`: `Linear(784,32), ReLU, Linear(32,32), ReLU, Linear(32,10)`): 25120 + 1056 + 330 = **26506**, 6 tensors; output shapes for a batch of 1: `(1,32), (1,32), (1,32), (1,32), (1,10)`. MNIST CNN (`examples/mnist_cnn/mnist_cnn.cpp`: `Conv2d(1,8,k3,s2,p1), ReLU, Conv2d(8,16,k3,s2,p1), ReLU, Flatten, Linear(16*7*7,10)`): 80 + 1168 + 7850 = **9098**, 6 tensors; shapes `(1,8,14,14), (1,16,7,7), (1,784), (1,10)` (28 -> 14 -> 7 = `(H+2P-K)/S+1`). `Sequential::parameters()` totals equal the per-layer sums. Plan: 25 120 / 26 506 / 9 098: **same**.

## 3. Observed / Expected / Suspected

**Observed (V).** Weight is `{out, in}` and row `k` is output neuron `k`; `forward = x W^T + b` with the bias row added to every sample; a wrong input width fails inside `matmul` with the message above; bias starts at zero, weights Gaussian with the std shown; the gradient of the layer parameters equals the hand formulas; `Linear` has no activation inside; `parameters()` of a Sequential lists layer by layer.

**Expected (math).** `Y = X W^T + 1 b^T` with `X{B,in}`, `W{out,in}`, `Y{B,out}`. For loss `L`: `dW = dY^T X` (`{out,B}{B,in}`), `db = 1^T dY` (column sums, because the bias row was broadcast over the batch), `dX = dY W`. For `L = mean((Y-T)^2)`: `dY = 2 (Y-T) / (B*out)`. Parameters: `in*out + out`. Kaiming: if a layer's inputs `a` have `E[a^2] = q` and the weights are independent with variance `s^2`, the pre-activation variance is `in * s^2 * q`; with `s^2 = 2/in` it is `2q`; ReLU of a symmetric variable keeps half the second moment, so `E[relu(z)^2] = q` again: the second moment is preserved layer to layer. The reported **std** of the ReLU output is `sqrt(E[a^2] - E[a]^2)` with `E[a] = sigma_z / sqrt(2 pi)`: for `q = 1`, `sigma_z = sqrt 2`, std `= sqrt(1 - 1/pi) = 0.8256` (shipped layer 1: 0.830, 0.57 % off; **V in NumPy**). For a fixed `s`: layer-1 `sigma_z = s sqrt(256)` and std `= 0.5842 sigma_z`: 0.0934 for `s = 0.01` (shipped 0.0939), 9.34 for `s = 1` (9.39). Each further layer multiplies the second moment by `256 s^2 / 2 = 128 s^2`, i.e. the std by `11.31 s`: 0.1131 per layer for 0.01 (0.0939 -> 0.0112 -> 0.00133 matches) and 11.31 for 1.0 (9.39 -> 112 -> 1330 matches); Kaiming gives exactly 1 (so the drift to 0.75-1.0 is sample noise of 256 random weights per output, not a trend of the method). Zero weights: all rows identical -> identical outputs -> identical gradients -> neurons never differentiate (only the **bias** is zero by default).

**Suspected discrepancy (not for the lesson page).**
1. `MSELoss` gradient with a broadcast target is `N` times too large relative to the loss (`sgd-step.md` section 3, minimal repro). Not triggered by any example on the lesson (all same shape).
2. `std::normal_distribution` results are implementation-defined, so `Linear(in, out, 0, seed)` gives different weights under libc++ than under libstdc++: "seeded" means reproducible per standard library only (NV across libc++; V repeat runs identical on libstdc++ 14). The trace ships weights for this reason. Default `seed = std::random_device{}()` makes every unseeded `Linear` different on every run (documented design).
3. `Module::parameters()` order unspecified (audit D5). `Sequential::parameters()` therefore cannot be matched to layers by position within a layer; the trace sorts `named_parameters()` by name.
4. NV: rank-3 input backward (see above).

## 4. Misconceptions to address
Weights are `{in, out}` (they are `{out, in}`; the error message shows the transposed shape); `Linear` includes an activation (it does not; `ReLU` is a separate module); zero weights are fine (identical neurons; the **bias** is the one that starts at zero); bigger init learns faster (`init_std` 1.0 makes the signal grow 11x per layer: 9.4 -> 2.2e6 over six layers, V); bias is per input element (it is per output neuron: shape `{out}`); the input needs a gradient (usually `requires_grad=false`; `dX` is only needed to pass gradient back to an earlier layer); `gb` is a plain copy of `dY` (it is the column sum over the batch).

## 5. Test hooks suggested by the data
Engine equals scenes `linear` and `linear-dx` (values and grads, `1e-5 (1+|b|)`); parity flags true; `param_counts[*].total == sum(in*out+out)` (and for conv `out*in*k*k+out`) using `layers[*].params`; `init_stats[*].std` within 5 % of `expected_std` when `numel >= 1000`; `shape_errors[*].what` matches `/inner dimensions must match/` for the width cases and `/rank >= 2/` for the rank-1 case; `x.grad` rows equal column sums of `W`; `example.gW_equals_dYT_x_max_abs_diff < 1e-5`; FD present and `ok`; `signal` consumers read the table, never recompute it; quoted numbers bound to paths (4.6, -0.7, 0.3125 in `scenes[0]`, 25120 and 26506 in `param_counts`, shipped stds in `init_stats`).

## Lesson status (concise rewrite, decisions 7-9)

Files: `site/js/lessons/linear-layer.js`, `site/js/widgets/matmul-grid.js` (editable x, W, b; pick an output to light up its rows;
"Copy neuron 1 into neuron 2" copies weights and bias; Reset), `init-signal.js` (six Linear(256, 256) + ReLU layers, three recorded
initialisations and all three on a log scale), helpers `linear-ui.js`, `linear-math.js`, `linear-shapes.js`, `linear-signal-data.js`,
`site/test/linear-layer.test.mjs`.

**As built**: hero → TL;DR → 1 shapes (one equation with the shapes, 25088 + 32 = 25120, the real `ShapeError` for x (8, 10)) →
2 one neuron is one row of W → grid → 3 backward (three formulas; bias sums over the batch) → predict (dX rows = column sums of W) →
4 initialisation (√(2/in), 1.036 / 2.17e-6 / 2.17e6 after six layers) → signal widget → 2 checks (shapes; raising b[2]) → course map.
Measured: 359 words, 9 numbers in the text (+2 in scientific form), 2 widgets, 3 questions.

**Removed in the rewrite**: the shape explorer, the graph lab of the three operations and the parity box, the backward widget, the
weight histogram, the long initialisation derivation and seed discussion, the six mix-up cards, the recap, the gotcha box (one sentence
now), the colour legend for printed/derived numbers and the long footer.

**Definition of done**: tests and site check green; Chromium at 1366×768 and 390×844 (grid copy/Reset/edit b[2], signal views; no
console errors, no horizontal overflow). Not done: independent review, screen-reader pass.
