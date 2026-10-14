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

## Lesson status (as built; module `site/js/lessons/linear-layer.js`, still `draft` in course.json)

**Storyboard as built.** Hero → (1) *Shapes first*: `widgets/shape-explorer.js` (steppers in/out/B, to-scale diagram x · Wᵀ + b = y with the inner
dimension checked, shape table, live `in·out + out`, "Break it" + four recorded runs; pure logic in `linear-shapes.js`) + the real-message
gotcha → (2) *One neuron is one row of W*: `matmul-grid.js` (editable x, W, b of the lesson example; hover/focus/click an output; paper work;
"b = 0", "copy neuron 1 into 2" twins demo) → (3) *Three operations*: the shared graph lab on scenes `linear` and `linear-dx` with own layout/text
(`linear-layer.scenes.js`) and the parity badge → (4) *Backward*: prose with the real dY, gW, gb + `linear-backward.js` (pick gW, gb or dX entries,
mse and sum scenes, verdict against TorC's stored grads) + predict (dX = column sums of W) → (5) *Why initialise like that*: `init-signal.js`
`mountWeightHistogram` (six recorded cases, expected bell curve, table) and `mountInitSignal` (Kaiming / 0.01 / 1.0 / all on a log axis, seed band,
table) with the √(2/in) derivation labelled *Expected* → six misconception cards → recap → three checks (shapes of Linear(784,32), edited b, why {out,in}).

**Observed / expected / precomputed, as labelled on the page.** Live in the browser: shapes, parameter count, predicted error (explorer); all
arithmetic of the grid and of the gradient sums; the graph lab (engine). Recorded from TorC: every `what()` text (shown with a "Recorded" badge
only when the learner's numbers equal a recorded call, otherwise "Predicted with the recorded message template"), the parameter counts, the
histograms and the six-layer table (a browser cannot reproduce C++'s RNG). Amber numbers in prose are arithmetic on recorded numbers (`CALCS`).

**Content decisions.** Indices are 1-based in paper work. The recorded Linear(784, 32) / Linear(32, 32) / Linear(32, 10) counts light a badge in
the explorer. Very small/large std values use their own bound span (`class="rec"`, scientific form) because `T()` would print them in full; integers
below 1e6 (25088, 26506) go through `T()`, which keeps them exact (`fmt` rounds only non-integers to 4 significant digits); `linear-layer.test.mjs`
re-resolves every one. Zero weights are taught
only as far as verified: equal rows give identical output columns (shown live); no claim about gradients of equal neurons beyond "the forward pass
gives them no reason to differ". "Bigger init learns faster? No" is shown through the forward signal only; the backward statement is labelled
*expected, not measured*. Not in the lesson: rank-3 inputs, anything about `MSELoss` with a broadcast target.

**Definition of done (content-plan §5).**
- [x] (1) This note, with V/NV tags.
- [x] (2) Trace program and JSON exist (not touched here). `tools/trace/run.sh --check` was **not** re-run by the lesson author (no change under `tools/` or `data/`).
- [x] (3) Engine == scenes `linear`, `linear-dx` (values, grads); FD present and ok; parity flags true; `Σ(in·out+out)` for every layer and total; init std within 5 % (n ≥ 1000);
  shape-error texts reproduced by the explorer's predictor; dX = W column sums; KaTeX `throwOnError` for all paper work and all prose formulas; prose numbers bound
  (`T`, `rec`, `calc` each re-checked by `site/test/linear-layer.test.mjs`).
- [x] (4) `node --test site/test/` green; browser: laptop 1366×768 and phone 390×844 for every step of every widget (see the report of the session); `/TorC/` prefix server;
  page-overflow check at 320, 360, 390, 768, 1024, 1366, 1920 px.
- [x] (5) Keyboard path through every widget, focus kept on re-render, `aria-live` status, no autoplay, `prefers-reduced-motion` honoured (CSS; page loaded and used under emulation),
  no colour-only encoding (shape, dash, text and underline cues), a table view for both plots. text colours computed against their backgrounds with the WCAG formula (not a browser audit): muted 5.3–6.2, accent colours 5.8–9.5, body text 10.8–14.8, all ≥ 4.5:1; decorative cell borders are not text.
- [ ] (6) Independent read-only review and an outside reader: **not done** (lead).
- [x] (7) No audit talk on the page (a test greps for it).

**Review fixes (round 1).** The Kaiming bullet and the myth card now quote the ten-draw average and layer-6 range (the first run, which ends above that band, is
named as such; the plot draws the average as a white line and marks the first run); the std-versus-second-moment drop 0.998 -> 0.83 is explained with the
quoted root-mean-square and the derived sqrt(1 - 1/pi); the gotcha lists Linear failures only (no MSELoss message on the page; `Linear(0, 2)` is explained);
`predict-dx` precedes the gradient widget and no longer hints at its answer; the zero-weights card and bullets are scoped to a hidden layer; the seed paragraph
names `std::random_device` and the standard library of the recorded run; N = 4 is stated; negative terms are parenthesised after a plus sign; ReLU, variance,
normal distribution and log scale are explained at first use; the {out,in} layout is called a convention (the source gives no reason); the three signal curves
are declared one experiment at three scales (Kaiming = std_1.0 x sqrt(2/256)^layer, checked to 1e-5 in the test); invalid stepper and grid entries show visible,
`aria-live` text. Tests check the new prose (`review fixes` test). Checked: the graph lab scrolls horizontally inside its own box on a 390 px screen (940 px wide
content, page overflow 0).
