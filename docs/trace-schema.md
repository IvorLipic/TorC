# Trace JSON schema (`torc-trace/1`)

Every file in `site/data/` is written by a program in `tools/trace/` that runs the real TorC library at the ref pinned in
`tools/trace/TORC_REF` (`tools/trace/run.sh` regenerates all of them; `run.sh --check` fails on drift). `site/data/` is generated-only:
never edit a file there by hand. This page says what is in each file so a lesson author knows what can be quoted without typing a number.

Producers, in `tools/trace/`: `trace_util.hpp` (shared: JSON writer, `Graph`, `Scene`/`run_scene`, tensor serialisation, finite-difference
checker, envelope), `autograd_trace.cpp`, `trace_why.cpp`, `trace_optim.cpp`, `trace_linear.cpp`, `trace_tensors.cpp`, `trace_nonlinear.cpp`, `trace_classification.cpp`, `trace_deep.cpp`, `trace_modules.cpp`, `trace_losses.cpp`, `trace_training.cpp`, `trace_optimizers.cpp`, `trace_generalization.cpp`, `trace_mnist_mlp.cpp`, `trace_conv.cpp`, `trace_mnist_cnn.cpp`, `trace_whole_game.cpp`, `trace_mnist_mistakes.cpp`, `trace_one_step.cpp`, `trace_shift.cpp`,
`trace_mnist_seeds.cpp` (the MNIST programs share `trace_mnist.hpp`: the two example models, evaluation with a confusion matrix, image
shifts, and reading trained weights back from a trace file). Behaviour notes per lesson (what the
numbers mean, derivations, library symbols): `docs/lessons/<lesson>.md`.

## Conventions

- **Envelope** (top of every file, in this order): `schema` = `"torc-trace/1"`; `library` = `{ref, compiler, flags}` (`ref` is the content of
  `TORC_REF`, a release tag such as `"v0.1.0"` or a full commit SHA such as `"38785db59b98a78548b58a16c51a03ec56a65a9d"`; `compiler` e.g. `"clang 18.1.3 (libstdc++ 14)"`; `flags` the build flags, including `-ffp-contract=off`, which the trace build sets for TorC and the producers so that no
  fused multiply-add changes a result on a CPU that has one; the last two are informational and
  are not compared by the drift check); `generator` = source file of the producer; `kind` = topic (`autograd` | `why` | `optim` | `linear` | `tensors` | `nonlinearity` | `classification` | `deep` | `modules` | `losses` | `training` | `optimizers` | `generalization` | `mnist-mlp` | `conv` | `mnist-cnn` | `whole-game` | `mnist-mistakes` | `one-step` | `shift` | `mnist-seeds`);
  `lesson` = first lesson that uses the file. Everything else is topic data. Unknown keys must be ignored by consumers.
- **Numbers.** Values the library computed in `float32` are written with 9 significant digits (they round-trip the float32 exactly, so
  `0.1f` appears as `0.100000001`; present them rounded). Values that are configuration or were derived in `double` (learning rates, `h`,
  eigenvalues, the factor `1 - 8*lr`, ...) are written as the shortest round-trip double (`0.05`). `-0` is written `0`. No `NaN`/`Inf`
  ever appears (the writer refuses). A JS port works in `double`, so it differs from these numbers at the `1e-7` relative level per
  operation; near a converged optimum, where a gradient is a small difference of large terms, measured differences reach `2e-6` absolute
  (relative `2e-4` for a gradient of `0.0086`): use a mixed tolerance such as `|a-b| <= 1e-5 * (1 + |b|)`.
- **Derived values** (plain arithmetic on recorded numbers, or textbook mathematics evaluated in the producer; *not* library output) are
  either under a key named `derived`/`hessian`/`analysis`, or are named in the lesson note. A test that quotes one should say it is derived.
- **Scene schema** (`scenes[]`, identical in every file): `{id, title, expression, note, root, root_has_grad, nodes[], finite_difference}`;
  `nodes[] = {name, op, role ("leaf"|"op"), requires_grad, inputs[], value{shape,values}, grad{shape,values}|null}` in construction order
  (leaves first, then ops). `inputs` are the names read from the real `TapeEntry` before `backward()` (so `mul(e,e)` has `["e","e"]`; a
  constant such as a target is listed too). `grad` is `null` for the root (the library never stores the root's own gradient: the seed `1`
  is internal) and for constants. `finite_difference = {h, atol, leaves[{name, numerical[], analytic[], max_abs_err, ok}]}` is a central
  difference in float32 with `h = 0.01`.
- **Drift check** (`tools/trace/compare.mjs`): identical structure, numbers equal within relative `1e-6` (absolute `1e-7`). Exceptions, only
  in `linear.json`: values that come out of `std::normal_distribution` depend on the C++ standard library (the same seed draws different
  numbers under libc++), so `init_stats[*].weights|histogram|mean|min|max` are compared by structure only, `init_stats[*].std` within 10 %, and
  `signal[*].layers|across_seeds` within 50 % (errors compound over six layers). On one toolchain the output is bit-for-bit repeatable
  (checked: three runs, identical bytes). A consumer must never try to reproduce an RNG: ship-and-read only.

## `autograd_trace.json` (kind `autograd`, lesson `backprop-lab`)

`scenes[]` with ids `reuse` (`y = x*x + x`), `chain` (`f = sum(relu(a*b + c))`), `mlp` (tiny 2-2-1 network, batch of 2, MSE written as
`sub`, `mul`, `mean`). Unchanged by the refactor except for the envelope (`library` changed from the string `"torc"` to the object above).

## `why-gradients.json` (kind `why`, lesson `why-gradients`)

One weight, one datum: `p = w*x`, `e = p - t`, `L = e*e`, `x = 2`, `t = 6`, so `L(w) = 4(w-3)^2`.

- `scenes[0]` id `why`: nodes `w`(leaf, tunable) `x` `t`(constant leaves) `p`(mul) `e`(sub) `L`(mul, inputs `["e","e"]`), at `w = 0`.
  Backward gives `e.grad = -12`, `p.grad = -12`, `w.grad = -24`.
- `setup`: `{x, t, w_min, loss_formula, grad_formula, critical_lr}` (strings and `critical_lr = 0.25` are Expected mathematics, not library output).
- `grid[]`: `{w, p, e, L, grad}` for `w = -1, -0.75, ..., 7` (33 points, step `0.25`), each from a real forward and `backward()`.
- `probes[]`: same record at `w = 0, 0.1, 1, 1.2, 3, 6, 7.2` (the weights the lesson talks about).
- `secants[]`: for `h` in `1, 0.5, 0.1, 0.01, 0.001`: `{h, w0, L0, L_h, dL, slope, expected_slope, backward_slope}`. `slope = (L(h)-L(0))/h` with `L` from the real forward
  pass **in float32** (so `h = 0.001` gives `-23.9944`, not the exact `-23.996`: cancellation); `expected_slope = 4h - 24` is derived; `backward_slope = -24`.
- `nudge`: change of every stage when `w` moves from `0` by `delta = 0.01`: `stages[] = {name, from, before, after, change, local_slope}` for `p`, `e`, `L`
  (local slopes `2, 1, -12` read from the real backward pass as ratios of node gradients), plus `local_slope_product`, `backward_slope`, `predicted_dL`
  (= slope*delta, derived), `actual_dL`, `second_order_term` (derived).
- `trajectories[]`: for `lr` in `0.05, 0.1, 0.125, 0.25, 0.3`: `{lr, momentum:0, factor (= 1-8*lr, derived), stable_predicted, matches_manual, max_abs_diff_vs_manual, states[9]}`.
  `states[k] = {k, w, p, e, L, grad, w_next, tangent_L_next}`: state `k` is the forward/backward at the weight after `k` real `optim::SGD::step()` calls
  (`k = 0..8`); `w_next` is `w - lr*grad` (what the next step gives); `tangent_L_next` = `L + grad*(w_next - w)` is the straight-line promise (derived).
  `matches_manual` is `true` when every real `SGD::step()` produced bit-identically `w - lr*grad`.
- `checks`: `{grid_matches_formula_max_abs_diff, grid_matches_formula, sgd_matches_manual}`.

v3 additions (brief C3 and Appendix B). All are appended after the keys above; all are **derived** (double arithmetic on the recorded
float32 values, so `1.2` is `1.2000000476837158`), and exist so that a lesson never computes a step, a distance or a share:
- `trajectories[].states[]` also have `step` (`w_next - w`), `distance` (`|w - 3|`), `distance_next` (`|w_next - 3|`), `kept`
  (`distance_next / distance`, the share of the distance that the step keeps; `null` when `distance` is 0, as in lr 0.125 after one
  step) and `crossed` (`true` when `w_next` is on the other side of 3). Each trajectory also has `kept` and `kept_percent`: the share
  kept by its first step. Appendix B values: step sizes 1.2 and 0.72 are `trajectories.0.states.0.step` and `.states.1.step`;
  distances 3, 1.8 and 1.08 are `trajectories.0.states.0.distance`, `.states.1.distance`, `.states.2.distance`; the shares 60 %,
  100 % and 140 % are `trajectories.0.kept_percent`, `trajectories.3.kept_percent`, `trajectories.4.kept_percent`.
- `nudge` also has `miss_before` (6) and `miss_after` (5.98), the sizes `|e|` of the miss; `fall` (`-actual_dL`, 0.2396) and `ratio`
  (`actual_dL / delta`, -23.96).
- `steep`: four notebooks cost 12 euros, `x = 4`, `t = 12`, so `L(w) = (4w-12)^2 = 16(w-3)^2` with the same minimum `w = 3` and the
  gradient `32(w-3)` (-96 at `w = 0`, where `L = 144`). The same real ops and `optim::SGD` as above.
  `steep.setup` = `{x, t, w_min, loss_formula, grad_formula, critical_lr: 0.0625}` (Expected mathematics); `steep.grid[]`,
  `steep.probes[]` (same weights), `steep.secants[]` (`expected_slope = 16h - 96`) in the formats above;
  `steep.trajectories[]` in the format of `trajectories[]` including the v3 fields, for lr `0.05, 0.1, 0.125, 0.25, 0.3` (index `i` has
  the learning rate of `trajectories[i]`; only 0.05 converges, the others diverge) and then `0.0125, 0.025, 0.03125, 0.0625, 0.075`
  (index `5 + i` keeps the same share as `trajectories[i]`); `factor = 1 - 32 lr`, plus `same_factor_as_lr` (= `4 lr`, derived).
  `steep.compare[]` = `{w, distance, grad, steep_grad, L, steep_L, grad_ratio}` at `w = 0, 1, 1.2, 6` from the two real backward
  passes (`grad_ratio = steep_grad / grad` = 4, derived). `steep.checks` = `{grid_matches_formula_max_abs_diff, grid_matches_formula,
  sgd_matches_manual}`.

## `optim.json` (kind `optim`, lesson `sgd-step`)

A line `y = 2x + 1` fitted to `x = [1,2,3]`, `t = [3,5,7]` with `nn::Linear(1,1)` (w = b = 0), `nn::MSELoss`, `optim::SGD`.

- `scenes[0]` id `sgd`: leaves `x`(3x1) `W`(1x1) `b`(1) `t`(3x1); ops `Wt`(transpose, inputs `["W"]`) `xW`(matmul) `y`(add) `L`(**mse**, inputs `["y","t"]`).
  Has an extra `parity` object `{against, output_equal, grads_equal, loss_equal, max_abs_diff}`: the real `nn::Linear + MSELoss` gives bit-identical output, loss and gradients.
- `data`: `{x, t, true_w, true_b, n, optimum{w,b,loss}}`.
- `hessian` (**derived**, double): `{derived:true, note, matrix [[a,c],[c,d]], eigenvalues [lambda_max, lambda_min], eigenvectors [[..],[..]], critical_lr = 2/lambda_max}`.
- `runs[]`: `{id, optimizer:"SGD", lr, momentum, steps[10], final, observed_diverged, analysis}`. Runs: `sgd-lr0.05`, `sgd-lr0.1`, `sgd-lr0.15`, `sgd-lr0.2`,
  `sgd-lr0.05-mu0.5`, `sgd-lr0.05-mu0.9`. `steps[i] = {w, b, loss, gw, gb}` is the state **before** update `i+1` (parameters, loss, and the gradients `step()` is about to
  read); `final` is the same record after the 10th update. `observed_diverged = final.loss > steps[0].loss`. `analysis` (derived): `{critical_lr, spectral_radius, stable_predicted}`
  where `critical_lr = 2(1+momentum)/lambda_max` and `spectral_radius` is the largest `|root|` of `r^2 - (1+mu-lr*lambda) r + mu` (`|1-lr*lambda|` for `mu = 0`) over the two Hessian eigenvalues.
- `no_zero_grad`: `accumulate_only[3]` (three forward+backward at the same parameters, no step, no `zero_grad`: `gw = -22.67, -45.33, -68.00`) and `training_loop`
  `{lr:0.05, momentum:0, iterations[10]}` (a loop that calls `step()` but never `zero_grad()`). Both: `{w, b, loss, gw, gb, gw_this_pass, gb_this_pass, iteration}` with `gw/gb` the **accumulated**
  gradient `step()` reads and `*_this_pass` its increment (derived by subtraction).
- `skipped_step[]`: `{name, what, before{w,b}, after{w,b}, changed, threw, exception}` for `step_without_backward`, `step_after_zero_grad`, `momentum_step_without_backward`
  (all `changed:false, threw:false`) and the control `step_after_backward` (`changed:true`).
- `ctor_errors[]`: `{call, exception, what}` for `SGD` with `lr <= 0` or `momentum < 0` (exact `what()` text).
- `surface`: `{w[], b[], order, points[25]}` with `points[i] = {w, b, loss}`, loss from the real forward pass (conformance data for a live contour plot).
- `checks`: `{parity_output_and_grads_equal}`.

v3 additions (brief §6 `sgd-step`, the c-zero check of §7, D13), appended after the keys above:
- `data.points[]` = `{x, t}`: the three data points listed one by one.
- `start` = `{w, b, loss, gw, gb}` at `w = b = 0` from one forward and `backward()` of `nn::Linear` + `nn::MSELoss`: loss 27.67,
  `gw = -22.67` (the one-pass gradient that the c-zero check needs), `gb = -10`.
- `nudge_start` = `{w0, b0, L0, w[], b[], gradient, secant_gradient}`: the gradient found by nudging one parameter at a time. `w[]`
  nudges `w` alone (`b` stays 0) and `b[]` nudges `b` alone, for `h = 1, 0.5, 0.1, 0.01, 0.001`: `{h, L_h, dL, slope, expected_slope,
  backward_slope}`, with `L_h` from the real forward pass in float32 and `slope = dL / h` (float32, so cancellation shows at small
  `h`, as in `why-gradients.json`); `expected_slope` is derived (`-68/3 + 14h/3` for `w`, `-10 + h` for `b`). `gradient = [gw, gb]` from
  `backward()` (the gradient as a list of two slopes); `secant_gradient` = the two slopes at `h = 0.001`.
- `after_ten_steps` = `{lr: 0.05, updates: 10, with_zero_grad{w, b, loss}, without_zero_grad{w, b, loss}, note}`: one quantity for "the
  loss after ten steps" (D13), the state after the 10th `step()`. `with_zero_grad` equals `runs[0].final` (loss 0.001145);
  `without_zero_grad` is the `no_zero_grad.training_loop` after its 10th step (loss 0.8042). Note that `runs[i].steps[9]` is the state
  after **nine** updates (0.001184): do not call it "after ten steps".
- `no_zero_grad.training_loop.final` = `{w, b, loss}` after the 10th step (the same as `after_ten_steps.without_zero_grad`).
- `off_axis` = `{w0: 0, b0: 3, start{w, b, loss, gw, gb}, nudge{L0, w[], b[]}, runs[]}` (appended last; the lesson's start, which
  is off the steep axis of the valley, so the gradient does not point at the lowest point). `nudge.w`/`nudge.b` nudge one parameter
  alone by h = 1, 0.5, 0.1, 0.01, 0.001: `{h, L_h, dL, slope, backward_slope}`. `runs[]` are plain SGD at lr 0.05, 0.15, 0.2
  (ids `off-axis-sgd-lr…`), in the format of `runs[]` above: loss 6.667 and gradient (−10.67, −4) at the start; lr 0.2 diverges.

## `linear.json` (kind `linear`, lesson `linear-layer`)

- `scenes[0]` id `linear`: `x`(2x3) `W`(2x3) `b`(2) `t`(2x2); ops `Wt`(transpose) `xWt`(matmul) `y`(add) `L`(mse, inputs `["y","t"]`) for the plan's example
  `x=[[1,2,3],[-1,0,2]]`, `W=[[0.5,-1,2],[1,0,-0.5]]`, `b=[0.1,-0.2]`, `t=[[4,0],[3,-2]]`. `scenes[1]` id `linear-dx`: same but `x` trainable, `W`, `b` trainable, root `S = sum(y)`.
  Both carry `parity {against, output_equal, grads_equal, loss_equal, max_abs_diff}` versus the real `nn::Linear(3,2)` (all `true`, difference `0`).
- `example` (derived): `errors` (= y - t), `dY` (= errors/2), `gW_equals_dYT_x_max_abs_diff`, `gb_equals_colsum_dY_max_abs_diff`, `dX_equals_colsum_W_max_abs_diff`: the hand formulas against the library's gradients.
- `shape_errors[]`: `{id, call, in, out, x_shape, exception, what}` real wrong-shape calls with the exact `what()`; one entry (`exception:"none"`) shows the successful `(4,3)` call with `y_shape`.
- `init_stats[]`: for each `(in, out, seed, init_std)`: `{weight_shape, numel, kaiming_default, expected_std, mean, std, min, max, std_over_expected, histogram{lo,hi,bins,counts,under,over}, bias_all_zero}`
  and, when `numel <= 1024`, `weights{shape,values}` (the actual weights the library drew). `histogram` covers `[-4, +4]` expected stds in 40 bins. `expected_std` is `sqrt(2/in)` for the default or `init_std`.
  **Stdlib-dependent** (see conventions).
- `signal[]`: for `name` in `kaiming` (library default), `std_0.01`, `std_1.0`: six layers of `Linear(256,256)` + `ReLU` on a `64x256` `N(0,1)` input (input seed 100, weight seeds `200 + layer`).
  `layers[0..6]`: `{layer, pre_std, post_std, post_mean, post_rms, frac_zero}` (layer 0 is the input itself; `post_std` is the std of the ReLU output over all 16384 values, population std).
  `across_seeds[1..6]`: `{layer, post_std_mean, post_std_min, post_std_max}` over 10 other weight seeds, to say how typical the primary run is. **Stdlib-dependent.**
- `param_counts[]`: `Linear(784, 32)` (25120, 2 tensors), `MNIST MLP` (26506, 6 tensors), `MNIST CNN` (9098, 6 tensors), each `{name, source, input_shape, layers[{layer, params[{name,shape,numel}], count, output_shape}], total, tensors, sum_of_layers}`;
  `total`/`tensors` come from the real `Sequential::parameters()`, the per-layer breakdown from each module's `named_parameters()` (sorted by name).
- `checks`: `{all_internal_checks_passed}`.

## `tensors.json` (kind `tensors`, lesson `tensors`)

No scenes. `example` = `{shape, numel, values, index[{i, j, flat, value}]}` for `a = [[1, 2, 3], [4, 5, 6]]` (read through
`Tensor::operator[]`). `reshape` = `{shape, values, changing_the_copy_changes_a, error}`. `broadcast[]` = `{id, b{shape, values},
sum}` where `sum` is `{shape, values, exception:"none"}` or `{exception, what}`. `broadcast_shapes[]` = `{x, y, result | exception,
what}` from `broadcast_shape`. `reductions[]` = `{op: sum|mean|max, axis: 0|1, result}`; `whole` = `{sum, mean, max}`;
`reduce_error` (axis 2) and `reduce_negative_axis` (axis -1). `matmul` = `{transpose, a_at, at_a, a_a, batched, batched_both}`, each a
result or an error: `batched` is `(4, 2, 3)` holding 1..24 times `aᵀ`, `batched_both` is `(2, 1, 2, 3)` holding 1..12 times `(3, 3, 2)`
holding 1..18 (both operands have batch dimensions). `check_sum` = `{x_shape, sum0_shape}` for the check question (`x` of shape
`(4, 5)`). Producer: `tools/trace/trace_tensors.cpp`. Everything is library output except `flat` (`i*3 + j`, confirmed by a self-check
against `a.data()`) and `changing_the_copy_changes_a` (a comparison done by the producer).

## `nonlinearity.json` (kind `nonlinearity`, lesson `nonlinearity`)

`z` = 81 inputs from -4 to 4, built as `(k - 40) / 10` so that `z[40]` is exactly 0 (all grids are built from integers). `activations[]` = `{name: relu|sigmoid, value[], grad[]}` (grad from `backward()` of `sum(f(z))`).
`softmax[]` = `{id, input[3], output[3]}` for rows `small`, `shifted`, `large`, `equal` (axis 1). `data` = `{x[21], t[21], grid[45]}`
with `t = x²`; `setup` = `{points, hidden, steps, lr, optimizer, loss}`. `models[]` = `{name: linear|relu, checkpoints[{step, loss,
prediction[45]}]}`, steps 0, 25, 100, 400; the linear model also has `derived` = `{w_eff, b_eff, max_abs_diff_line_vs_model}`.
Top-level `derived` = `{best_line_slope, best_line_intercept}` (mathematics, not library output). Start weights are set by hand, so
no random generator is involved; the linear model starts as the line `0.4·x - 0.1`, not flat. Producer: `tools/trace/trace_nonlinear.cpp`.

Added for the v3 hinge builder (appended keys, nothing else changed):
- `models[*].checkpoints[*].params` = `{W1, b1, W2, b2}`, each `{shape, values}` as TorC holds them: `W1` (8, 1) is the first
  `Linear`'s `weight` (out, in), `b1` (8), `W2` (1, 8), `b2` (1). Both models, every checkpoint.
- `hinges` = `{model: "relu", step: 400, b2, units[8], sum_plus_b2[45], max_abs_diff_vs_prediction}` for the trained relu model on
  `data.grid`. `units[k]` = `{unit, w1, b1, w2, contribution[45], derived{kink_x, active_for}}`: `contribution` is
  `W2_k · relu(W1_k x + b1_k)` from the real ops (the first layer's forward pass, `relu`, `mul` by the row `W2`); `sum_plus_b2` is
  `sum(contribution, axis 1)` (TorC) plus `b2`, and `max_abs_diff_vs_prediction` compares it with `models[1].checkpoints[3].prediction`
  (0 in the committed run; the program fails above `1e-5`). `derived.kink_x = -b1_k / W1_k` (double) and `active_for` is
  `"x > kink_x"` when `W1_k > 0`, else `"x < kink_x"`.

## `modules.json` (kind `modules`, lesson `modules`)

`models[]` = `{id: mlp|mnist|no-relu|mismatch, input_shape, layers[{label, params[{name, shape, numel}] (sorted by name),
out_shape | exception, what | out:"not reached"}], total, tensors, sequential_parameters_total, sequential_parameters_count,
sequential_out_shape | sequential_exception, sequential_what}`. `orders[]` = `{id, code, exception, what?}` for three orders of
forward and backward calls. No weight value is recorded. Producer: `tools/trace/trace_modules.cpp`.

## `losses.json` (kind `losses`, lesson `losses`)

`mse` = `{prediction[3], target[3], loss, grad[3]}` (`nn::MSELoss`, shape `(3, 1)`). `ce[]` = `{id: uniform|mild|confident, logits[3],
cases[{target, loss, softmax[3], grad[3]}]}` for targets 0, 1, 2 (`nn::CrossEntropyLoss`, one sample). `batch` = `{logits[6], targets[2],
loss, grad[6]}` (the mild row with target 0 and the confident row with target 2). `curves` = `{z[49], ce_loss, ce_grad, sq_loss, sq_grad}`
for logits `[z, 0]` with target 0, z from -6 to 6 in steps of 0.25: cross-entropy, and `nn::MSELoss` of `softmax(z, 1)` against the
one-hot target; the gradients are with respect to `z` from `backward()`. `errors[]` = `{id, targets, exception, what}` for `not-integer`,
`out-of-range`, `rank-1-logits`, `length` and `column-targets` (`exception` `"none"`). `mse_mismatch` = `{prediction_shape [3, 1],
target_shape [3], exception, what}`: `nn::MSELoss` on shapes that differ (`ShapeError` since TorC PR #6; v0.1.0 broadcast them,
threw nothing and recorded `mse_broadcast` = `{…, loss, grad[3]}` instead). Not shown on the lesson page. Producer:
`tools/trace/trace_losses.cpp`.

## `training.json` (kind `training`, lesson `training-loop`)

`setup` = `{samples, true_weight, true_bias, noise_std, data_seed, loader_seed, lr, epochs}`. `data` = `{x[100], y[100]}` from
`SyntheticRegression::get`. `loaders[]` = `{batch_size: 10|32, shuffle, epochs[2][batches[indices]]}`: the indices are what
`next_batch()` returned on a `TensorDataset` whose inputs are `0..99`, with `reset()` between the two epochs (a self-check confirms
the same order on the regression data). `exhausted` = `{exception, what}` from `next_batch()` after the last batch. `runs[]` =
`{batch_size: 100|32|10, epochs[{epoch, steps, loss, weight, bias}]}` for epochs 0 to 30; `loss` is `nn::MSELoss` on all samples
with no gradient, `steps` counts all steps so far. `no_reset[]` = `{epoch, steps, loss}` for three epochs without `reset()`.
`true_line_loss` = `nn::MSELoss` of `3x + 1` against the targets. `derived` = `{noise_variance}`. The values depend on libstdc++'s
random distributions and `std::shuffle`. Producer: `tools/trace/trace_training.cpp`.

v3 addition (brief C1 and C5), appended after the keys above: `batch_compare` = `{batch_sizes [10, 100], seeds [7, 8, 9], seed_decides,
lr, steps: 300, runs[], band[], equal_steps[], equal_samples[], gradient_estimate}`. Every run uses the loop, data, lr 0.1 and start
`w = b = 0` of `runs[]`; the loader seed (the shuffled order) is the only thing that changes. With seed 7, the loss after 3000 samples
equals the 30-epoch loss of `runs[]` bit for bit (a self-check). `runs[]` = `{batch_size, seed, samples_per_step, loss[301]}`, where `loss[k]` is `nn::MSELoss` on all 100 samples after
`k` steps (`loss[0]` is the start). `band[]` = `{batch_size, min[301], max[301]}` over the seeds, per step. `equal_steps[]` = one entry
each for 30 and 300 steps, `{steps, results[{batch_size, seed, steps, samples_seen, loss}], summary[{batch_size, min, max, mean}]}`;
`equal_samples[0]` = the same for 3000 samples seen (`samples`: 300 steps of batch 10, 30 steps of batch 100). The summaries are plain
arithmetic on the recorded losses. Observed: batch 100 gives the same loss for every seed (one batch is all the data), so its min and
max are equal. `gradient_estimate` = `{at{w, b}, batch_size 10, seed 7, full{gw, gb}, batches[10]{gw, gb}}`: at `w = b = 0`, the
gradient from `backward()` on all 100 samples and on each batch of the first shuffled epoch (the batch gradient as an estimate).

## `optimizers.json` (kind `optimizers`, lesson `optimizers`)

`valley` = `{curvature [1, 100], start [-4, 0.4], steps 50, runs[], surface}`: the loss `sum(c·w·w)` with `c = [0.5, 50]` on one
parameter `w` of shape (2). `runs[]` = `{id: sgd|momentum|adam, optimizer, lr, momentum, path[51]{w1, w2, loss}}` (`path[k]` is the
state before step `k + 1`). `surface` = `{w1[61], w2[49], order, loss[49][61]}` from the same forward pass, for the contour lines.
`first_step[]` = `{optimizer, lr, grad[2], change[2]}` for SGD and Adam (lr 0.1) on gradients 0.001 and 1000. `decay_setup` =
`{lr, weight_decay, target, start}`; `decay[]` = `{optimizer: Adam|AdamW, steps[51]{step, w1, w2, u, w_has_grad, w2_grad,
u_has_grad}}` (the values before step `k + 1`; the three gradient fields are read between `backward()` and `step()`, and the last
entry has none) for
`(w1 - 2)² + 0·w2²` and a parameter `u` that the loss does not use. `derived` = `{sgd_lr_limit, decay_factor, w2_after_50_adamw}`.
Producer: `tools/trace/trace_optimizers.cpp`.

v3 additions (brief C4), appended after the keys above:
- `sweep` = `{lrs, start, steps, start_loss, diverged_rule, best_rule, optimizers[]}`: on the same valley, start and 50 steps, every
  optimizer runs with every lr in `lrs` = `0.001, 0.003, 0.01, 0.019, 0.03, 0.1, 0.2, 0.3, 1, 3`. `optimizers[]` = `{id: sgd|momentum|
  adam|adamw, optimizer, momentum (0.9 for momentum), weight_decay (0.01, TorC's default, for adamw), runs[], best{lr, final_loss}}`;
  `runs[]` = `{lr, final_loss, finite, diverged, lowest_loss, steps_completed, exception, what}`. A run diverged when its final loss is
  above `start_loss` (16), or when it did not finish: when `w` overflows, TorC's `sum()` throws `NumericalError` ("sum requires finite
  input values"), the run stops, `final_loss` is `null`, `steps_completed` < 50 and `exception`/`what` hold the error (otherwise
  `"none"` and `""`). `lowest_loss` is the lowest finite loss of the run. `best` is the lr with the lowest final loss among the runs
  that did not diverge (the sweep is coarse: one run per lr).
- `race[]` = `{id, optimizer, lr, momentum, weight_decay, final_loss, path[51]{w1, w2, loss}}`: each optimizer at its own `best.lr`,
  with the path in the format of `valley.runs[].path`.

## `generalization.json` (kind `generalization`, lesson `generalization`)

`setup` = `{degree 15, train_samples 10, test_samples 100, noise 0.3, lr 0.05, steps 20000, optimizer, loss}`. `data` =
`{train_x, train_t, test_x, test_t, grid[81]}`: input made by the producer (`sin(pi x)` plus uniform noise from `std::mt19937`), not
library output. `runs[]` = `{weight_decay: 0|0.1|1, checkpoints[27]{step, train_loss, test_loss, weight_sum_sq, curve[81]}}` for
steps 0 to 20000: `nn::Linear(15, 1)` on the powers of x, `optim::AdamW`, full batch; the losses are `nn::MSELoss` on each whole set
with no gradient, `curve` is the model on `grid`. `derived` = `{noise_variance}` (`a²/3` for noise uniform in `[-a, a)`).
Producer: `tools/trace/trace_generalization.cpp`.

Added for the validation split (C2) and the real settings (C10, D11), appended keys only (the training and test data are
unchanged, because the validation samples are drawn from the same `std::mt19937` after the test samples):
- `data.val_x[30]`, `data.val_t[30]`: the validation set, made like the test set (x uniform in `[-1, 1)`). The model never trains on it.
- `runs[*].checkpoints[*].val_loss`: `nn::MSELoss` on the whole validation set with no gradient.
- `runs[*].val_stop` = `{step, val_loss, test_loss, train_loss}`: the checkpoint with the lowest `val_loss` (the first one on a
  tie), with the test loss at that checkpoint. This is the step the "validation-stop game" reveals.
- `runs[*].best_test` = `{step, test_loss, val_loss}`: the checkpoint with the lowest `test_loss`, for comparison only.
  Committed run: weight decay 0 stops at step 200 (also the best test step), 0.1 at step 1500 (best test 15000), 1 at 20000.
- `settings` = `{train_samples 10, val_samples 30, test_samples 100, noise 0.3, data_seed 3, degree 15, features, model
  "nn::Linear(15, 1)", start_weights "all 0", optimizer "optim::AdamW", lr 0.05, weight_decays [0, 0.1, 1], steps 20000,
  batch_size 64, shuffle false, steps_per_epoch 1, loss "nn::MSELoss"}`: what the code on the page must show. `batch_size` is
  the `DataLoader` batch size; it is larger than the 10 samples, so each step uses all of them.

## `classification.json` (kind `classification`, lesson `classification`)

A disc of class 1 inside a ring of class 0, so no straight line separates them. Producer: `tools/trace/trace_classification.cpp`.
No scenes (the site's engine has no softmax or cross-entropy op).

- `setup` = `{train_samples 200, test_samples 200, data_seed, class_1, class_0, r_inner 0.9, r_ring [1.1, 1.8], hidden [0, 2, 4, 8],
  optimizer "optim::Adam", lr 0.05, steps 1000, batch, loss "nn::CrossEntropyLoss", init, checkpoints [0, 25, 100, 1000]}`.
- `data` = `{train_xy[400], train_label[200], test_xy[400], test_label[200]}`: `xy` is row-major `(n, 2)` (x, y per point), point `i`
  has label `i % 2`. Input made by the producer (`std::mt19937`, `std::cos`/`std::sin`), not library output.
- `grid` = `{axis[41], n 41, lo -2, hi 2, order}`: `axis[k] = (k - 20) / 10`; `prob1[j * n + i]` is the point `(axis[i], axis[j])`.
- `models[]` = `{hidden, architecture (text), parameters, seeds, checkpoints[4]}` for hidden 0 (`Linear(2, 2)`), 2, 4, 8
  (`Linear(2, H) -> ReLU -> Linear(H, 2)` in an `nn::Sequential`). `checkpoints[k]` = `{step, train_loss, test_loss, train_correct,
  train_accuracy, test_correct, test_accuracy, units[H] | derived, layers[], prob1[1681]}`, the state before step `step + 1`:
  - losses: `nn::CrossEntropyLoss` on the whole set with no gradient;
  - `*_correct`: points where the predicted class (1 when score 1 > score 0) equals the label, counted by the producer from the
    library's scores; `*_accuracy` = correct / samples (double);
  - `units[k]` (hidden > 0) = `{unit, w1, w2, b, active_train_points}`: row `k` of the first layer's `weight` and entry `k` of its
    `bias`, so the unit's line is `w1·x + w2·y + b = 0`; `active_train_points` counts training points with `w1·x + w2·y + b > 0`
    (where ReLU passes the value on; 0 would mean a dead unit);
  - `derived.decision_line` (hidden 0) = `{w1, w2, b}`: row 1 minus row 0 of the weight and bias (double), the line where the
    two scores are equal;
  - `layers[]` = `{weight{shape, values}, bias{shape, values}}` for each `Linear`, in order;
  - `prob1`: column 1 of `torc::softmax(scores, 1)` on the real forward pass over the grid points.
  Committed run, last checkpoint, test accuracy: hidden 0 `0.53`, 2 `0.785`, 4 `0.995`, 8 `1`.
- `sigmoid_identity[]` = `{scores [s0, s1], softmax[2], difference, sigmoid, abs_diff}` for six pairs: `softmax` is `torc::softmax`
  along axis 1 of the `(1, 2)` scores, `difference = sub(s1, s0)` and `sigmoid = torc::sigmoid(difference)`; `abs_diff` is
  `|softmax[1] - sigmoid|` (0 for all six; the program fails above `1e-6`).
- Values depend on libstdc++'s `std::normal_distribution` (TorC's Kaiming start weights, seeds `100 + H` and `200 + H`) and on the
  C library's `exp`/`log`. On one toolchain the output is bit-for-bit repeatable.

## `deep.json` (kind `deep`, lesson `deep-nets`)

Producer: `tools/trace/trace_deep.cpp`. No scenes. Per-layer statistics (mean absolute value, population std, rms, share of
exact zeros) are computed by the producer in double from the library's tensors.

- `setup` = `{depth 8, width 32, batch 64, data_seed, inputs, targets, weight_seeds, loss "nn::MSELoss", kaiming_std}`: a fixed batch
  `x` (64, 32) uniform in `[-√3, √3)` (std 1) and targets (64, 1) uniform in `[-1, 1)` (producer input); `kaiming_std` is
  `torc::kaiming_normal_std(32)` = `0.25`. The stack is 8 × (`Linear(32, 32)` → activation) and a head `Linear(32, 1)` in an
  `nn::Sequential`; layer `l` uses seed `300 + l`, the head `309`.
- `vanishing[]` = `{activation: sigmoid|relu, loss, input_std, layers[9], derived{first_over_last_hidden}}`, both with the same
  start weights (TorC's default Kaiming init). `layers[l]` = `{layer 1..9, name ("hidden 1".."hidden 8", "head"), mean_abs_grad,
  signal_std?}`: the mean of `|weight.grad|` after one forward pass and one `backward()` of the loss, and (hidden layers) the std of
  the signal after the activation. `first_over_last_hidden` = layer 1's `mean_abs_grad` over layer 8's (derived). Committed run:
  sigmoid `6.1e-5`, relu `0.49`.
- `init[]` = `{name: kaiming|too_small|too_large, init_std (0 = TorC's default), weight_std, layer1_weight_std_observed, loss,
  layers[9]}` for the ReLU stack with `nn::Linear(32, 32, init_std, seed)` and `init_std` 0, 0.05, 0.5. `layers[l]` = `{layer 0..8,
  std, rms, frac_zero, mean_abs_grad?}`: layer 0 is the input, layer `l` the signal after the `l`-th ReLU; `mean_abs_grad` is layer
  `l`'s weight gradient after one `backward()`. Committed run, std after layer 8: `1.2`, `3.1e-6`, `313`.
- `twins_setup` = `{model "Linear(2, 2) -> ReLU -> Linear(2, 1)", x (4, 2) XOR inputs, t (4, 1), optimizer "optim::SGD", lr 0.2,
  momentum 0, loss, recorded_steps 10, long_steps [0, 10, 100, 500, 2000], note}`. Start weights set by hand (no generator).
- `twins[]` = `{id: twins|different, start{W1, b1, W2, b2}, steps[11], equal_every_step, long_run[5]}`. `steps[k]` = `{step, loss,
  neuron0, neuron1, grad0, grad1, output_bias, neurons_equal}`: the state before update `k + 1`; `neuronN` = `{w[2] (row N of W1),
  b (bias N), v (output weight N)}`, `gradN` the same fields of the gradients `step()` reads (after `backward()`);
  `neurons_equal` is a bit-for-bit comparison of the two neurons' parameters and gradients. `equal_every_step` covers all 2000
  steps (`true` for `twins`, `false` for `different`; the program fails otherwise). `long_run[k]` = `{step, loss, prediction[4],
  neurons_equal}`. Committed run: the twins stay at loss `0.1667` (predictions 1/3, 1/3, 1, 1/3), the different start reaches `4e-14`.
- Values of `vanishing` and `init` depend on libstdc++'s `std::normal_distribution`; on one toolchain they are bit-for-bit repeatable.

## `mnist-mlp.json` (kind `mnist-mlp`, lesson `mnist-mlp`)

Input: MNIST CSV files from `tools/trace/fetch_mnist.sh` (`MNIST_DIR`). `setup` = `{train_samples, test_samples, batch_size, epochs,
lr, weight_decay, optimizer, loss, image_side, pixel_max, parameters}`. `epochs[]` = `{epoch, steps, train_loss, test_correct,
per_class_correct[10]}` for epochs 0 to 20 (epoch 0 is before training and has no `steps` or `train_loss`; `train_loss` is the mean
batch loss of the epoch). `test_per_class` = `{correct[10], total[10]}` after training. `layers[3]` = `{weight_shape, weight, bias}`
for the three `Linear` layers (all trained values, row-major `(out, in)`). `shown[20]` = `{index, label, pixels[784] (0..255),
logits[10]}`: test images 0 to 15 and the first four after them that the model reads wrongly. The values depend on libstdc++'s
`normal_distribution` (start weights) and `std::shuffle` (batch order). Producer: `tools/trace/trace_mnist_mlp.cpp`.
These two MNIST files come from thousands of training steps, and their values also depend on the C library's `exp` and `log`. A
new compiler or standard library on the CI runner can move them past the drift check's tolerance. Then regenerate them with
`run.sh`, check that the lesson tests still pass (they test the claims, not the exact values), and commit the new files.

Added for v3 (additive: same model, seeds and settings, so `layers`, `shown` and every older value are unchanged; in each object the
new keys come after the old ones). Accuracies are fractions `correct / total`, derived in double (D9).
- `setup.pixels` (784, values per image as the dataset gives them) and `setup.classes` (10, outputs of the last `Linear`): anchors.
- `epochs[]` also have `test_accuracy`, `test_loss` (mean `nn::CrossEntropyLoss` on the 10000 test images) and `train_subset` =
  `{correct, total, accuracy, loss}`: the same model, measured the same way, on a fixed subset of its own training images (the first
  `train_subset_samples` = 10000). This is train vs. test for the main model; the test curve is still a test curve (C2).
- `test_per_class.accuracy[10]`.
- `settings` (C9) = `{model, parameters, optimizer, lr, weight_decay, batch_size, epochs, steps (9380), steps_per_epoch, loss, shuffle,
  loader_seed, init, init_seeds, train_samples, example{source, epochs, lr, batch_size}}`. `example` holds the values in
  `examples/mnist_mlp/mnist_mlp.cpp` at the pinned ref (copied from that source; the trace does not run the example).
- `final` = `{test_correct, test_samples, test_accuracy, test_mistakes, test_loss, train_subset}` for the final model.
- `confusion` = `{rows: "true digit", columns: "predicted digit", matrix[10][10], total, correct, accuracy}` on all 10000 test images
  (`matrix[t][p]` = images of digit `t` classified as `p`). The producer checks that the matrix sums to `total` and that its diagonal
  equals `correct` (= `final.test_correct`).
- `mistakes` = `{rule, total_wrong, count, pixels_in: "mnist-mistakes.json", items[24]{index, label, predicted, probabilities[10]}}`:
  the first 24 wrong test images in test-set order, with TorC's softmax probabilities. Their pixels are in `mnist-mistakes.json`.
- `validation_run` (C2) = `{note, train_samples 55000, validation_samples 5000, validation_first_index 55000, train_subset_samples 5000,
  steps_per_epoch 430, epochs[21]{epoch, steps, train_loss, train_subset{correct, total, accuracy, loss}, validation{...}}, final_test{...}}`
  (epoch 0 has no `steps` or `train_loss`): a second run of the same recipe (seeds 1, 2, 3 and 11, AdamW lr 5e-4, 20 epochs) on the
  first 55000 training images only. The last 5000 are never trained on and are the validation set. The test set is measured once,
  after training. Use this run to show overfitting as a train-validation gap that grows. Holding images out of the main run itself
  would change its batch order and so its weights, which the digit pad and the tests use.

## `mnist-mistakes.json` (kind `mnist-mistakes`, lesson `mnist-mlp`)

The grid of mistakes, in its own file so that `mnist-mlp.json` stays small. `source` = `"mnist-mlp.json mistakes.items"`, `rule`,
`total_wrong`; `items[24]` = `{index, label, predicted, probabilities[10], pixels[784] (0..255)}`, the images of
`mnist-mlp.json mistakes.items` in the same order; `checks` = `{weights_reproduce_mnist_mlp_logits, predictions_match_mnist_mlp}`.
Producer: `tools/trace/trace_mnist_mistakes.cpp` (a weight-reading producer, see below).

## `one-step.json` (kind `one-step`, lesson `one-step`)

One real training step of the final MLP (weights read from `mnist-mlp.json`) on one training image, with `optim::SGD`, no momentum,
`lr` = 0.001 and a batch of one image, so that "change = -lr × gradient" is the update rule. The real run used AdamW and batches of 128.
- `setup` = `{optimizer, lr, momentum, batch_size 1, loss, weights_from, model, parameters, rule}`.
- `full_run` = `{steps 9380, epochs, batch_size, train_samples, steps_per_epoch, optimizer, lr, weight_decay, source}`, read from
  `mnist-mlp.json` (`epochs[-1].steps` and `setup`): the number of `step()` calls of the real run.
- `image` = `{rule, set "train", train_index, label, pixels[784]}`: the first training image that the model classifies correctly with a
  probability of the true digit in [0.5, 0.9). A training image, because a training step on a test image would contradict C2.
- `forward` = `{hidden1{pre[32], post[32]}, hidden2{pre[32], post[32]}, scores[10], probabilities[10], predicted, p_true, rival, p_rival,
  loss}`: `pre` is the `Linear` output, `post` the ReLU output; `rival` is the most probable wrong digit.
- `backward` = `{scores_grad[10], hidden1_pre_grad[32], hidden2_pre_grad[32]}`, read from the Variables after `loss.backward()`.
- `weights[4]` = `{id, what, layer, param, row, col?, before, grad, lr_times_grad, after, change, path, update_exact}` for `first_layer`
  (the layer-0 weight from the brightest pixel to the hidden unit where its gradient is largest), `output_true` and `output_rival` (the
  layer-2 weights from the most active second-layer unit to the true and the rival score) and `output_bias_true`. `lr_times_grad` is
  `grad * lr` in float32, as `Tensor::mul` computes it. `update_exact` is true when `after == before - lr_times_grad` bit for bit,
  which is what `SGD::step()` does. `change = after - before` (float32) equals `-lr_times_grad` only up to the float spacing near `before`
  (relative 1e-5 here), so quote `lr_times_grad` as the size of the step. `path`: for `first_layer` `{pixel, pixel_row, pixel_col,
  pixel_value (0..1), pixel_255, unit, unit_pre, unit_post, unit_grad, product, grad_equals_product}` (the gradient is the pixel value
  times the gradient at the unit, exactly); for the output weights `{unit, unit_post, class, score_grad, product, grad_equals_product}`;
  for the bias `{class, score_grad, p_true_minus_one}`.
- `after` = `{probabilities[10], predicted, p_true, p_rival, loss}`: the same image after the step. `changed_parameters`: how many of the
  26506 numbers `step()` changed. `derived` = `{p_true_change, loss_change, loss_check (= -ln p_true), note}`. `checks` =
  `{weights_reproduce_mnist_mlp_logits, modules_one_at_a_time_equal_sequential, every_update_exact, p_true_rose}`.
Producer: `tools/trace/trace_one_step.cpp`.

## `shift.json` (kind `shift`, lesson `convolution`; C8)

Both final models (weights read from `mnist-mlp.json` and `mnist-cnn.json`) on all 10000 test images moved by whole pixels with zero
fill. The producer moves the pixels; everything after that is TorC.
- `setup` = `{test_samples, shifts [0..4], directions [left, right, up, down], fill "zero", note}`.
- `accuracy` = `{mlp, cnn}`, each `{parameters, source, left|right|up|down: [{shift, correct, accuracy}] × 5}` (the index is the shift
  in pixels; shift 0 is one evaluation, repeated in the four lists).
- `feature_maps` = `{index 0, label, pixels[784], stride 2, shift_px 2, direction "right", cell_shift 1, shape (1, 8, 14, 14), original[],
  shifted[], check{compared_cells, max_abs_diff, equal, ink_pixels_pushed_out}, not_a_multiple{shift_px 1, max_abs_diff_unmoved,
  max_abs_diff_moved_one_cell}}`: the CNN's first convolution after relu. A move by the stride moves the map by exactly one cell
  (`max_abs_diff` 0 over 1456 cells). A 1 px move changes the map instead (both differences are not 0). The CNN ends in Flatten and
  Linear, so its accuracy still falls with the shift: it is not shift-invariant.
- `single_image` = `{index, label, right[5]{shift, mlp_predicted, mlp_p_true, cnn_predicted, cnn_p_true}}`.
- `checks`: the loaded weights reproduce the recorded logits, shift 0 equals the recorded final test counts, and `original` equals
  `mnist-cnn.json first_maps`. The digit pad re-centres a drawing, so a learner cannot reproduce these shifts by drawing.
Producer: `tools/trace/trace_shift.cpp`.

## `mnist-seeds.json` (kind `mnist-seeds`, lesson `mnist-cnn`; C5)

Each model at the exact settings of its own trace, with 3 seed sets. `setup` = `{seed_sets[3]{id, layer_seeds[3], loader_seed,
source}, test_samples, note}`: set 1 is the run of `trace_mnist_mlp.cpp` / `trace_mnist_cnn.cpp` (its counts are read from those files),
sets 2 and 3 (layer seeds 101-103 / 201-203, loader seeds 111 / 211) are trained here, in parallel threads. `mlp`, `cnn` = `{model,
settings{epochs, lr, batch_size, optimizer}, runs[3]{seed_set, test_correct_by_epoch[epochs + 1], test_correct, test_accuracy},
min_accuracy, max_accuracy, mean_accuracy}`. `derived` = `{cnn_min_minus_mlp_max, cnn_always_ahead}`. The models keep their own
settings (20 vs. 5 epochs, lr 5e-4 vs. 1e-3), so the comparison is still not perfectly fair (C9). The values depend on the standard
library like the training traces. Producer: `tools/trace/trace_mnist_seeds.cpp`.

**Weight-reading producers.** `trace_mnist_mistakes`, `trace_one_step`, `trace_shift` and `trace_mnist_seeds` read `mnist-mlp.json`
and `mnist-cnn.json` from `site/data` (or from `$TRACE_DATA_DIR`). `run.sh` lists them after the two training traces, so a full run uses
the weights it has just written, and `run.sh --check` uses the committed weights. Each one checks that the loaded weights give the
recorded logits bit for bit (the writer's 9 significant digits round-trip float32 exactly). When `mnist-mlp.json` or `mnist-cnn.json`
is regenerated, regenerate these four files with it.

## `conv.json` (kind `conv`, lesson `convolution`)

`example` = `{input (1, 1, 5, 5), kernel (1, 1, 3, 3), runs[{stride, padding, output}]}` (tensors as `{shape, values}`) for
stride/padding 1/0, 1/1 and 2/1. `digit` = `{label, shape (1, 1, 28, 28), pixels[784] (0..1), maps[{id: vertical|horizontal,
kernel[9], shape, output[784], relu[784]}]}` (stride 1, padding 1; MNIST from `MNIST_DIR`). `cnn[]` = `{label, in_shape, out_shape,
parameters, params[{name, shape}], kernel?, stride?, padding?}` for the six modules of the example's CNN on (2, 1, 28, 28);
`cnn_parameters` is their sum. `dense_compare` = `{label, parameters}` for `Linear(784, 8)`. `errors[]` = `{id: channels|too-small,
input_shape, weight_shape, exception, what}`. Producer: `tools/trace/trace_conv.cpp`.

## `mnist-cnn.json` (kind `mnist-cnn`, lesson `mnist-cnn`)

The same layout as `mnist-mlp.json` (`setup`, `epochs[]` for epochs 0 to 5, `test_per_class`, `layers[3]`, `shown[20]`), for the
CNN of `examples/mnist_cnn`: `layers` are Conv2d (8, 1, 3, 3), Conv2d (16, 8, 3, 3) and Linear (10, 784), weights in TorC's
row-major order. Extra: `first_maps` = `{index, shape (1, 8, 14, 14), values}`, the first convolution after relu for the first
shown image. Producer: `tools/trace/trace_mnist_cnn.cpp`.
Added for v3 (additive, same model): `setup.pixels`, `setup.classes`; `epochs[].test_accuracy`, `epochs[].test_loss`;
`test_per_class.accuracy[10]`; `settings` (C9, as in `mnist-mlp.json`: `settings.example.epochs` is 10 while the trace trains 5, and the
learning rate is 1e-3 against the MLP's 5e-4); `final` = `{test_correct, test_samples, test_accuracy, test_mistakes, test_loss}`;
`confusion` (as in `mnist-mlp.json`). Shift robustness is in `shift.json`, the spread over seeds in `mnist-seeds.json`.

## `whole-game.json` (kind `whole-game`, lesson `what-is-learning`)

Honestly synthetic data: hours of practice in a week (`x`, 0.5 to 4) and the score in a test out of 10 (`y`) for 15 students, made in
the producer (not by TorC) from the true line `y = 1.5x + 2` plus noise with standard deviation 0.6, drawn with `std::mt19937` (seed
2026, fully specified by the C++ standard) and a Box-Muller transform in double; `x` and `y` are rounded to one decimal. The first 10
points train the model, the last 5 are hidden (never trained on).
- `data` = `{story, x_name, y_name, true_line{w, b}, noise_std, seed, generator, n_train, n_hidden, train{x[10], y[10]}, hidden{x[5],
  y[5]}, points[15]{i, x, y, set: train|hidden}}`.
- `runs[]` = `{id, start{w, b}, lr: 0.1, optimizer, batch, settled, settle_rule, steps[], final{steps, w, b, train_loss, hidden_loss}}`:
  `nn::Linear(1, 1)` with `w` and `b` set by hand, `nn::MSELoss` on `(10, 1)` vs `(10, 1)`, `optim::SGD` on all 10 training points per
  step. `runs[0]` (`from-zero`) starts at `w = b = 0`, `runs[1]` (`from-w3`) at `w = 3, b = 0`. `steps[k]` = `{step, w, b, train_loss,
  hidden_loss}` is the state after `k` steps (`train_loss` is the loss `backward()` was called on, `hidden_loss` the loss on the hidden
  points with no gradient). A run stops after the first step that changes neither `w` nor `b` by more than `1e-5` (`settled`), or at
  1000 steps; the last entry is the settled state. Observed: `from-zero` settles after 200 steps at `w = 1.595`, `b = 1.897`.
- `probes[]` = `{id, w, b, train_loss, hidden_loss, train_pred[10], train_residual[10], hidden_pred[5], hidden_residual[5]}`: TorC's
  `MSELoss` of fixed lines (`zero`, `flat-5`, `true`, `w1-b3`, `w2-b1`, `w3-b0`, `w-0.5-b6`), with the predictions of `nn::Linear` and
  the residuals `prediction - target` (float32): conformance data for the JS that scores a hand-placed line.
- `true_line` = `{w, b, train_loss, hidden_loss}`: `MSELoss` of the true line (the noise that no line removes).
- `derived` = `{least_squares{w, b, train_loss}, hessian_lambda_max, critical_lr}`: the best line on the training points and the
  stable-lr limit `2 / lambda_max`, in double (Expected mathematics; the producer checks `lr < critical_lr`).
Producer: `tools/trace/trace_whole_game.cpp`.
