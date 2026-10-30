# Lesson note: `sgd-step` (internal, not shipped)

Library = TorC `38785db` (`main` = v0.1.0 + PR #5 + PR #6) in `build/trace/_deps/torc-src/`; paths relative to it. Data: `site/data/optim.json`
(producer `tools/trace/trace_optim.cpp`; schema in `docs/trace-schema.md`). **V** = verified by running the real library (or the independent
NumPy float64 replica, said where used), **NV** = not verified. No lesson prose here.

## 1. Library symbols

| Symbol | Where | Behaviour (V unless noted) |
|---|---|---|
| `optim::SGD(const std::vector<Variable*>&, float lr, float momentum = 0)` | `include/torc/optim.hpp:8-23`, `src/optim.cpp:7-15` | stores raw `Variable*` and one zero-initialised velocity `Tensor` per parameter; throws `TorcError("SGD: lr must be positive")` if `lr <= 0` and `TorcError("SGD: momentum must be non-negative")` if `momentum < 0` (exact text in `ctor_errors`). The header takes `const std::vector<Variable*>&` (the design doc in `main` says non-const: doc drift, harmless) |
| `SGD::step()` | `src/optim.cpp:17-30` | for each param **with `has_grad()`** (others silently skipped): no momentum `p = p - lr*grad`; momentum `v = mu*v + grad; p = p - lr*v` (velocity starts at 0, so the first momentum step equals a plain step: V, all `lr0.05` runs have `5.4885` after step 1). Does **not** touch `.grad` |
| `SGD::zero_grad()` | `src/optim.cpp:32-36` -> `Variable::zero_grad` `src/autograd.cpp:110` | `has_grad_ = false`, `grad_` emptied |
| `Variable::backward()`, `accumulate_grad` | `src/autograd.cpp:90`, `include/torc/autograd.hpp:75` | grads **add** into existing `.grad` (first call copies, later calls `+=`). `backward()` consumes the tape (`tape_.clear()` at its end), so accumulating needs a **fresh forward pass** before each `backward()`; a second `backward()` on the same root does nothing (V, see `grad-accumulation` in the content plan) |
| `nn::MSELoss::forward` | `src/nn/losses.cpp:12-42` | `mean((input-target)^2)` as one fused op: one `TapeEntry` with inputs `[input, target]`; throws `ShapeError` unless input and target have the same shape (`losses.cpp:13-15`); backward `2(input-target)/N * grad_output` for the input and its negative for the target. `N` is the element count of `input - target`, equal to `input.numel()` (`losses.cpp:33`). Used as the loss node `L` (op name in the scene: `"mse"`) |
| `nn::Linear(1,1)`, `Module::named_parameters()` | `src/nn/linear.cpp:9-50`, `src/nn.cpp:10-16` | parameters `"weight"` ({1,1}) and `"bias"` ({1}); set through `named_parameters().at("weight").data()`. `Module::parameters()` iterates an `unordered_map`, so its **order is unspecified** (SGD does not care: per-parameter independent) |
| `torc::transpose(a)`, `matmul`, `add` | `src/autograd.cpp:689`, `:583`, `:159` | transpose backward = transpose of the incoming grad; matmul `dA = G B^T`, `dB = A^T G`; add broadcasts the bias `{1}` over 3 rows and `reduce_sum_to_shape` sums its gradient back |
| Tests that already pin this | `tests/test_nn.cpp:567` `SGD.StepWithoutMomentumUpdatesParams`, `:596` `StepWithMomentumAccumulatesVelocity`, `:617` `ZeroGradClearsAllParamGrads`, `:637` `RejectsNegativeLearningRate` | |

## 2. The setup and the numbers (`optim.json`)

Data `x = [1,2,3]^T`, `t = [3,5,7]^T` (exactly `y = 2x + 1`); model `y = w x + b`, start `w = b = 0`; `L = (1/3) sum (w x_i + b - t_i)^2`. Float32 throughout.

**Scene `sgd` at (0,0) (V).** `Wt = [[0]]`, `xW = [0,0,0]`, `y = [0,0,0]`, `L = 27.666666`. Gradients: `y.grad = xW.grad = [-2, -3.3333335, -4.666667]`, `W.grad = Wt.grad = -22.666668`, `b.grad = -10`. Finite-difference check ok (`W`: numerical `-22.66665`, error `2.3e-5`; `b`: `-10.0003`, error `3.2e-4`). **Parity (V):** the real `nn::Linear(1,1)` + `nn::MSELoss` gives bit-identical output, loss, `W.grad`, `b.grad` (`max_abs_diff = 0`).

**The first update (V).** `w = 0 - 0.05*(-22.666668) = 1.1333334`, `b = 0 - 0.05*(-10) = 0.5`, new `L = 5.4885173` (an 80.2 % drop). Plan section 3.2: same.

**Runs, loss before each of the 10 updates, then after the 10th (V):**

| run | loss_0 .. loss_9 | after 10 | (w, b) after 10 | diverged? |
|---|---|---|---|---|
| lr 0.05 | 27.67, 5.489, 1.090, 0.2172, 0.04413, 0.009781, 0.002943, 0.001562, 0.001265, 0.001184 | 0.001145 | (2.0386, 0.9105) | no |
| lr 0.1 | 27.67, 0.3319, 0.005268, 0.001304, 0.001198, 0.001141, 0.001087, 0.001035, 0.0009858, 0.0009389 | 0.0008943 | (2.0347, 0.9210) | no |
| lr 0.15 | 27.67, 12.20, 5.377, 2.371, 1.046, 0.4615, 0.2039, 0.09036, 0.04026, 0.01814 | 0.008363 | (1.9967, 0.9153) | no |
| lr 0.2 | 27.67, 41.08, 61.01, 90.59, 134.5, 199.8, 296.6, 440.5, 654.1, 971.3 | 1442 | (-12.73, -5.555) | **yes** |
| lr 0.05, mu 0.5 | 27.67, 5.489, 0.1740, 2.447, 1.616, 0.1771, 0.05824, 0.1908, 0.08646, 0.004406 | 0.009179 | (2.0707, 0.9349) | no |
| lr 0.05, mu 0.9 | 27.67, 5.489, 2.505, 17.95, 18.29, 3.763, 1.536, 11.65, 12.08, 2.577 | 0.9393 | (2.4011, 1.1100) | no |

Parameter path for lr 0.05: `(0,0) -> (1.1333,0.5) -> (1.6378,0.7233) -> (1.8621,0.8234) -> (1.9618,0.8687)`, losses `1.0897, 0.2172, 0.0441` at the last three. Plan (section 3.2): **same**. lr 0.1: step 1 gives `w = 2.2667`, `b = 1`, loss `0.3319` (overshoots the true `w = 2`, V). lr 0.2: `27.67, 41.08, 61.01, 90.59` (V, as in the plan). mu 0.5: plan's `27.67, 5.49, 0.174, 2.447, 1.616, 0.177, 0.058` (V). mu 0.9: plan says "18.0 at step 3"; observed `17.95` at index 3 (rounds to 18.0), `18.29` at index 4: **same** (to the digits the plan used).
Note for the page: with lr 0.15 the **loss** falls monotonically over the 10 steps (V) although `w` oscillates (0, 3.4, 1.14, 2.638, 1.641, ...); "oscillates but shrinks" is about the parameters. With lr 0.05 the loss is still `1.1e-3` after 10 steps, not 0, because the slow direction of the bowl (below) contracts by only 0.988 per step.

**Hessian and critical learning rate (derived mathematics, not library output).** Written in the JSON as `hessian`/`analysis`; independently recomputed with NumPy (`numpy.linalg.eigvalsh`, same values to 1e-9).
`H = (2/N) [[sum x^2, sum x],[sum x, N]] = (2/3)[[14,6],[6,3]] = [[9.3333, 4],[4, 2]]`. Eigenvalues `11.092940` and `0.240393` (eigenvectors `(0.9153, 0.4027)` and `(0.4027, -0.9153)`). Plain gradient descent on a quadratic multiplies the error along an eigenvector by `1 - lr*lambda`, so it converges iff `lr < 2/lambda_max = 0.180295`. Check against the runs: 0.05, 0.1, 0.15 are below and converge; 0.2 is above (factor `1 - 0.2*11.093 = -1.219`) and diverges (V: spectral radius 1.2186; the loss multiplies by about `1.219^2 = 1.485` per step: 41.08/27.67 = 1.485, V). Slow direction: `1 - 0.05*0.2404 = 0.98798` per step (spectral radius of the lr 0.05 run); at 0.1: 0.976; at 0.15: 0.964.
With momentum (`v = mu v + g; p -= lr v`) the per-direction recursion has characteristic polynomial `r^2 - (1 + mu - lr*lambda) r + mu`; stable iff `lr < 2(1+mu)/lambda_max`: `0.2704` for mu 0.5 and `0.3426` for mu 0.9. Spectral radii (largest root modulus; confirmed against the eigenvalues of the explicit 4x4 iteration matrix in NumPy): mu 0.5 -> 0.9753, mu 0.9 -> 0.9487 (complex roots of modulus `sqrt(mu)`, i.e. oscillating decay), consistent with the observed ups and downs of the loss that still shrink in the long run. At 10 steps mu 0.9 has not settled (final loss 0.939).
Plan (3.2 step 5): "eigenvalues 11.09 and 0.240, so lr < 2/11.09 = 0.180": **same**.

**no_zero_grad (V).**
- `accumulate_only`: three forward+backward passes at the same parameters, no step and no `zero_grad`: `gw = -22.666668, -45.333336, -68` and `gb = -10, -20, -30` (accumulation, not replacement). Plan: -22.67, -45.33, -68.00: **same**.
- `training_loop` (lr 0.05, `step()` every iteration, `zero_grad()` never): the gradient `step()` reads is the running sum: `gw` = -22.67, -32.76, -24.67, -2.88, +20.53, +32.58, +26.59, +5.89, -18.04, -31.91; loss = 27.67, 5.49, 3.51, 25.5, 29.4, 7.80, 1.94, 23.1, 30.8, 10.4: oscillates instead of converging, because the effective update is the sum of all past gradients (a momentum with decay 1). The independent NumPy replica reproduces these (tolerance 1e-4 relative since the loop is sensitive).

**skipped_step (V).** `step()` with no `backward()` ever called, `step()` after `backward()`+`zero_grad()`, and `step()` x2 with momentum 0.9 and no gradient all leave `(w,b)=(0,0)` unchanged and throw nothing; the control (backward then step) gives `(1.1333, 0.5)`. `ctor_errors`: `SGD: lr must be positive` (lr 0 and -0.1), `SGD: momentum must be non-negative`.

**Surface (V).** `surface.points`: 25 samples, `w` in {0..4}, `b` in {-1..3}, loss from the real forward pass; e.g. `(0,0)` 27.67, `(0,-1)` 38.67, `(2,1)` is exactly the optimum with loss 0 (`data.optimum`).

Independent replica (plain NumPy float64): all `runs[*].steps` (w, b, loss, gw, gb), `final`, `surface`, `accumulate_only`, eigenvalues and spectral radii match the JSON. Measured float32-vs-float64 deviation over the 6 runs x 10 steps: `|a-b| <= 2.3e-6 (1+|b|)` for every field; pure relative error can reach `1.8e-4` on a near-zero gradient (`gw = 0.0086` at lr 0.1, step 9: float64 `0.0085553`, float32 `0.0085568`). Therefore the JS test tolerance should be `|a-b| <= 1e-5 (1 + |b|)` (this resolves the plan's "tolerance to be measured"); a pure relative `1e-5` would fail. No `Math.fround` needed.

## 3. Observed / Expected / Suspected

**Observed (V).** `backward()` fills `.grad`; `step()` reads `.grad` and rewrites `.data`; `zero_grad()` clears `.grad`; `step()` does not zero; a missing grad skips the parameter without error; SGD formulas as above; the input `x` and target `t` (requires_grad false) are never updated. Hand-built graph (`transpose`, `matmul`, `add`, `mse`) equals `nn::Linear` + `nn::MSELoss` bit for bit.

**Expected (math).** `dL/dy_i = 2 e_i / N` with `e = y - t = [-3,-5,-7]` -> `[-2, -3.333, -4.667]`; `gw = sum_i (dL/dy_i) x_i = -2 - 6.667 - 14 = -22.667`; `gb = sum_i dL/dy_i = -10`; `L = (9+25+49)/3 = 27.667`; step: `w = 0 + 0.05*22.667 = 1.1333`, `b = 0.5`; new `L` by hand: `y = 1.1333x + 0.5 = [1.6333, 2.7667, 3.9]`, `e = [-1.3667, -2.2333, -3.1]`, `L = (1.8678+4.9878+9.61)/3 = 5.4885`. Matches the library.

**Suspected discrepancy (do not put on a lesson page).**
1. **(v0.1.0, fixed at the pin) `MSELoss` with a broadcasting target has a gradient that does not match its loss.** Forward averages over the broadcast result (`squared_data.mean()`, `losses.cpp:16`); backward scales by `2 / input.numel()` (`losses.cpp:30-32`). When `input` and `target` have different shapes that broadcast, the two normalisers differ. Minimal repro (V, run against `v0.1.0`):
   ```cpp
   Variable a(Tensor({1,2,3},{3,1}), true), b(Tensor({3,5,7},{3}), false);   // prediction (3,1), target (3)
   Variable l = nn::MSELoss{}(a, b);   // broadcasts to (3,3): loss = 111/9 = 12.3333
   l.backward();                       // a.grad = [-8, -6, -4]
   // central finite differences of the same loss (h = 0.01): [-2.667, -2.0, -1.333]  -> the library gradient is 3x too large
   ```
   Expected: `dL/da_i = (2/9) sum_j (a_i - b_j) = -2.667` for `i=0`. The classic way to hit it is a prediction of shape `(N,1)` against a rank-1 target of shape `(N)`: no error, a plausible-looking loss, and a gradient `N` times too large. Same-shape inputs (this lesson, and `linear_regression.cpp`) are correct (V). **Consequence for the site:** the JS `mse` op must only be offered for equal shapes (or the lesson must say "same shape"); do not port the broadcast case. Report to the owner; fix on `main` with a test (suggested: normalise by the broadcast element count in backward), then bump the pin.
   **Outcome:** PR #5 made the backward divide by the broadcast count; PR #6 (`38785db`, the pin) makes `forward()` throw `ShapeError` for unequal shapes, so the repro above now throws `MSELoss input shape (3, 1) must match target shape (3)`. Same-shape results are unchanged (V: every trace other than the mismatch entries is identical after the bump).
2. `Module::parameters()` order is unspecified (`nn.cpp:18-24`, audit D5): never index it by position; use `named_parameters().at("weight")`.
3. NV: whether `SGD` keeps working if the `Module` is moved after the optimizer was built (it stores raw pointers into the module's map); not tested, not needed.

## 3b. The v3 page (curriculum v3, brief §6 Part 1, §7, C6, D13): observed / expected / suspected

**Why the page starts at (0, 3).** From (0, 0) the first step (22.67, 10)·lr aims almost exactly at the lowest point (2, 1),
because (0, 0) lies on the steep axis of the valley (its flat coordinate is 0.11 of 2.2). The demo then taught "the gradient points
at the minimum" (review H1). `trace_optim.cpp` therefore adds `off_axis` (additive, appended last): the same model from w = 0,
b = 3, its nudges, and plain SGD at lr 0.05, 0.15 and 0.2. The old keys (`start`, `nudge_start`, `runs`, `after_ten_steps`) are
unchanged and stay for other uses.

**Observed (V, `optim.json` `off_axis` and `why-gradients.json`, every item asserted in `site/test/sgd-step.test.mjs`).**
- `data.points` = (1, 3), (2, 5), (3, 7): `t = 2x + 1`, so the lowest loss is 0 at (2, 1).
- `off_axis.start`: loss 6.667 (typical miss √L = 2.6 euros), gradient (−10.67, −4) from `nn::Linear` + `nn::MSELoss`.
- `off_axis.nudge`: the same nudges for w and b (1 … 0.001). At h = 0.1: w gives −10.2, b gives −3.9; both sequences come strictly
  closer to the backward() values as h shrinks. The rounded numbers shown (6.667, 6.277, 0.1) give the shown ratio −3.9.
- lr 0.05, step 1: w = 0.5333, b = 3.2. b moves up, away from the best fee 1, so the step does not aim at the star (the angle between
  the first step and the direction to the star is above 45 degrees). Step 2: w = 0.7778 with gradient −4.889. The trail bends within
  five steps (more than 30 degrees), every step is shorter than the last, and every step lowers the loss. After ten steps it is on
  the valley floor (steep coordinate under a tenth of the flat one) but still more than 1 from the star.
- lr 0.15: every step changes the sign of the steep coordinate and shrinks it (zig-zag across the valley), and the flat coordinate
  shrinks at every step (creeps along the floor), but after ten steps more than half of it is left.
- lr 0.2: the first step lands w at 2.133, near the best w, and b at 3.8. The steep coordinate changes sign and grows at every step,
  the loss grows at every step (to 304, typical miss 17), each step still starts downhill, and the flat coordinate still shrinks.
- Cumulative quiz (`why-gradients.json`): `steep.probes[0].grad = −96 = 4 × probes[0].grad`, same start w = 0, same w_min = 3.

**Expected mathematics.** `L(w, b) = (1/3) Σ (w x_i + b − t_i)²` is an exact quadratic with Hessian `[[9.333, 4], [4, 2]]`
(eigenvalues 11.09 and 0.240: a long narrow valley). Plain gradient descent multiplies the error along an eigenvector by
`1 − lr·λ`: steep −0.66 at lr 0.15 and −1.22 at lr 0.2; flat 0.988, 0.964, 0.952 at lr 0.05, 0.15, 0.2. So at lr 0.2 only the steep
direction diverges: the steep direction sets the largest safe learning rate, which is what §5 says without a number (`2/λ_max =
0.1803` stays in the data and in this note). The gradient is perpendicular to the level sets; the widget shows it because `fitView`
gives one unit of w and of b the same pixel length.

**Suspected discrepancy.** None. Not verified: how a screen reader announces the map (role="img" with a changing aria-label that
names the arrow, a status line for Step, Reset, the presets and the fog, and an sr-only table of the trail).

## 4. Misconceptions to address
On the v3 page: "the gradient points at the lowest point" (refuted by the map: from (0, 3) the first step moves b away from the best
fee, and the task says "The arrows do not point at the star"); "the optimizer computes the gradient" (refuted by the real line of
`SGD::step`, which reads `grad`); "a step against the gradient always lowers the loss" and "a larger learning rate is faster"
(refuted with the widget: at lr 0.2 the loss grows at every step); "the gradient tells the distance" (the cumulative quiz).
Momentum (C6) moves to `optimizers`, `zero_grad` to `training-loop` and the appendix (`torc-pitfalls`).

## 5. Test hooks
`site/test/sgd-step.test.mjs` checks the claims above, the JS port against the three recorded off-axis runs, the arrows (`arrowsAt`:
the step ends at the next trail point and the gradient arrow is its mirror image), that the arrow tip stays inside the view at every
preset and step (`viewFor`, review M1), the quoted line, and the names (no update, iteration, momentum, zero_grad, slope, W, bias,
minimum, "your point", `loss.backward`, Hessian or 0.18 on the page). `optim.test.mjs` keeps the port's checks against every run.

## Lesson status (v3, `template: 3`, `loop: 'adjust'`)

Files: `site/js/lessons/sgd-step.js`; `site/js/widgets/descent.js` (one widget, two instances: lr 0.05 with fog; lr 0.05, 0.15,
0.2; start and base view from `off_axis`), `contour2d.js` (gradient arrow labelled "gradient × lr", fog, "lowest point", contour and
step labels that never collide, the view holds the arrow tip), `sgd-ui.js`, `sgd-code.js`; `tools/trace/trace_optim.cpp`
(`off_axis`); `site/test/sgd-step.test.mjs`. `loop-timeline.js` stays for the appendix lesson `torc-pitfalls`.

**As built** (after the three reviews): hero → Questions (gradient with two weights; does a step go straight to the lowest loss; what
sets the largest safe learning rate) → "Three taxi rides" (the points, +2 euros per km, the model from (0, 3), the typical miss) →
predict (how to nudge two weights) → 1 two nudges (worked b nudge, the w ratio) → 2 the gradient as a list, a vector, the contour map
with its legend and the `contour-map` refresher → map widget (2 tasks, fog) → 3 one step for both weights (worked; b moves away from
the star) → "Your turn" (the second step for w) → 4 the real line of `SGD::step` (once per group of weights) → predict (lr 0.2:
compute the first w) → rates widget (2 tasks) → 5 the steep direction limits the learning rate → transfer (steep and flat weight) →
cumulative (`why-gradients`, the steep bowl) → explain → The answer. About 1,390 words with every quiz text and the refresher.

**Course code card**: ten plain SGD steps on `nn::Linear(1, 1)` with `nn::MSELoss`, the loss named `L` as on the page;
`tools/snippets/check.mjs` checks that they land on `runs[0].final` and that `zero_grad()` leaves no gradient. The card keeps
`zero_grad()` (and its symbol) because a loop without it is wrong code; the page does not teach it.
