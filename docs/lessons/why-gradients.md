# Lesson note: `why-gradients` (internal, not shipped)

Behaviour note for the lesson author. Library = TorC `38785db` (`main` = v0.1.0 + PR #5 + PR #6), fetched to `build/trace/_deps/torc-src/` by `tools/trace/run.sh`; paths below are relative to it.
Data: `site/data/why-gradients.json` (producer `tools/trace/trace_why.cpp`; schema in `docs/trace-schema.md`).
Tags: **V** = verified by running the real library (or an independent NumPy/Python replica, stated), **NV** = not verified.
No lesson prose here, and nothing here goes on the lesson page except what the page's own trace paths quote.

## 1. Library symbols this lesson touches

| Symbol | Where | What it does here |
|---|---|---|
| `torc::mul(a,b)`, `torc::sub(a,b)` | `src/autograd.cpp:195`, `:177` | forward `a*b`, `a-b`; each records one `TapeEntry` on its output. `mul` backward: `grad_a = g*b`, `grad_b = g*a`; `sub` backward: `g`, `-g`. (V: scene `why` gradients) |
| `Variable::backward()` | `src/autograd.cpp:90`, `include/torc/autograd.hpp:115` (`backward_with_grad`) | requires a scalar (`numel()==1`), seeds `1`, topologically orders the tracked inputs, runs each closure once all consumers have contributed, adds into `.grad` with `accumulate_grad` (`autograd.hpp:75`). The root's own gradient is **not stored** (`root_has_grad:false`, V) |
| `optim::SGD(params, lr, momentum=0)` | `include/torc/optim.hpp:8`, `src/optim.cpp:7` (ctor), `:17` (`step`), `:32` (`zero_grad`) | `step()`: for each param with `has_grad()`, `data = data - grad*lr` (momentum 0 branch, `optim.cpp:25-27`). Takes `const std::vector<Variable*>&` |
| `Variable::zero_grad()` | `src/autograd.cpp:110` | `has_grad_=false` |

## 2. The numbers (all from `why-gradients.json`; x = 2, t = 6)

`L(w) = (x w - t)^2 = 4 (w-3)^2`, minimum at `w = 3`.

**Scene `why` at w = 0 (V).** `p = 0`, `e = -6`, `L = 36`. Backward: `e.grad = -12` (two contributions of `-6` from `mul(e,e)`, which lists `e` twice in its tape inputs), `p.grad = -12`, `w.grad = -24`. `x` and `t` have `requires_grad:false` and `grad:null`. Finite-difference check (h=0.01, float32): numerical `-23.9999771`, analytic `-24`, error `2.3e-5` (ok).

**Probes (V).** (w: L / grad) 0: 36 / -24; 0.1: 33.64 / -23.2; 1: 16 / -16; 1.2: 12.96 / -14.4; 3: 0 / 0; 6: 36 / 24; 7.2: 70.56 / 33.6 (float32 prints e.g. `33.6400032`). Plan section 3.1 quotes the same values: **no difference**.

**Grid (V).** 33 points `w = -1..7` step 0.25; `grad == 8(w-3)` and `L == 4(w-3)^2` exactly (max abs difference `0`: every number is exactly representable in float32).

**Secants at w = 0 (V; L from the real forward pass, float32).**

| h | L(h) | (L(h)-L(0))/h | expected `4h-24` |
|---|---|---|---|
| 1 | 16 | -20 | -20 |
| 0.5 | 25 | -22 | -22 |
| 0.1 | 33.6400032 | -23.599968 | -23.6 |
| 0.01 | 35.7603989 | -23.9601135 | -23.96 |
| 0.001 | 35.9760056 | -23.9944439 | -23.996 |

Plan: -20, -22, -23.6, -23.96: **same** (after rounding to the plan's digits). New fact for the page: at `h = 0.001` the float32 secant is `-23.9944`, not `-23.996`: subtracting two nearby 36-ish floats loses digits (float32 spacing near 36 is `3.8e-6`). If the lesson shows h = 0.001 it should either say "about -24.0" or compute the secant in the browser (double, gives -23.996). This is a good, true teaser for `gradient-checking`; do not hide it.

**Nudge, delta = 0.01 at w = 0 (V).** `p: 0 -> 0.02` (local slope 2), `e: -6 -> -5.98` (change 0.02, local slope 1), `L: 36 -> 35.7603989` (change `-0.2396011`, local slope `-12`); product of local slopes `-24`; straight-line prediction `-24 * 0.01 = -0.24`; difference `0.000399` (`= 4 delta^2 = 0.0004`, Expected). Plan: "p +0.02, e +0.02, L -0.2396": **same**. The local slopes are read from the real backward pass as ratios of node gradients (`w.grad/p.grad = 2`, `p.grad/e.grad = 1`, `e.grad/seed = -12`), not typed.

**Trajectories, real `optim::SGD` on a one-element Variable (V), states k = 0..8:**

| lr | w_k | L_k |
|---|---|---|
| 0.05 | 0, 1.2, 1.92, 2.352, 2.6112, 2.7667, 2.8600, 2.9160, 2.9496 | 36, 12.96, 4.6656, 1.6796, 0.60466, 0.21768, 0.078364, 0.028211, 0.010156 |
| 0.1 | 0, 2.4, 2.88, 2.976, 2.9952, 2.9990, 2.9998, 3.0000, 3.0000 | 36, 1.44, 0.0576, 0.0023040, 9.2e-5, 3.7e-6, 1.5e-7, 5.9e-9, 2.3e-10 |
| 0.125 | 0, 3, 3, ... | 36, 0, 0, ... |
| 0.25 | 0, 6, 0, 6, 0, 6, 0, 6, 0 | 36 at every step |
| 0.3 | 0, 7.2, -2.88, 11.232, -8.5248, 19.135, -19.589, 34.624, -41.274 | 36, 70.56, 138.30, 271.06, 531.28, 1041.3, 2041.0, 4000.3, 7840.6 |

Plan table (lr 0.05: w = 0, 1.2, 1.92, 2.352, 2.6112, 2.7667; L = 36, 12.96, 4.666, 1.680, 0.605, 0.218; lr 0.3: 36, 70.56, 138.3, 271.1): **same**. `matches_manual:true` for all five: every real `SGD::step()` produced bit-identical `w - lr*grad` in float32 (V; `max_abs_diff_vs_manual = 0`).
The tangent promise at step 0 for lr 0.05 is `tangent_L_next = 7.2` (`36 - 24*1.2`), actual 12.96 (V). For lr 0.1 the promise is `-21.6`, a negative squared error, which makes the "promise holds only locally" point sharply.

Independent replica (plain NumPy float64, script run during this work, not committed): grid, trajectories (`w -= lr * 8 (w-3)`), and `expected_slope` all match the JSON within `1e-5 (1+|x|)`.

## 3. Observed / Expected / Suspected

**Observed (V).** (1) `backward()` returns, for each leaf, the derivative of the scalar loss with respect to it at the current values; for scene `why` that is `-24` at `w=0`. (2) `SGD::step()` is exactly `w <- w - lr*grad` in float32 for a single parameter (bitwise). (3) The root `L` has no stored gradient. (4) A constant (`requires_grad=false`) leaf gets no gradient.

**Expected (mathematics, derived by hand and re-checked in Python).**
- `dL/dw = 2 x (x w - t) = 8 (w - 3)`; `d2L/dw2 = 8`. Chain: `dL/dw = (dL/de)(de/dp)(dp/dw) = (2e)(1)(x) = (-12)(1)(2) = -24` at `w = 0`.
- Secant slope: `(L(h)-L(0))/h = (4h^2 - 24h)/h = 4h - 24`; tends to `-24` linearly in `h`.
- Exact quadratic expansion: `L(w + d) = L(w) + L'(w) d + 4 d^2`. So the tangent (straight-line) prediction misses by exactly `4 d^2` (lr 0.05, d = 1.2: `5.76 = 12.96 - 7.2`).
- Gradient descent on this bowl: `w' - 3 = (1 - 8 lr)(w - 3)`. Hence `w_k - 3 = (1-8 lr)^k (0 - 3)` and `L_k = 36 (1-8 lr)^(2k)`. Stable iff `|1 - 8 lr| < 1`, i.e. `0 < lr < 0.25`. `lr = 0.125` gives factor `0`: lands on 3 in one step (this is the Newton step `1/L''`, special to an exactly quadratic bowl). `lr = 0.25`: factor `-1`, bounces forever (`w` = 0, 6, 0, 6, ...; `L` constant 36). `lr = 0.3`: factor `-1.4`, `L` multiplies by `1.96` per step (`36 -> 70.56 -> 138.30`) (all V against the trajectories).
- The stability predicate `|1-8 lr| < 1` agrees with the observed sign of divergence for the five presets (V; `stable_predicted` field).

**Suspected discrepancy.** None found for this lesson. Known and harmless: the library never stores the root's gradient (audit D4, `docs/library-audit/implementation-contract.md`); the scene draws it as the seed `1`.

## 4. Misconceptions to address (from plan 3.1, with what the data shows)

1. "The gradient tells you how far to go." It is a rate (loss per unit weight). At w = 0 it is `-24`; the distance to the minimum is 3. `lr` converts rate to distance and the right `lr` depends on curvature (`1/8` here).
2. "Smaller lr is always better / bigger is always faster." Both false: 0.125 reaches the minimum in one step, 0.25 never converges, 0.05 needs many steps (L = 0.0102 after 8).
3. "The tangent line is a prediction." It is a local promise: for a step of 1.2 the line promised 7.2 and the loss was 12.96; at lr 0.1 it promised a negative loss.
4. "The chain rule is a formula to memorise." It is local ratios multiplying; the trace gives the three ratios (2, 1, -12) and their product (-24).
5. "The gradient points at the minimum." In 1-D it is just a sign: from w=6 the gradient is +24 and the minimum is to the left; from w=0 it is -24 and the minimum is to the right (V: `probes`). The step goes against it.
6. The `h = 0.001` secant in float32 is `-23.9944`, not `-23.996`: finite differences have a floating-point floor (teaser for `gradient-checking`).

## 5. Test hooks suggested by the data

(a) JS `buildGraph/runBackward` on scene `why` equals `grid[*].{L,grad}` (rel 1e-5). (b) `w -= lr*g` in double reproduces `trajectories[*].states[k].w` within `1e-5 (1+|w|)`; diverging runs (lr 0.3) grow to `4e3`, relative agreement still `1e-6`. (c) `grad == 8(w-3)` on all grid points. (d) `secants[*].expected_slope` equals `4h-24`; `|slope + 24|` decreases monotonically with `h` even in the float32 record (4, 2, 0.4000, 0.03989, 0.00556), but the `h = 0.001` float32 value misses the exact secant (-23.996) by 0.0016, so assert the exact formula only on the JS (double) value, and compare the float32 record with a tolerance of about `1e-2` there. (e) `stable_predicted` vs sign of `L_8 - L_0`. (f) Quoted numbers (36, -24, 12.96, 7.2, -12, -0.2396) bound to paths: `scenes[0].nodes[L].value`, `scenes[0].nodes[w].grad`, `trajectories[0].states[1].L`, `trajectories[0].states[0].tangent_L_next`, `scenes[0].nodes[e].grad`, `nudge.stages[2].change`.

## 6. Lesson status (concise rewrite, decisions 7-9)

Files: `site/js/lessons/why-gradients.js`, `why-gradients.scenes.js` (graph-lab layout and text for scene `why`),
`site/js/widgets/bowl.js` (the one widget: curve, tangent, start slider, lr presets 0.05/0.125/0.25/0.3, Step/Reset, readout),
`landscape-model.js` (pure model: curve, steps, nudge, factor, through `engine.buildGraph/runBackward`), `landscape-tex.js`
(the update line), `site/test/why-gradients.test.mjs`.

**As built**: hero → TL;DR (4 bullets) → setup (TorC code, `L(w) = 4(w-3)^2`) → 1 slope (`-24` is a rate, not a distance) →
2 step (`w ← w − lr·g`, 0 → 1.2, 36 → 12.96) → predict (lr 0.25 bounces) → bowl widget → 3 step size (factor `f = 1 − 8 lr`:
converge / bounce / diverge / one step) → 4 chain rule (`2 · 1 · (−12) = −24`, `e` used twice adds) → graph lab → 2 checks
(sign at w = 6; why lr = 1/8 does not generalise) → Next. Measured: 388 words, 24 numbers in the text, 2 widgets, 3 questions.

**Removed in the rewrite**: knob, secant/`h`, stepper and step-size widgets (merged into the bowl), the gears widget, the
learning-rate table, four mix-up cards, the recap, the dotted-underline/derived marking, provenance badges and the long footer.
Misconceptions 1, 2 and 5 are covered by sections 1 and 3 and the sign check; 3 (tangent promise) and 6 (float32 secant) are
no longer on the page.

**Definition of done**: tests green (`node --test site/test/`, `tools/check_site.mjs`); browser check at 1366×768 and 390×844
under `/TorC/` (bowl presets, Step, slider by keyboard, no console errors, no horizontal overflow). Independent review done (2026-10-15,
no blocker/high; findings fixed); `tools/trace/run.sh --check` passes. Not done: a real screen reader.
