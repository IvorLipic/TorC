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

**Steep bowl, four notebooks cost 12 euros (V; `steep` key, added for brief C3).** `L(w) = (4w - 12)^2 = 16(w-3)^2`, same
minimum `w = 3`. Probes: w = 0: `e = -12`, `L = 144`, `grad = -96`; `steep.compare` (both bowls at w = 0, 1, 1.2, 6): the
distance to 3 is the same, the gradient ratio is exactly 4 at every weight (`grad_ratio`). Real `SGD` with lr 0.05 from 0 lands
at `4.8` (`steep.trajectories.0.states.0.w_next`, factor -0.6: past the bottom). `steep.checks`: grid == 16(w-3)^2 and
32(w-3), SGD == manual update (bitwise).

**v3 derived fields (V, double arithmetic on the recorded float32 values).** Each trajectory state has `step`, `distance`,
`distance_next`, `kept`, `crossed`; each trajectory has `kept`, `kept_percent` of step 1. lr 0.05: steps 1.2, 0.72, 0.432;
kept 0.6 at every state (within 1.2e-6, float32 drift). lr 0.1: kept 0.2. lr 0.25: kept 1, crossed every step. lr 0.3: kept
1.4. `nudge.miss_before` 6, `miss_after` 5.98, `fall` 0.2396, `ratio` -23.96.

## 3. Observed / Expected / Suspected

**Observed (V).** (1) `backward()` returns, for each leaf, the derivative of the scalar loss with respect to it at the current values; for scene `why` that is `-24` at `w=0`. (2) `SGD::step()` is exactly `w <- w - lr*grad` in float32 for a single parameter (bitwise). (3) The root `L` has no stored gradient. (4) A constant (`requires_grad=false`) leaf gets no gradient.

**Expected (mathematics, derived by hand and re-checked in Python).**
- `dL/dw = 2 x (x w - t) = 8 (w - 3)`; `d2L/dw2 = 8`. Chain: `dL/dw = (dL/de)(de/dp)(dp/dw) = (2e)(1)(x) = (-12)(1)(2) = -24` at `w = 0`.
- Secant slope: `(L(h)-L(0))/h = (4h^2 - 24h)/h = 4h - 24`; tends to `-24` linearly in `h`.
- Exact quadratic expansion: `L(w + d) = L(w) + L'(w) d + 4 d^2`. So the tangent (straight-line) prediction misses by exactly `4 d^2` (lr 0.05, d = 1.2: `5.76 = 12.96 - 7.2`).
- Gradient descent on this bowl: `w' - 3 = (1 - 8 lr)(w - 3)`. Hence `w_k - 3 = (1-8 lr)^k (0 - 3)` and `L_k = 36 (1-8 lr)^(2k)`. Stable iff `|1 - 8 lr| < 1`, i.e. `0 < lr < 0.25`. `lr = 0.125` gives factor `0`: lands on 3 in one step (this is the Newton step `1/L''`, special to an exactly quadratic bowl). `lr = 0.25`: factor `-1`, bounces forever (`w` = 0, 6, 0, 6, ...; `L` constant 36). `lr = 0.3`: factor `-1.4`, `L` multiplies by `1.96` per step (`36 -> 70.56 -> 138.30`) (all V against the trajectories).
- The stability predicate `|1-8 lr| < 1` agrees with the observed sign of divergence for the five presets (V; `stable_predicted` field).

**Suspected discrepancy.** None found for this lesson. Known and harmless: the library never stores the root's gradient (audit D4, `docs/library-audit/implementation-contract.md`); the scene draws it as the seed `1`.

## 4. Misconceptions the page refutes, with the evidence it uses

1. "The gradient tells the distance to the answer" (brief C3, the page's one refutation with widget evidence). In the
   two-notebook bowl the gradient is `8(w - 3)`, proportional to the distance, so that bowl alone would support the
   misconception. The four-notebook bowl has the same minimum and start (same distance 3) and the gradient -96 instead of -24.
   The two-bowls widget shows both at w = 0, 1, 1.2, 6 (recorded `steep.compare`): same distance, four times the gradient.
2. "A smaller learning rate is always better." The check quiz `best-lr` and section 5: after eight steps lr 0.1 is closer to
   3 than lr 0.05 (each step keeps 20 % vs 60 %), and 0.25 / 0.3 never settle (100 %, 140 %). The transfer quiz shows that the
   same lr 0.05 overshoots on the steep bowl (0 -> 4.8).
3. "The gradient points to the minimum." In 1-D the sign does say on which side the minimum lies (probes at 0 and 6). The page
   says what the gradient tells (direction and steepness here) and what it does not (where the bottom is, how far). The 2-D
   case, where the gradient does not point at the minimum, belongs to `sgd-step`.
Not on the page any more: the tangent promise (`tangent_L_next`), the float32 secant floor at h = 0.001 (the widget shows 3
significant figures, -24.0 either way), and the chain rule (moved to the future `backprop` lesson).

## 5. Test hooks

All in `site/test/why-gradients.test.mjs`: the JS model (`landscape-model.js`) against grid, probes, nudge and every trajectory;
the nudge-lens rows against `secants` (and the JS curve between them, double vs float32 within 1e-2 on the secant); the two-bowls
rows against `steep.compare` and the probes; every claim of the prose from the data (story constants, the nudge quiz, "ten times
smaller nudge, ten times smaller change", the signs, the two worked steps, the shrinking step, 60 % kept, -96 = 4 x -24, the
doubled miss and slope, the fog tasks, the best learning rate after eight steps, the shares, the transfer step 4.8, the
cumulative quiz); one use of "slope"; `\mathrm{lr}` and no `g`; the SGD line against the pinned `src/optim.cpp` when the trace
checkout exists; no hand-typed digit; KaTeX for every formula and widget string; readout formats.

## 6. Lesson status (v3 rewrite, 2026-10-29; brief Appendix B, C3, C7, D20, decisions 13-23)

Files: `site/js/lessons/why-gradients.js` (template 3, loop `adjust`), widgets `site/js/widgets/nudge-lens.js` (new),
`two-bowls.js` (new), `bowl.js` (fog mode), `landscape-tex.js` (3-figure readouts, the new step line), `landscape-model.js`
(unchanged). `why-gradients.scenes.js` is kept unchanged for the future `backprop` lesson; `lessons.test.mjs` still lints its texts.

**As built**: hero -> Questions (2) -> "The price of a notebook" (setup) -> predict quiz (nudge to 0.01, new loss 35.76) -> 1 The
gradient (ratio by nudging, the one use of "slope", dL/dw read aloud) -> nudge lens (task: nudge 1 down to 0.001) -> 2 One step
(worked 0 - 0.05 x (-24) = 1.2, the arrow read as "replace w by", `L.backward()` as the real line) -> "Your turn" quiz (1.92,
reveal worked, steps 1.2 then 0.72, 60 % kept) -> 3 A tempting idea that is wrong (four notebooks, -96 vs -24) -> two bowls (task
w = 1, 6) -> 4 Steps in the fog (fog analogy with its limit) -> fog bowl (3 tasks: 0.05 in the fog, predict 0.25, 0.3) ->
check quiz best-lr -> 5 How to choose the learning rate (shares 100 %, 140 %; In TorC panel with the `SGD::step()` line) ->
transfer quiz (steep bowl, lr 0.05 -> 4.8) -> cumulative quiz from `what-is-learning` (which guess has the largest loss) ->
explain -> The answer (next question: many weights).

**Adaptations of Appendix B (reported to the lead).**
- Appendix B says that with four notebooks "each nudge of w now changes the prediction four times as much". That is wrong:
  the prediction is 4w instead of 2w, so it changes twice as much, and the miss is twice as large (12 vs 6). Together they give
  four times the gradient. The page says "twice" twice.
- The nudge reveal binds `nudge.actual_dL` and `nudge.ratio` at 2 significant figures (-0.24, -24) instead of "24 times the
  nudge", because the ratio is negative.
- The learner step's reveal uses a `worked()` block for 1.2 - 0.05 x (-14.4) = 1.92, because a reveal sentence may hold at most
  two numbers. The step sizes and the 60 % come from the new trace fields (`step`, `kept_percent`), not from arithmetic.
  "The distance goes from 3 to 1.8 to 1.08" is left out (three numbers in one sentence, and the 60 % says it).
- The fog analogy sits in section 4 with the fog widget, not in the refutation section, and its limit keeps the bank's "a hiker
  remembers the path". The refutation uses its own widget (two bowls) so that the evidence is on the page.
- The fog bowl offers lr 0.05, 0.1, 0.25, 0.3 (0.1 added for the "smaller is not always better" check; 0.125 dropped with the
  one-step special case). The start slider is gone (every task starts at w = 0).
- The old check "why does lr 0.125 not work for a real network?" is replaced by a transfer quiz (the same lr on the steep bowl),
  so the word "curvature" is not needed.
- The nudge lens offers the five recorded nudge sizes as buttons (no dragging). Its measured values are the recorded secants.
- The cumulative quiz comes from `what-is-learning`, which is still `planned` in `course.json`, so its "comes from the lesson"
  link has no page until that lesson exists.
- dL/dw is written once, with a new notation entry `dl-dw` (lesson `why-gradients`) in `site/js/content/notation.js`.
- `course.json` `code` now shows `L.backward()`, `optim::SGD` and `step()` (no `Variable w(...)` setup line); its SPEC in
  `tools/snippets/check.mjs` checks the gradient -24 and the new weight 1.2 against the pinned TorC.

**Not done here**: pedagogy reviewer, cold reader, independent review, human think-alouds (brief §10), a real screen reader.
