# Lesson `optimizers` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`optim::SGD(params, lr, momentum = 0)` (`src/optim.cpp`): `v = momentum·v + g; p -= lr·v`, or `p -= lr·g` without momentum.
`optim::Adam(params, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)`: `m = b1·m + (1-b1)·g`, `v = b2·v + (1-b2)·g²`, bias correction
with `t = step_++` (`step_` starts at 1), `p -= lr · m̂ / (sqrt(v̂) + eps)`. `optim::AdamW(params, lr, weight_decay = 0.01, ...)`:
the Adam update, then `- lr·wd·p` with the value of `p` before the step (decoupled weight decay). All three `step()` functions skip
a parameter whose `has_grad()` is false, so AdamW does not decay it. The lesson calls Adam's second average `s` (the source calls it
`v`), because the page uses `v` for the momentum velocity.

## Observed (tools/trace/trace_optimizers.cpp → site/data/optimizers.json)
- Valley `L = 0.5 w1² + 50 w2²` from (-4, 0.4), 50 steps: SGD lr 0.019 ends at loss 1.175 (w1 = -1.533), SGD lr 0.019 with momentum
  0.9 at 0.0868, Adam lr 0.2 at 0.0355. Adam's first step is (+0.2, -0.2).
- First step with lr 0.1 on gradients 0.001 and 1000: Adam changes both parameters by -0.1, SGD by -0.0001 and -100.
- Adam and AdamW (lr 0.1, wd 0.1) on `(w1 - 2)² + 0·w2²` from (0, 3): under Adam w2 stays 3, under AdamW it is `3·0.99^k`
  (1.815 after 50 steps). A parameter `u` outside the loss has no gradient and stays 3 under both.
- Writing the trace hit TorC's lifetime rule: a helper that built the loss and returned only `L` left the graph with destroyed
  intermediates, and `backward()` threw `autograd graph input expired before backward()`. The ops are now written out in the loops.

## Expected (mathematics)
Plain gradient descent on a quadratic is stable for `lr < 2/λmax` (here 0.02), and the flat direction then shrinks by `1 - lr·λmin`
per step. Adam's first step is `lr·g/(|g| + eps)`, about `lr·sign(g)`. AdamW with a zero gradient gives `w_k = w_0 (1 - lr·wd)^k`.
The tests recompute every SGD, momentum and Adam step from the recorded positions.

## Suspected discrepancy
None. (The skipped decay for a parameter with no gradient is by design of the `has_grad()` check, and the lesson says so.)

## v3 additions (curriculum v3, template 3; brief §6, §7, C4, C6, D18)
**Observed** (`sweep`, `race`, `bowl`, `race_surface`; schema in `docs/trace-schema.md`):
- Sweep on the valley, ten lrs from 0.001 to 3, one run each, 50 steps from (-4, 0.4). Best final loss: plain SGD 1.175 at 0.019
  (0.03 ends at 1.0e31, 0.1 and above make TorC's `sum()` throw `NumericalError`), momentum 0.9 0.0471 at 0.01, Adam 0.00909 at 1,
  AdamW (wd 0.01) 0.0173 at 1. Momentum at 0.03 ends at 0.137, below plain SGD's best. Adam at 0.019 ends at 4.76, above plain
  SGD's 1.175 at the same rate.
- The race at those rates: momentum's loss rises at 22 of the 50 steps (the swing across the valley) and ends 25 times below
  plain SGD. Adam and AdamW overshoot to w1 = 1.8 and w2 = -0.6, outside `valley.surface`, so the map draws `race_surface`.
- Round bowl `L = 2 w1² + 2 w2²`, same start and grid: plain SGD's best is lr 0.2 (each step multiplies w by 1 - 0.2·4 = 0.2); its
  loss is below 1e-4 from step 4 and exactly 0 in float32 at the end. Momentum 0.00646 (lr 0.003), Adam 0.0355 (lr 1), AdamW 0.0188
  (lr 3). The momentum sweep on the bowl is not monotone in lr (0.00646 at 0.003, 0.150 at 0.01): a final loss of an oscillating
  path depends on where the swing is at step 50, so "best" on a coarse grid is partly luck. The page says "one run each".

**Expected mathematics.** On a quadratic, plain SGD is stable for lr < 2/c_max (0.02 on the valley, 0.5 on the bowl); heavy-ball
momentum with μ = 0.9 is stable up to lr < 2(1 + μ)/c_max = 0.038 on the valley, which is why 0.03 works with momentum and not
without. On the round bowl one lr suits both directions, so plain SGD converges geometrically at its best rate. Adam's step is about
lr in each coordinate at first and shrinks only as the gradient falls below the long memory of s (β₂ = 0.999), so with a fixed lr it
does not settle as fast as SGD on a well-scaled problem. These are the reasons the page gives in words only ("every direction is
equally steep, so one learning rate suits both parameters"); it does not claim the stability formulas.

**Suspected discrepancy.** None in TorC. Teaching limits: two problems, one start, one run per lr; the page says so and does not
claim a general ranking. The race map of the bowl uses non-equal axis scales, so the round bowl's contours look like ellipses.

**C6.** Momentum is taught here only; `sgd-step` no longer shows it.

## Lesson status
Rewritten as `template: 3` on 2026-11-02 (branch `part4-train`): two instances of the race widget (`optimizer-widgets.js`: plain
SGD and momentum on the valley; all four on two problems with a map and a loss-vs-step plot on a log axis, curves labelled on the
plot, no table). The `code` in `course.json` is unchanged. Earlier history: published 2026-10-20. Widget: `optimizer-widgets.js` (optimizer race, contour lines from the recorded loss grid). Tests:
`site/test/optimizers.test.mjs`. The code lines were compiled and run against v0.1.0. Checked: Node tests, `check_site`,
`tools/prose/check.sh`, Chromium at both widths, axe-core clean.
Independent review 2026-10-20 (no blockers; fixed: the gradient fields of the decay runs were read after `zero_grad()` and
are now read between `backward()` and `step()`, the summary says AdamW shrinks each weight that has a gradient, beta and epsilon
are explained). Not done: screen reader.
