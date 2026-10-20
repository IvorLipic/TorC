# Lesson `optimizers` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC v0.1.0)
`optim::SGD(params, lr, momentum = 0)` (`src/optim.cpp`): `v = momentum·v + g; p -= lr·v`, or `p -= lr·g` without momentum.
`optim::Adam(params, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)`: `m = b1·m + (1-b1)·g`, `v = b2·v + (1-b2)·g²`, bias correction
with `t = step_++` (`step_` starts at 1), `p -= lr · m̂ / (sqrt(v̂) + eps)`. `optim::AdamW(params, lr, weight_decay = 0.01, ...)`:
the Adam update, then `- lr·wd·p` with the value of `p` before the step (decoupled weight decay). All three `step()` functions skip
a parameter whose `has_grad()` is false, so AdamW does not decay it. The lesson calls Adam's second average `s` (the source calls it
`v`), because the lesson Using the gradient already uses `v` for the momentum velocity.

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

## Lesson status
Published 2026-10-20. Widget: `optimizer-widgets.js` (optimizer race, contour lines from the recorded loss grid). Tests:
`site/test/optimizers.test.mjs`. The code lines were compiled and run against v0.1.0. Checked: Node tests, `check_site`,
`tools/prose/check.sh`, Chromium at both widths, axe-core clean.
Independent review 2026-10-20 (no blockers; fixed: the gradient fields of the decay runs were read after `zero_grad()` and
are now read between `backward()` and `step()`, the summary says AdamW shrinks each weight that has a gradient, beta and epsilon
are explained). Not done: screen reader.
