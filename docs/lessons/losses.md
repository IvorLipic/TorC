# Lesson `losses` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`nn::MSELoss` (`include/torc/nn/losses.hpp`, `src/nn/losses.cpp:12-42`): mean over all entries of `(y - t)²`. `y` and `t` must
have the same shape; otherwise `forward()` throws `ShapeError` (PR #6). The backward pass is `2 (y - t) / n`, `n` the element count.
`nn::CrossEntropyLoss`: rank-2 logits `(B, C)`, targets of shape `(B)` or `(B, 1)` holding integral class indices in `[0, C)`;
log-softmax of each row, negative log-probability of the target, mean over the batch. Its backward is `(softmax - onehot) / B`.

## Observed (tools/trace/trace_losses.cpp → site/data/losses.json)
- MSE on predictions `2.5, 0, 2` and targets `3, -0.5, 2`: loss 0.1667, gradient `-0.3333, 0.3333, 0`.
- Cross-entropy for one sample, three rows of logits times three targets. Uniform logits give `log 3` = 1.0986 for every target.
  Logits `2, 1, 0.1`: softmax 0.659, 0.242, 0.099; loss 0.417 (target 0), 1.417 (1), 2.317 (2). Logits `4, 1, -2`, target 2: 6.051.
- A batch of the two (mild, target 0; confident, target 2): loss 3.234, the mean; each gradient row is the single-sample row / 2.
- Logits `[z, 0]`, target 0, z from -6 to 6: at z = -6 the cross-entropy gradient is -0.9975 and the gradient of the squared error
  on the softmax probabilities is -0.00492.
- MSE with predictions of shape (3, 1) and targets of shape (3) (`mse_mismatch`): `ShapeError: MSELoss input shape (3, 1) must
  match target shape (3)`. The page does not show it (audit findings stay in `docs/library-audit/`). At v0.1.0 the same call gave
  no error, loss 3.333 (the mean over the 3 × 3 broadcast differences) and gradient `2, -3, 1`, 3 times the derivative of that loss.
- Errors (all `ShapeError`): a non-integral label, a label out of `[0, C)`, rank-1 logits, a label count that differs from the batch
  size. Labels of shape `(B, 1)` are accepted.

## Expected (mathematics)
`∂L/∂y = 2(y - t)/N` for MSE; `∂L/∂z = (softmax(z) - onehot(t)) / B` for cross-entropy. For two classes with logits `[z, 0]` and
`p = σ(z)`: CE gradient `p - 1`; squared-error gradient `-2(1 - p)² p`, which goes to zero as `p → 0`. The tests check all of these.

## Suspected discrepancy
None at the pin. At v0.1.0, `MSELoss` with shapes that broadcast gave no error and a gradient that did not match the reported loss
(`docs/library-audit/v0.1.0-findings.md`). PR #5 fixed the gradient and PR #6 made unequal shapes a `ShapeError`. With the pin at
`38785db`, the trace records the error (`mse_mismatch` replaces `mse_broadcast`) and the losses test checks the message.

## Lesson status
Published 2026-10-16 as "Losses". Widgets: `loss-widgets.js` (loss explorer, gradient comparison). Tests:
`site/test/losses.test.mjs`. Both code blocks follow the home-page snippet, which the snippet check compiles and runs.
Checked: Node tests, `check_site`, `tools/prose/check.sh`, Chromium at both widths, axe-core clean.
Independent review 2026-10-20 (no blockers). Fixed after it: the gradient sign (the gradient is negative for the correct class,
and the step raises that logit), the two-class setup of the comparison is stated, one name per thing (logit, label, one-hot
target), the labels on the page come from the trace, the MSE shape warning. Not done: screen reader.
2026-10-29 (pin bump to `38785db`): the MSE shape warning is removed from the page (pedagogy review D6). The library now throws
for unequal shapes, and the code comment "y and t must have the same shape" stays.
