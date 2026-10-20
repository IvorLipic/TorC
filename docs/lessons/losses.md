# Lesson `losses` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC v0.1.0)
`nn::MSELoss` (`include/torc/nn/losses.hpp`, `src/nn/losses.cpp`): mean over all entries of `(y - t)²`. It does not check the
shapes: `sub()` broadcasts them, and the backward pass divides by `input.numel()`, not by the broadcast count.
`nn::CrossEntropyLoss`: rank-2 logits `(B, C)`, targets of shape `(B)` or `(B, 1)` holding integral class indices in `[0, C)`;
log-softmax of each row, negative log-probability of the target, mean over the batch. Its backward is `(softmax - onehot) / B`.

## Observed (tools/trace/trace_losses.cpp → site/data/losses.json)
- MSE on predictions `2.5, 0, 2` and targets `3, -0.5, 2`: loss 0.1667, gradient `-0.3333, 0.3333, 0`.
- Cross-entropy for one sample, three rows of logits times three targets. Uniform logits give `log 3` = 1.0986 for every target.
  Logits `2, 1, 0.1`: softmax 0.659, 0.242, 0.099; loss 0.417 (target 0), 1.417 (1), 2.317 (2). Logits `4, 1, -2`, target 2: 6.051.
- A batch of the two (mild, target 0; confident, target 2): loss 3.234, the mean; each gradient row is the single-sample row / 2.
- Logits `[z, 0]`, target 0, z from -6 to 6: at z = -6 the cross-entropy gradient is -0.9975 and the gradient of the squared error
  on the softmax probabilities is -0.00492.
- MSE with predictions of shape (3, 1) and targets of shape (3): no error, loss 3.333 (the mean over the 3 × 3 broadcast
  differences), gradient `2, -3, 1`, which is 3 times the derivative of the reported loss. The page warns about this.
- Errors (all `ShapeError`): a non-integral label, a label out of `[0, C)`, rank-1 logits, a label count that differs from the batch
  size. Labels of shape `(B, 1)` are accepted.

## Expected (mathematics)
`∂L/∂y = 2(y - t)/N` for MSE; `∂L/∂z = (softmax(z) - onehot(t)) / B` for cross-entropy. For two classes with logits `[z, 0]` and
`p = σ(z)`: CE gradient `p - 1`; squared-error gradient `-2(1 - p)² p`, which goes to zero as `p → 0`. The tests check all of these.

## Suspected discrepancy
`MSELoss` with shapes that broadcast: no shape check, and a gradient that does not match the reported loss. Logged in
`docs/library-audit/v0.1.0-findings.md`. The gradient is fixed on `main` by PR #5, after v0.1.0; the broadcasting without an error
is kept there by design. When `TORC_REF` moves past PR #5, `mse_broadcast.grad` in the trace changes, and the losses test that
expects the old gradient must change with it.

## Lesson status
Published 2026-10-16 as "Losses". Widgets: `loss-widgets.js` (loss explorer, gradient comparison). Tests:
`site/test/losses.test.mjs`. Both code blocks follow the home-page snippet, which the snippet check compiles and runs.
Checked: Node tests, `check_site`, `tools/prose/check.sh`, Chromium at both widths, axe-core clean.
Independent review 2026-10-20 (no blockers). Fixed after it: the gradient sign (the gradient is negative for the correct class,
and the step raises that logit), the two-class setup of the comparison is stated, one name per thing (logit, label, one-hot
target), the labels on the page come from the trace, the MSE shape warning. Not done: screen reader.
