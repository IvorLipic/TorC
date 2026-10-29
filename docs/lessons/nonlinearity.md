# Lesson `nonlinearity` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`relu` (`src/autograd.cpp`, forward `x > 0 ? x : 0`, backward passes the gradient only where the input is `> 0`, so the
derivative at exactly 0 is 0), `sigmoid` (backward `s(1 - s)` from the saved output), `softmax(a, axis)` and `Tensor::softmax(axis)`
(subtracts the maximum of each slice before `exp`), `nn::Linear`, `nn::MSELoss`, `optim::Adam`.

## Observed (tools/trace/trace_nonlinear.cpp → site/data/nonlinearity.json)
- relu and sigmoid on 81 inputs from -4 to 4 (built as `(k - 40) / 10`, so the input 0 is exact on every CPU; the trace build
  also sets `-ffp-contract=off`, because a fused multiply-add made `-4 + 0.1·40` equal `6e-8` and `relu'` there 1) with derivatives from `backward()` of `sum(f(z))`. `sigmoid'(0) = 0.25`,
  `sigmoid'(4) = 0.01766`, `relu'(0) = 0`.
- softmax of `[1, 2, 3]`, `[11, 12, 13]` and `[1001, 1002, 1003]` are identical (`0.09003, 0.2447, 0.6652`); `[5, 5, 5]` gives a third each.
- Fit to `t = x²` on 21 points, hidden width 8, Adam lr 0.05, 400 steps, the same hand-set start weights for both models. The
  linear model starts as the line `0.4·x - 0.1` (loss 4.398), falls to 1.710 by step 25 and ends at 1.709 as the line `y = 1.4667`
  (the mean of the targets); multiplying out the layers gives the same line within 2e-7. With relu the loss is 0.0041 after 400 steps.

## Expected (mathematics)
With one sample per row, as in `nn::Linear`: `(x W1ᵀ + b1) W2ᵀ + b2 = x (W2 W1)ᵀ + (b1 W2ᵀ + b2)`. The least-squares line for `x²` on symmetric points is flat at the mean
(recorded as `derived.best_line_intercept`). Softmax is invariant to adding a constant to every entry.

## Suspected discrepancy
None.

## Lesson status
Published 2026-10-16. Widgets: `nonlinear-widgets.js` (activation plot, curve fit), recorded values only. Tests:
`site/test/nonlinearity.test.mjs`. Checked: Node tests, `check_site`, `tools/prose/check.sh` (0 hard), Chromium at 1366×768 and
390×844, axe-core clean. Independent review 2026-10-16 (no blockers); its findings are fixed: the trace no longer depends on
FMA, the relu derivative is drawn in two pieces, the start line is tilted, terms (activation function, neuron, classifier) are
defined, the formula uses the row convention of the lesson Layers, and Adam is no longer named before its lesson. Not done:
screen reader.
