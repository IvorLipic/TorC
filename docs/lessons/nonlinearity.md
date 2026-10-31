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
- `hinges` (V): the trained relu model's eight hidden neurons as contributions `W2_k·relu(W1_k x + b1_k)` on the grid from the
  real ops; their sum plus `b2` equals the model's prediction exactly (`max_abs_diff_vs_prediction = 0`). Neuron 4 has its kink
  at x = 0.149 and rises to the right; neuron 1 has its kink at x = -1.17 and rises to the left (1.66 at x = -2.2).
- `collapse` (V, added for v3): two real `nn::Linear(1, 1)`, `f = 2x + 1` and `g = 3x + 4`, on x = -2 .. 3: `g(f(x))` equals the one
  `Linear(1, 1)` with `6x + 7` exactly. With `relu` between them, `g(relu(f(-2))) = g(0) = 4`, where `6x + 7` gives -5.

## Expected (mathematics)
With one sample per row, as in `nn::Linear`: `(x W1ᵀ + b1) W2ᵀ + b2 = x (W2 W1)ᵀ + (b1 W2ᵀ + b2)`. The least-squares line for `x²` on symmetric points is flat at the mean
(recorded as `derived.best_line_intercept`). Softmax is invariant to adding a constant to every entry. A sum of `H` ReLU hinges plus a
constant is piecewise linear with at most `H` kinks (one per neuron, fewer if two coincide). The JS of the hinge builder (`hingeSum`,
`typicalMiss`) is plain addition and √(mean squared miss) at the 21 data points; with every neuron on, its square equals the recorded
loss (tested).

## Suspected discrepancy
None.

## Lesson status (v3, template 3, "Activation functions")

Files: `site/js/lessons/nonlinearity.js`, `site/js/widgets/hinge-builder.js` (new), `nonlinear-widgets.js` (curves labelled on the
plot, readouts at 3 significant figures, the curve-fit readout is the typical miss √loss instead of the bare loss, "gradient" for the
derivative), `site/test/nonlinearity.test.mjs`, `tools/trace/trace_nonlinear.cpp` (`collapse`, additive).

**As built**: hero → Questions (does a second layer bend; what a function between the layers changes) → setup "Points on a curve"
(21 points of t = x², two layers, 8 hidden neurons, a hidden layer, 400 training steps with "an improved kind of gradient descent",
Adam not named) → predict (what the no-activation model draws; the old hint is gone; the "tilted line" option names the symmetry belief)
→ 1 two layers act like one (f, g, worked 6x + 7 and the check at x = 2; then for layers in words) → 2 activation function and ReLU →
Your turn g(relu(f(-2))) = 4 → curve-fit widget (2 tasks) → 3 each ReLU neuron is a hinge (analogy from the bank with its limit;
misconception "ReLU itself is the curve" refuted with the widget) → hinge builder (3 tasks: neuron 4 alone, all on, neuron 1 off) →
transfer (bends of three neurons) → 4 sigmoid (σ and e introduced, gradient 0.25 vs 0.018, "that weight barely learns", ReLU's
gradient is one for positive sums) → activation plot (1 task) → cumulative from `why-gradients` (step = lr × gradient) → explain → answer.
Budgets: 1,396 words, 4 numbered sections, 5 quizzes, 2 worked blocks, palette 1 (the analogy).

**Deviations**: softmax moved to `losses` (its data and tests stay, `losses` reads them); the overflow fact left the page and no In
TorC panel was added; the exp-ln refresher is not opened (e is named as a fixed number a little below three, linked to the notation
guide); "the weights barely learn" points to `backprop` for the chain-rule reason. The old line `Variable r = relu(h);` code block
left the concept text (course.json `code` keeps relu and sigmoid only, SPEC updated).

**Definition of done**: Node tests, `check_site`, `tools/prose/check.sh` (0 hard), `tools/snippets/check.mjs`, `run.sh` then
`--check`, Chromium at 1366×768 and 390×844 by keyboard (predict quiz, hinge toggles, slider), no console errors, labels inside the
plots. Not done: pedagogy reviewer, cold reader, independent review, human think-alouds, screen reader.
