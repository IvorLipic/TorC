# Lesson `generalization` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected. Page title (v3): "Does it work on new data?".

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`data::TensorDataset`, `data::DataLoader` (one loader per set; the validation and test loaders are only read), `nn::Linear`,
`nn::MSELoss`, `optim::AdamW(params, lr, weight_decay)`, `Variable::NoGradGuard` for the evaluation passes.
`AdamW::step` (`src/optim.cpp`): `p = p - lr * m_hat / (sqrt(v_hat) + eps) - (lr * wd) * p`, with the old `p` in the decay term,
for every parameter that has a gradient (weight and bias alike). So each step keeps `1 - lr * wd` of each parameter besides the
Adam update (`derived.decay_factor`: 1, 0.995, 0.95). A parameter without a gradient is skipped and does not decay; that does not
happen here.

## Observed (tools/trace/trace_generalization.cpp → site/data/generalization.json)
- Data (made by the trace program, not by TorC): 10 training points at evenly spaced x in [-1, 1], 30 validation and 100 test
  points at uniform x, targets `sin(pi x)` plus noise uniform in [-0.3, 0.3). The "more data" set is the 10 training points plus
  90 more drawn after the validation points, so all earlier data are unchanged (checked: no value of the old file changed).
- Model: `nn::Linear(15, 1)` on x, x², ..., x¹⁵ (16 parameters, `settings.parameters`), weights 0 at the start, AdamW lr 0.05,
  full batch (loader batch 64 for 10 points; 128 for the 100 points), 20000 steps.
- Typical misses (square roots of the MSE, `derived.misses`), no weight decay: at the validation stop (step 200, also the best
  test step) training 0.087, validation 0.27, test 0.24, ratio 3.1; at step 20000 training 0.0092, validation 0.46, test 0.36,
  ratio 50. The validation curve has a bump (0.47 at steps 10-15, 0.60 at step 30) before its lowest point. Near x = 0.9 the
  final curve is 1.4 where the wave is 0.31.
- Weight decay 0.1: validation stop at step 1500 (0.29); at the end training 0.067, validation 0.29 (it does not climb), squared
  weights 13 against 441.
- Weight decay 1: training 0.24, validation 0.43 at the end (underfitting, ratio 1.75).
- 100 training points, no weight decay: training 0.16, validation 0.20, test 0.17 at the end (ratio 1.27).
- The wave itself (`derived.true_wave_loss`): typical miss 0.18 on the validation set, 0.17 on the test set.

## Expected (mathematics)
No model can get a typical miss far below the noise on new points: the noise of a new point is independent of everything the
model saw. The RMS of noise uniform in [-0.3, 0.3) is 0.3/√3 ≈ 0.17; on the 30 drawn validation points the wave's own miss is
0.18. A flexible model's training miss can go below it, because it fits the training noise. With 100 points the 16-parameter
least-squares fit averages the noise, so both misses approach the noise level.

## Suspected discrepancy
None.

## Teaching choices
- C2: the stopping step is chosen from the validation curve; the game locks the slider and only then shows the test miss and,
  labelled "only so that you can compare", the test curve. The remedies widget never shows the test set.
- C10: overfitting as a degree, measured as validation miss ÷ training miss (worked at the last step, the learner does the
  validation stop). "More parameters ⇒ overfitting" is refuted by the same 16 parameters on 100 points and by the MNIST anchors.
- The noise floor is stated in section 5 without the word "variance" (the separate noise-floor check quiz of the old page was cut
  to keep the lesson under 1,400 words).
- Single draw of random points, labelled (C5).

## Review fixes (2026-11, three gates)
- The validation set is harder by chance (the wave itself misses it by 0.18, the test set by 0.17), so from step 20 on the test
  curve lies below the validation curve (0.24 against 0.27 at the stop). The page does not claim that the validation miss is
  below the test miss. It says, in the game result next to those numbers, that small sets differ by chance by more than the
  small optimism of one choice, and that the optimism grows with every further choice made with the same points (tested).
- The ratio of the two misses compares steps of one run only; between runs the page compares the validation miss (weight decay
  0.1 ends at a ratio of 4.4 and weight decay 1 at 1.75, so the ratio alone would rank them wrongly).
- "The weights grow, and the curve swings" is shown by a weight-size readout (sum of the squared weights: 440 against 13). With a
  hundred points the weights also grow (150) while the curve follows the wave, so the page claims only what the two ten-point
  runs show.
- The test miss at the stop appears only in the game, after the commit. The trace uses one learning-rate constant (`LR_SET`);
  the JSON is byte-identical.

## Lesson status
v3 rewrite 2026-11 (branch `part4-general`). Widgets: `val-stop.js` (the game), `generalization-widgets.js` (helpers and the
remedies widget). Tests: `site/test/generalization.test.mjs`. The code block is the home-page snippet, which the snippet check
compiles and runs (now with the validation loader). The three review gates ran later,
and their fixes are `cd2a498`. Pending: human readers.
