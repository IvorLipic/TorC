# Lesson `generalization` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC v0.1.0)
`data::TensorDataset`, `data::DataLoader` (one loader per set; the test loader is only read), `nn::Linear`, `nn::MSELoss`,
`optim::AdamW(params, lr, weight_decay)` (decoupled decay `- lr·wd·p` each step, see `docs/lessons/optimizers.md`),
`Variable::NoGradGuard` for the evaluation passes.

## Observed (tools/trace/trace_generalization.cpp → site/data/generalization.json)
- Data (made by the trace program, not by TorC): 10 training samples at evenly spaced x in [-1, 1] and 100 test samples at
  uniform x, targets `sin(pi x)` plus noise uniform in [-0.3, 0.3) (variance 0.03). The randomness is `std::mt19937` output with
  plain arithmetic, which is the same in every standard library.
- Model: `nn::Linear(15, 1)` on the features x, x², ..., x¹⁵, weights 0 at the start, AdamW lr 0.05, full batch, 20000 steps.
- No weight decay: the training loss falls to 8.4e-5 (lowest 3.7e-5); the test loss is lowest after step 200 (0.0567) and rises
  to 0.128. The sum of the squared weights grows to 441.
- Weight decay 0.1: training loss 0.0045, test loss 0.0599 at the end (it does not rise), squared weights 13.2.
- Weight decay 1: training loss 0.0588, test loss 0.178 (underfitting).

## Expected (mathematics)
No model can get a test loss far below the noise variance (0.03), because the test noise is independent of everything the model
saw. The training loss can go below it, because a flexible model fits the training noise. The tests check the data against
`sin(pi x)`, the step-0 losses against the mean of `t²`, and the order of the three runs.

## Suspected discrepancy
None.

## Lesson status
Published 2026-10-20. Widget: `generalization-widgets.js` (the model's curve and the two loss curves for three weight decays).
Tests: `site/test/generalization.test.mjs`. The code block is the home-page snippet, which the snippet check compiles and runs.
Checked: Node tests, `check_site`, `tools/prose/check.sh`, Chromium at both widths, axe-core clean. Not done: independent review,
screen reader.
