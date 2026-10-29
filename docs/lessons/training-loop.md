# Lesson `training-loop` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`data::Dataset` (`len`, `get`, `get_batch`, `gather`), `data::TensorDataset`, `data::SyntheticRegression(n, features, weight,
bias, noise_std, seed = 42)` (`include/torc/data.hpp`, `src/data.cpp`): `x ~ U(-1, 1)` from `std::uniform_real_distribution`,
noise from `std::normal_distribution`, one `std::mt19937`. `data::DataLoader(dataset, batch_size, shuffle = false, seed =
random_device)`: the constructor fills the indices `0..n-1` and shuffles them once if `shuffle`; `next_batch()` returns the next
`batch_size` indices (the last batch is smaller) and throws `TorcError` when none is left; `has_next()`; `reset()` sets the
position to 0 and shuffles again. It never resets itself. The loop is the one in `examples/linear_regression/linear_regression.cpp`.

## Observed (tools/trace/trace_training.cpp → site/data/training.json)
- `SyntheticRegression(100, 1, 3, 1, 0.1, 42)`: the true line has loss 0.00828 on the samples (the noise variance is 0.01).
- Batch size 32 gives batches of 32, 32, 32, 4. Without shuffle every epoch is `0..99` in order. With shuffle (seed 7) the
  constructor order and the order after `reset()` differ. After the last batch, `next_batch()` throws
  `TorcError: DataLoader::next_batch: no more batches in current epoch`.
- Linear(1, 1) from weight 0 and bias 0, `optim::SGD` lr 0.1, 30 epochs, shuffled, `reset()` before each epoch: batch 100 (30 steps)
  ends at loss 0.1012 (weight 2.42), batch 32 (120 steps) at 0.00839, batch 10 (300 steps) at 0.00811 (weight 3.002, bias 1.013).
  The batch-10 loss is slightly below the true line's loss, because least squares also fits a little of the noise.
- Without `reset()`, epoch 1 takes 10 steps and epochs 2 and 3 take 0. There is no error.

## Expected (mathematics)
Steps per epoch = ceil(n / batch size). For MSE with this data (mean x = -0.090, mean x² = 0.281), full-batch gradient descent
shrinks the weight error by a factor of about `1 - 2·lr·mean(x²) = 0.944` per step (the coupling with the bias changes this a
little), so 30 steps leave much of it, and 300 steps remove it. The least-squares line on the 100 samples (w = 3.0025,
b = 1.0134) has loss 0.008108, below the true line's 0.008282, because it fits part of the noise; the batch-10 run reaches it. The tests check the step
counts, that each epoch gives each sample once, and the loss of the true line against the recorded samples.

## Suspected discrepancy
- `SyntheticRegression` with `num_features > 1` fills only column 0 of each row. The other columns stay 0 and `y` depends on
  column 0 only. The lesson uses one feature, so this does not show. Logged in `docs/library-audit/v0.1.0-findings.md`.
- The recorded orders and samples come from libstdc++'s `std::shuffle` and distributions. A libc++ build gives other values,
  which the drift check would report. CI uses libstdc++.

## Lesson status
Published 2026-10-20. Widgets: `training-widgets.js` (batch viewer, training curves). Tests: `site/test/training-loop.test.mjs`.
Both code blocks were compiled and run against v0.1.0. Checked: Node tests, `check_site`, `tools/prose/check.sh`, Chromium at both
widths, axe-core clean. Independent review 2026-10-20 (no blockers). Fixed after it: section 4 said the true line's loss is the lowest loss, but the
fitted line is a little below it on the training samples (the page now says so, and a test checks the sign); "update" and
"step" are one name each; the runs are found by batch size; the seed of the widget's loader is stated; the log scale is
labelled. Not done: screen reader.
