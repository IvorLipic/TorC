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

## v3 additions (curriculum v3, template 3; brief §6, §7, C1, C5)
**Observed** (`batch_compare`, `sorted_order`, `accumulate`, extra fields of `gradient_estimate`; schema in `docs/trace-schema.md`):
- At w = b = 0 the gradient of all 100 samples is (gw, gb) = (−1.505, −1.484). The ten batches of 10 of the first training epoch
  (seed 7, the order after one `reset()`, so `loaders[10, shuffle].epochs[1]`) give gw from −2.462 to −0.528 and gb from −2.412
  to −0.233, every one with the sign of the full gradient, none within 0.05 of it. Their mean equals the full gradient to 1e-6.
  One `optim::SGD` step (lr 0.1) on the first batch alone moves w from 0 to 0.1548.
- Batch 10 against batch 100 (lr 0.1, start 0, loader seeds 7, 8, 9): after 30 steps the losses are 0.0986 to 0.1005 (typical miss
  0.314 to 0.317) and 0.1012 (0.318). After 3000 samples (300 against 30 steps) they are 0.008108 (0.090) and 0.1012 (0.318). After
  300 steps both are at 0.008108. Batch 100 is the same for every seed to 7 significant figures (not bit for bit at every step,
  because the shuffled order changes the order of the float sums). The three batch-10 runs spread apart in the first steps (loss
  2.32 to 2.82 after one step, against 2.60 for batch 100).
- Sorted by x, no shuffle, batch 10: the loss on all samples rises from 3.019 to 3.357 over the first two steps and is still above the start after three (typical miss 1.74
  to 1.83), then falls. Shuffled (seed 7) it falls from the first step. After one epoch: 1.21 against 0.93 (typical miss).
- `accumulate`: at the start, `L.backward()` stores gw −1.505. After `opt.step()`, `zero_grad()`, a new forward pass and
  `L.backward()`: −1.447. The same without `zero_grad()`: −2.952 = −1.505 + (−1.447) (a self-check in the program). Calling
  `L.backward()` twice on the same `L` leaves −1.505, because `backward_with_grad` ends with `tape_.clear()` on the root
  (`include/torc/autograd.hpp`). So the lesson says that gradients add up over a new forward pass and backward, never over a
  second backward on the same loss.

**Expected mathematics.** For MSE, the gradient is a mean over the samples, so the mean of the gradients of equal-sized batches that
partition the data is the full gradient (an unbiased estimate). Far from the minimum the noise of a batch of 10 is small next to the
gradient, so per step batch 10 and batch 100 progress about equally; per sample seen, batch 10 takes ten times more steps. Near the
minimum the full gradient goes to zero, and the batch noise dominates (the loss of batch 10 wobbles slightly by seed). A sorted order
makes consecutive batches biased estimates (the first ones see only small x), which is why the loss rises at first.

**Suspected discrepancy.** None found. The page says "a different random order gives slightly different numbers, so we ran three"
(C5). The comparison is one problem (a line with 100 samples); the page does not claim more.

**Moved.** The `reset()` question (`no_reset`) left this page for the appendix lesson `torc-pitfalls`; the data stays in the trace.
The batch-32 run stays in the trace for the "Your turn" quiz (4 steps per epoch).

## Lesson status
Rewritten as `template: 3` on 2026-11-01 (branch `part4-train`): widgets `training-widgets.js` (batch viewer, batch gradients,
batch-size comparison with three views) and `curve-labels.js` (direct curve labels). The `code` in `course.json` is the five-line
loop of the page; `tools/snippets/check.mjs` runs it and checks that it lands at w 3.0023, b 1.0134 with the gradients cleared.
Review fixes 2026-11-08 (three gates): equal epochs are equal samples here, so the page compares per step (a tie)
and per sample (batch 10 wins); the transfer quiz asks what was held equal; both sizes use lr 0.1, chosen and not tuned, and a larger
batch can often take a larger rate (not tried); "average out" holds at the same w and b for equal batches; the batch viewer uses the
loop's orders (`loop_orders`, reset() before every epoch), batch 10 only, and counts batches taken; the loop's comments sit on their
own lines; one name for the noise. Earlier history: published 2026-10-20. Widgets: `training-widgets.js` (batch viewer, training curves). Tests: `site/test/training-loop.test.mjs`.
Both code blocks were compiled and run against v0.1.0. Checked: Node tests, `check_site`, `tools/prose/check.sh`, Chromium at both
widths, axe-core clean. Independent review 2026-10-20 (no blockers). Fixed after it: section 4 said the true line's loss is the lowest loss, but the
fitted line is a little below it on the training samples (the page now says so, and a test checks the sign); "update" and
"step" are one name each; the runs are found by batch size; the seed of the widget's loader is stated; the log scale is
labelled. Not done: screen reader.
