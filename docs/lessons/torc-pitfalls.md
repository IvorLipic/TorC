# Lesson `torc-pitfalls` — internal note

Not shipped. Appendix page "Common errors in TorC programs" (part `using-torc`, outside the lesson order; brief §6 appendix, D6,
decision 22). New in v3. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`Variable::backward_with_grad` (`include/torc/autograd.hpp`: `accumulate_grad` adds to a stored gradient, `build_topo` and the loop
throw `TorcError` when an input that needs a gradient has an expired lifetime token, `tape_.clear()` on the root at the end),
`Variable::zero_grad` (sets `has_grad_` false, so `step()` skips the parameter), `SGD::step`, `Adam::step`, `AdamW::step`
(all: `if (!param->has_grad()) continue;`), `DataLoader::{has_next, next_batch, reset}`, `MSELoss::forward` (since PR #6 it throws
`ShapeError` when the two shapes differ), `CrossEntropyLoss::forward` (rank-2 logits, rank-1 or `[n, 1]` targets, integer targets in
range), `nn::Linear::forward` (`matmul`), `Tensor::operator[]`, `Tensor::reduce_axis`.

## Observed (tools/trace/autograd_trace.cpp and the files named below)
- `autograd_trace.json` `accumulation`: L = w·w at w = 3: 6; a second `backward()` on the same loss 6 (nothing happens); a new forward
  pass and its `backward()` 12; `has_grad()` false after `zero_grad()`.
- `autograd_trace.json` `lifetime[]`: `Variable sq = mul(w, w); Variable L = sum(sq);` works (6). `Variable L = sum(mul(w, w));`,
  `Variable L = mse(layer(x), t);` and a loss built from a tracked Variable made inside a function all throw `TorcError`
  "autograd graph input expired before backward(); keep all graph Variables alive". Naming the model output works.
- `autograd_trace.json` `unused_parameter`: a loss that uses one of two parameters: the used one goes from 3 to 2.4 (lr 0.1, gradient 6),
  the other keeps 3 and has no gradient (`step()` skipped it, no error).
- `optim.json`: `no_zero_grad.training_loop` (ten iterations at lr 0.05): with `zero_grad()` the loss falls at every iteration to about
  0.001; without it the stored gradient changes sign and the loss jumps between about 2 and 30. `skipped_step[]`: `step()` before any
  `backward()`, after `zero_grad()`, and with momentum and no gradient change nothing and throw nothing. `ctor_errors[]`.
- `training.json` `no_reset`: ten steps in epoch 1 and zero in epochs 2 and 3 (loss unchanged); `exhausted`: `next_batch()` after
  the last batch throws `TorcError` "DataLoader::next_batch: no more batches in current epoch".
- `losses.json`: `mse_mismatch` (`ShapeError` "MSELoss input shape (3, 1) must match target shape (3)"), `errors[]` for
  `CrossEntropyLoss` (not-integer, out-of-range, rank-1-logits, length; a column of targets works).
- `linear.json` `shape_errors[]`; `tensors.json` `index_errors[]` and `reduce_error`.

## Expected (mathematics)
Gradient of L = w·w is 2w. A step of lr·gradient. Sums of gradients over passes.

## Suspected discrepancy
None on this page. The AdamW claim (a parameter without a gradient is skipped and does not decay) is read from `src/optim.cpp`,
not recorded in a trace.

## Lesson status
New 2026-11 (`template: 3`, appendix): predict quizzes for gradient accumulation, graph lifetime, a missing `reset()` and the MSE
shape; five numbered sections with `gotcha` boxes for every message; the zero_grad loop widget (`loop-timeline.js`, kept from
`sgd-step`, no main-path lesson imports it). Tests: `site/test/torc-pitfalls.test.mjs`. Checked: Node tests, `check_site`,
`tools/prose/check.sh`, Chromium at 1366x768 and 390x844. Not done: human think-aloud, screen reader.
