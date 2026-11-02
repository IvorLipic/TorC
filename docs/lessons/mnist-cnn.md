# Lesson `mnist-cnn` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`data::MNISTDataset(path, max_samples = 0, sample_shape)` (samples of shape (1, 28, 28) with `{1, 28, 28}`), `nn::Conv2d`,
`nn::ReLU`, `nn::Flatten`, `nn::Linear`, `nn::Sequential`, `nn::CrossEntropyLoss`, `optim::AdamW(params, 1e-3)`. The model is that
of `examples/mnist_cnn/mnist_cnn.cpp` (see `docs/lessons/convolution.md` for `conv2d` itself).

## Observed
- `trace_mnist_cnn.cpp` → `mnist-cnn.json`: 9098 parameters; batch 128, lr 1e-3, weight decay 0.01, 5 epochs (the example uses
  10, `settings.example.epochs`): test correct 9435, 9693, 9715, 9764, 9794 (206 mistakes). The MLP (`mnist-mlp.json`): 20 epochs,
  lr 5e-4, 9666 (334 mistakes). A fresh full `run.sh` on 2026-11-01 reproduced both files unchanged.
- `trace_mnist_seeds.cpp` → `mnist-seeds.json` (C5): three random starts each, each model at its own settings. MLP 96.66 %,
  96.66 %, 96.51 %; CNN 97.94 %, 97.53 %, 97.70 %. The worst CNN run beats the best MLP run by 0.87 points. On the page: the best
  MLP run makes about 3 mistakes per hundred, the worst CNN run about 2 (`T.human` per100, tested).
- Confusion matrices: fours read as nines MLP 19, CNN 14; nines as fours 9 and 5; sevens as twos MLP 10, CNN 20 (the CNN does not
  win on every pair; the reveal says so). The brief's example question "why would a CNN confuse 7 and 1 less often?" is not
  supported by these runs (7↔1: 3 and 3), so the page asks about 4 and 9, which also continues `mnist-mlp`'s question.
- First-layer kernels (Pearson correlation of each map with a stroke-edge indicator over the first 16 test images, computed by
  the test with the port): the kernels with a negative top row and a positive rest (2 and 5) follow top edges (0.82, 0.89); the
  only kernel with a positive left and negative right column (7) follows right edges (0.88).
- The last four shown images (18, 115, 247, 259) are misread by the CNN.

## Expected (mathematics)
The browser port (`digit-widgets.js` `conv2d`, `forwardCnn`) gives TorC's logits (tolerance 1e-4) and the first maps (1e-5).
Transfer quiz: on 56 × 56 images the maps are 28 × 28 and 14 × 14, so the linear layer reads 16·14·14 = 4 × 784 cells; the
convolution layers keep C_out·(C_in·K·K + 1) parameters.

## Suspected discrepancy
None. Not verified: why the CNN separates 4 and 9 better. The matrices show how often, not why; the page labels its reason as
likely and not proven.

## Lesson status
Rewritten as `template: 3` on 2026-11-01 (branch `part5-cnn`; brief §6, §7, C5, C9), new title "Reading digits with a
convolutional network". Questions, a predict (which network makes fewer mistakes), a worked mistake count with the learner's
step (206), the honest comparison (C9 wording, settings, three starts), the In TorC panel (the model code with the shape comments
on their own lines and `B` for the batch; the example trains 10 epochs, the run 5), the trained kernels (widget, tasks that link
the kernel's colours to the edges its map finds), the 4/9 reasoning question, the digit pad, a transfer quiz (images twice as
large), a cumulative quiz from `generalization` (choosing with the test set), an explain prompt, the recap of the whole course
(one line and one question per part, links, recommends going back; it doubles as the Part 5 review) and The answer with the
bridge to `from-digits-to-chatgpt`. The next-lesson link appears when the epilogue is published (`neighbors()` skips planned
lessons). The misconception "more parameters read better" is refuted by the predict and the comparison. Tests:
`site/test/mnist-cnn.test.mjs`.
