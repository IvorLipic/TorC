# Lesson `convolution` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`Tensor::conv2d(weight, stride, padding)` (`src/tensor.cpp`): NCHW input, weight `(C_out, C_in, KH, KW)`, zero padding, output
size `(H + 2P - K) / S + 1` with C++ integer division (the numerator is never negative for a valid call, so it rounds down;
a non-positive size throws), cross-correlation (the kernel is not flipped), no bias. `torc::conv2d(Variable, Variable, stride,
padding)` (`src/autograd.cpp`) with autograd: the weight gradient is `db[...] += a[...] * g` over every batch item and every
output position, so a kernel weight gets the sum of the gradients of all its uses (the cumulative quiz). `nn::Conv2d(in, out,
kernel, stride = 1, padding = 0, init_std = 0, seed)`: Kaiming normal weights, zero bias, `forward` = `conv2d` +
`reshape(bias, {1, C_out, 1, 1})` broadcast. `nn::Flatten`: `(B, ...)` to `(B, product)`. Errors: `conv2d input channels {} must
match weight channels {}`, `conv2d output dims must be positive, ...`.

## Observed
- `tools/trace/trace_conv.cpp` → `site/data/conv.json`: the 5 × 5 example (columns 0, 0, 4, 4, 4, centre pixel 8) with the kernel
  `1 0 -1 / 2 0 -2 / 1 0 -1` at stride/padding 1/0, 1/1, 2/1; two edge kernels on the first MNIST test image; the example CNN on
  (2, 1, 28, 28): (2, 8, 14, 14), (2, 16, 7, 7), (2, 784), (2, 10), parameters 80, 1168, 7850, 9098 in total; `Linear(784, 8)`
  6280. New for v3 (additive): `derived.layers[2]`, the steps of the size rule (28 + 2 − 3 = 27, 27 / 2 = 13.5, 13, 14; and
  14 + 2 − 3 = 13, 6.5, 6, 7) and of the parameter count (8 × (1·9 + 1) = 80; 16 × (8·9 + 1) = 1168). Plain arithmetic in the
  producer, checked there against the library's shapes and counts. A full `run.sh` (2026-11-01) changed only `conv.json`.
- `tools/trace/trace_shift.cpp` → `site/data/shift.json` (C8): on all 10,000 test images moved by whole pixels with zero fill.
  3 px right: MLP 5390 (53.9 %), CNN 7653 (76.5 %); unmoved 9666 and 9794. The CNN falls with every move in every direction but
  stays above the MLP. 4 px down: MLP 14.7 %, CNN 28.9 %. The first-layer maps of test image 0 (a 7) moved 2 px right equal the
  original maps moved one cell, over 1456 cells (max difference 0). A 1 px move: differences 2.36 (unmoved) and 2.24 (one cell).
- The lesson's JS (`shift-widgets.js`, with the `conv2d` port of `digit-widgets.js` and the weights of `mnist-cnn.json`)
  reproduces the original and the moved maps within 1e-5 and both 1 px differences within 1e-4 (tested). In the port, the 4 px
  move of image 0 also moves the maps exactly two cells; the 3 px move changes them.

## Expected (mathematics)
`⌊(28 + 2 − 3) / 2⌋ + 1 = 14`, `⌊(14 + 2 − 3) / 2⌋ + 1 = 7`. A convolution has `C_out·(C_in·K·K + 1)` parameters, independent of
the image size. Equivariance: a move by `k·S` pixels moves every map by `k` cells, except at the borders (zero fill and padding
agree here) and where ink leaves the image. A move that is not a multiple of the stride samples the image at other positions,
so the map changes. With stride 1 every whole-pixel move is a multiple (the transfer quiz; checked with the port at stride 1,
not by a TorC run). A model that ends in Flatten → Linear is not shift-invariant, because the linear layer has a weight per cell.

## Suspected discrepancy
None.

## Lesson status
Rewritten as `template: 3` on 2026-11-01 (branch `part5-cnn`; brief §6, §7, C8, D14). Questions, a predict on the parameter count
(80 vs 6,280 vs "more for a larger image"), the sliding window (picture) before the ⌊ ⌋ rule, a worked size example with the
learner's step (the second layer, 7), channels with a worked parameter count (1,168), an In TorC panel (the Conv2d call with
the shape comments on their own lines and `B` for the batch, no kernel flip, the channel error), the moved feature map (aha
widget, task asks for a guess first), the misconception "a CNN reads a digit equally well anywhere" refuted with the measured
accuracy on moved test images, a transfer quiz (stride one), a cumulative quiz from `backprop` (a shared weight gets the sum
of its gradients), an explain prompt and The answer. Palette: the stencil analogy (with its limit) and the In TorC panel.
Deviation: the brief's second predict ("what happens to the maps after a 2 px move?") became the first task of the moved-map
widget, because a quiz there did not fit the 1,400-word budget. Widgets: `conv-widgets.js` (sliding window), new
`shift-widgets.js` (moved maps, accuracy on moved images). Tests: `site/test/convolution.test.mjs`.
