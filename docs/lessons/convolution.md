# Lesson `convolution` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC v0.1.0)
`Tensor::conv2d(weight, stride, padding)` (`src/tensor.cpp`): NCHW input, weight `(C_out, C_in, KH, KW)`, zero padding, output
size `(H + 2P - K) / S + 1` with integer division, cross-correlation (the kernel is not flipped), no bias. `torc::conv2d(Variable,
Variable, stride, padding)` with autograd. `nn::Conv2d(in, out, kernel, stride = 1, padding = 0, init_std = 0, seed)`: Kaiming
normal weights, zero bias, `forward` = `conv2d` + `reshape(bias, {1, C_out, 1, 1})` broadcast. `nn::Flatten`: `(N, ...)` to
`(N, product)`. Errors: `conv2d input channels {} must match weight channels {}`, `conv2d output dims must be positive, ...`.

## Observed (tools/trace/trace_conv.cpp → site/data/conv.json)
- 5 × 5 input with a vertical edge (columns 0, 0, 4, 4, 4, centre pixel 8) and the kernel `1 0 -1 / 2 0 -2 / 1 0 -1`: outputs
  3 × 3 (stride 1, padding 0), 5 × 5 (stride 1, padding 1) and 3 × 3 (stride 2, padding 1).
- The first MNIST test image (a 7), read with `MNISTDataset(path, 1, {1, 28, 28})`, through a vertical-edge and a horizontal-edge
  kernel (stride 1, padding 1): two 28 × 28 maps, before and after relu.
- The example's CNN on (2, 1, 28, 28): (2, 8, 14, 14), (2, 16, 7, 7), Flatten (2, 784), Linear (2, 10). Parameters 80, 1168 and
  7850, 9098 in total. `Linear(784, 8)` has 6280.

## Expected (mathematics)
`(28 + 2 - 3) / 2 + 1 = 14`, `(14 + 2 - 3) / 2 + 1 = 7`. A convolution has `C_out·C_in·K·K + C_out` parameters. The tests
recompute every recorded output from its input and kernel (exactly for the small example, within 1e-5 for the digit).

## Suspected discrepancy
None.

## Lesson status
Published 2026-10-20. Widgets: `conv-widgets.js` (sliding window, feature maps). Tests: `site/test/convolution.test.mjs`. The CNN
code is the example's; the home-page snippet is compiled and run by the snippet check. Checked: Node tests, `check_site`,
`tools/prose/check.sh`, Chromium at both widths, axe-core clean. Not done: independent review, screen reader.
