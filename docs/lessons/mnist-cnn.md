# Lesson `mnist-cnn` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC v0.1.0)
`data::MNISTDataset(path, 0, {1, 28, 28})` (samples of shape (1, 28, 28)), `nn::Conv2d`, `nn::ReLU`, `nn::Flatten`, `nn::Linear`,
`nn::Sequential`, `nn::CrossEntropyLoss`, `optim::AdamW(params, 1e-3)`. The model is that of `examples/mnist_cnn/mnist_cnn.cpp`
(see `docs/lessons/convolution.md` for `conv2d` itself).

## Observed (tools/trace/trace_mnist_cnn.cpp → site/data/mnist-cnn.json)
- 9098 parameters. Batch 128, lr 1e-3, 5 epochs (the example uses 10): test images correct 9435, 9693, 9715, 9764, 9794. With
  10 epochs (one run while choosing the setting) it reached 9829.
- About 26 s per epoch here against 1.8 s for the MLP: `conv2d` and its backward pass are plain loops. The trace takes about two
  minutes, and the drift check in CI runs it on every push.
- Wrong after the first 16 images: 18 (3 read as 8), 115 (4 as 9), 247 (4 as 2), 259 (6 as 0). The MLP reads 115, 247 and 259
  wrongly too, and 18 correctly; the test runs the MLP's port on these images.

## Expected (mathematics)
The browser port (`digit-widgets.js` `conv2d`, `forwardCnn`) must give TorC's logits (test tolerance 1e-4, observed 1.5e-5) and
the first feature maps after relu (observed 4.4e-7). Flatten keeps the (C, H, W) order, so the port reads the second
convolution's output in that order.

## Suspected discrepancy
None.

## Lesson status
Published 2026-10-20 as "Digits with a CNN" (the plan said "Project: MNIST CNN"). Widgets: `digit-widgets.js` (`mountCnnMaps`,
and `mountDigitPad` with the CNN's forward pass). Tests: `site/test/mnist-cnn.test.mjs`. Checked: Node tests, `check_site`,
`tools/prose/check.sh`, Chromium at both widths (drawing, the test images), axe-core clean. Independent review 2026-10-20: the page said the MLP reads one
of the four wrongly (it reads three; the old test could not see it), now "most of them", tested with the MLP's port; the comparison
is worded "with the settings of the two TorC examples" and tested at equal epochs; the kernel claim is softened. Not done: screen
reader, a real touch device.
