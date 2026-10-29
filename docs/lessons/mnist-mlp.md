# Lesson `mnist-mlp` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`data::MNISTDataset(path, max_samples = 0, sample_shape = {})` (`src/data.cpp`): one CSV line per image, the label then the
pixels; every pixel is divided by 255. `nn::Sequential`, `nn::Linear(in, out, init_std = 0, seed)` (Kaiming normal start from
`seed`), `nn::ReLU`, `nn::CrossEntropyLoss`, `optim::AdamW(params, lr, weight_decay = 0.01)`, `data::DataLoader`. The model and
the settings are those of `examples/mnist_mlp/mnist_mlp.cpp` (batch 128, 20 epochs, lr 5e-4), with fixed seeds.

## Data
`tools/trace/fetch_mnist.sh` downloads the four IDX archives from `storage.googleapis.com/cvdf-datasets/mnist`, checks their
SHA-256 (the published MNIST values), and writes `build/mnist/mnist_{train,test}.csv` with raw pixels 0..255. TorC's own
`datasets/mnist/convert_to_csv.py` writes `p / 255`, which `MNISTDataset` divides by 255 again (logged in
`docs/library-audit/v0.1.0-findings.md`). CI caches the archives (`verify.yml`).

## Observed (tools/trace/trace_mnist_mlp.cpp → site/data/mnist-mlp.json)
- 26,506 parameters. Test images correct: 9204 after epoch 1, 9666 of 10000 after epoch 20 (96.66 %). The mean training loss
  of the last epoch is 0.0713. The test count rises with small drops (epochs 10, 12, 14, 17, each under 0.5 %), so there is no
  overfitting in 20 epochs. The test loss is not recorded.
- Per class, the digit 9 is read worst (958 of 1009, 94.9 %); the page computes this from the data.
- The first 16 test images are all read correctly. The first four errors after them are images 115 (4 read as 9), 149 (2 as 5),
  151 (9 as 8) and 233 (8 as 7).
- About 1.8 s per epoch on the CI-like container, so the trace adds under a minute to `run.sh`.

## Expected (mathematics)
The browser forward pass (`digit-widgets.js` `forward`: `x·Wᵀ + b`, ReLU, twice, then `x·Wᵀ + b`) must give TorC's logits;
the test allows 1e-4 and observes 8.4e-6. The preprocessing of a drawing follows the MNIST recipe (crop, fit into 20 × 20,
centre of mass at the centre); 18 of the 20 recorded images keep their class after being upscaled and preprocessed again.

## Suspected discrepancy
None in the parts used here (see the data note above for the converter).

## Lesson status
Published 2026-10-20 as "Digits with an MLP" (the plan said "Project: MNIST MLP"; a colon in a title breaks the writing rules).
Widgets: `digit-widgets.js` (weights gallery, digit pad). Tests: `site/test/mnist-mlp.test.mjs`. The home-page code is compiled
by the snippet check. Checked: Node tests, `check_site`, `tools/prose/check.sh`, Chromium at both widths (drawing with the mouse,
the test images), axe-core clean. Independent review 2026-10-20 (no blockers; fixed: "MLP" and "hidden layer" defined, the overfitting answer no longer says
"almost every epoch", the digit pad announces only on a click or the end of a stroke). Not done: screen reader, a real touch device.
