# Lesson `mnist-mlp` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`data::MNISTDataset(path, max_samples = 0, sample_shape = {})` (`src/data.cpp`): one CSV line per image, the label then the
pixels; every pixel is divided by 255 (`rows[i][j + 1] / 255.0f`). `data::DataLoader::next_batch` returns a last, smaller batch
(`std::min(current_ + batch_size_, size)`), so one epoch of 60,000 images in batches of 128 is 469 steps. `nn::Sequential`,
`nn::Linear(in, out, init_std = 0, seed)` (Kaiming normal start from `seed` when `init_std <= 0`), `nn::ReLU`,
`nn::CrossEntropyLoss`, `optim::AdamW(params, lr, weight_decay = 0.01)`. The model and the settings are those of
`examples/mnist_mlp/mnist_mlp.cpp` (batch 128, 20 epochs, lr 5e-4), with fixed seeds (`settings` in the trace).

## Data
`tools/trace/fetch_mnist.sh` downloads the four IDX archives, checks their SHA-256, and writes `mnist_{train,test}.csv` with raw
pixels 0..255. TorC's own `datasets/mnist/convert_to_csv.py` writes `p / 255`, which `MNISTDataset` divides by 255 again (logged
in `docs/library-audit/v0.1.0-findings.md`).

## Observed (tools/trace/trace_mnist_mlp.cpp → mnist-mlp.json; trace_mnist_mistakes.cpp → mnist-mistakes.json; trace_mnist_seeds.cpp → mnist-seeds.json)
- 26,506 parameters; layers `(32, 784)`, `(32, 32)`, `(10, 32)`, as the real line on the page.
- Validation run (C2): 55,000 training images, the last 5,000 held out. After pass 1: 8.1 mistakes per 100 on 5,000 training
  images, 6.9 on the validation images (validation lower). After pass 20: 1.6 and 2.8 (whole numbers on the page: 2 and 3). The
  training curve keeps falling; the validation curve falls more slowly (its loss is roughly flat from about pass 12). A small gap:
  overfitting a little, as a degree (C10).
- The validation run reads the test set once, after training: 9,661 of 10,000 (3 mistakes per 100); the page quotes this. The main
  run (the pad network, all 60,000 images) and the seed runs measured the test set after every epoch (`epochs[].test_accuracy`,
  `test_correct_by_epoch`); the page says so, binds none of those values, and chooses nothing with them. Its final 9,666 is the
  confusion matrix.
- Three seed sets (C5): 96.66 %, 96.66 %, 96.51 % on the test set. The page shows the range as mistakes per hundred, 3.3 to 3.5
  (`mnist-seeds.json` `mlp.min_mistakes_per100` and `mlp.max_mistakes_per100`, added in `14e119d`, role `contrast`).
- At epoch 1 the validation images have fewer mistakes (6.9) than the 5,000 training images (8.1); different images, so the
  counts differ a little by chance. The validation curve is flat over the last five epochs and rises from epoch 18 (2.6) to 20 (2.8).
- Confusion matrix (test set, shown only to look, nothing chosen with it): the largest off-diagonal count is a four read as a nine
  (19); a nine as a four is 9. Nines go wrong most as eights (14), fours (9) and sevens (9). The nine has the lowest per-digit
  accuracy (958 / 1,009 = 0.9495). The first of the 24 recorded mistakes (test image 115) is a four read as a nine.
- First-layer weights: the mean neighbour difference relative to the mean size is 0.85 in the middle and 1.04 at the edge (about
  1.41 for independent random values), so the middle forms patches and the edge looks speckled. 67 pixels have no ink in any
  training image; their weights stay near the Kaiming start (std 0.049 against 0.0505 × the AdamW decay). Pixels with rare ink get
  large weights (AdamW scales each step by the size of recent gradients). The page says only what the tasks show.

## Expected (mathematics)
The browser forward pass (`digit-widgets.js` `forward`: `x·Wᵀ + b`, ReLU, twice, then `x·Wᵀ + b`) must give TorC's logits;
the test allows 1e-4. Per-digit accuracy is the diagonal over the row sum; the matrix sums to the test count. Mistakes per 100
are `T.human(…, 'per100')`, 100 minus the whole percent.

## Suspected discrepancy
None in the parts used here (see the data note above for the converter).

## Lesson status
Rewritten 2026-11 as a `template: 3` lesson, revised after the three review gates (H1 and the medium items), "Reading handwritten digits with a network" (brief §6, §7, C2, C5, D9). Widgets:
`digit-widgets.js` (digit pad and weights gallery; two additive pad options, `markWrong` and `torcNote`, off by default for the other callers) and `mnist-widgets.js` (training vs validation curves, confusion
matrix with the grid of mistakes). Tests: `site/test/mnist-mlp.test.mjs`. The home-page code is compiled by the snippet check
(compile only). Not done: a second review round, human think-aloud, screen reader, a real touch device.
