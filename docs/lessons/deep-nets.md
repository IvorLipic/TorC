# Lesson `deep-nets`: Why deep networks are hard to start

Part 3 `networks`, curriculum v3 (`template: 3`, `loop: 'model'`), the last lesson of the part (it ends with a short review that
links back and never blocks). Brief §6 "deep-nets": short (about 11 minutes). Module `site/js/lessons/deep-nets.js`, widgets
`site/js/widgets/deep-widgets.js` (twins, vanishing, signal), tests `site/test/deep-nets.test.mjs`. Trace: `data/deep.json`
(`tools/trace/trace_deep.cpp`).

## What the page teaches
1. **Twins stay twins**, with the missing reason: same inputs and weights give the same output; the same output weight gives the
   same gradient (the rule of `backprop`); a weight's gradient is its neuron's gradient times its input; so every step keeps them
   equal. The layer acts like one neuron. Misconception refuted with the run: "zero weights are a fine start" (zero is the extreme
   twin case, and with zero output weights no weight moves at all; only the output bias learns, the prediction is 0.5 everywhere).
2. **Vanishing gradients** with sigmoid (the sigmoid's slope is at most 0.25, read from TorC): the first hidden layer gets about
   16,000 times less gradient than the eighth; with ReLU about 2 times. Misconception: "more layers always learn better".
   Single seed, labelled (C5).
3. **The size of the start**: too small shrinks the signal at every layer, too large grows it, TorC's √(2/in) keeps it level.
   The reason for the 2: ReLU sets about half the values (48 %) to zero; without the 2 (√(1/in)) the spread falls at every layer.
   Worked example √(2/32) = 0.25, then the learner computes √(2/784) ≈ 0.0505 for the digit network's first layer.

## Observed (traced)
- `twins[twins]`: equal start, equal at every one of 2,000 steps (bit for bit, parameters and gradients); loss 0.1667 (typical miss
  0.41; predictions 1/3, 1/3, 1, 1/3 for targets 0, 1, 1, 0). `twins[different]`: learns all four points (loss 4e−14).
  `twins[zeros]` (added): every weight and bias starts at 0; every hidden gradient is 0 at every step; only the output bias moves,
  to 0.5; loss 0.25. `long_run[]` now also records both neurons and the output bias (added for the widget).
- `vanishing[]`: sigmoid mean |gradient| per hidden layer 2.0e−6 … 3.3e−2 (rising at every layer), ReLU 0.58 … 1.5;
  `derived.last_over_first_hidden` 16,421 (sigmoid) and 2.02 (ReLU), added.
- `setup.sigmoid_max_slope` = 0.25: `sigmoid(0)` backward in TorC (added). `setup.kaiming_digit` = {784, 0.0505} from
  `torc::kaiming_normal_std(784)` (added); `nn::Linear` with `init_std = 0` uses `fill_kaiming_normal`, std √(2/fan_in)
  (`include/torc/utils.hpp`, `src/nn/linear.cpp`).
- `init[]`: std after layer 8: Kaiming 1.2, too small (0.05) 3.1e−6, too large (0.5) 313, `no_two` (√(1/32) = 0.177, added) 0.076,
  falling at every layer. Share of zeros after each ReLU 0.40 to 0.58.

## Expected mathematics
- Twins: by symmetry of the update, equal parameters receive equal gradients, so plain SGD (no noise, no momentum) never splits them.
- Kaiming (He et al. 2015): for z = Σ wᵢ aᵢ with independent zero-mean weights, Var(z) = in · Var(w) · E[a²]; after ReLU of a
  symmetric z, E[a²] = Var(z)/2. Var(w) = 2/in keeps Var(z) constant from layer to layer; Var(w) = 1/in halves it at every layer
  (std × 0.71 per layer; measured 0.58 to 0.79 per layer in `no_two`).
- Sigmoid′(z) = σ(z)(1 − σ(z)) ≤ 1/4.

## Suspected discrepancy
None.

## Deviations and choices
- The reader's "spread" is the standard deviation, defined in one sentence; no refresher block (the word budget of 1,400 counts
  refreshers, and the lesson must stay short). The log axes are explained in each widget's hint.
- The brief mentions "the six-layer signal plot"; the trace has an eight-layer stack (`trace_deep.cpp`), which the page uses.
- The old `init-signal.js` / `linear-signal-data.js` and the matmul-grid "copy neuron" button are not used; the new widgets read
  `deep.json` only.
- The part review is a plain text section with three questions and links (brief §9 "Per part").
