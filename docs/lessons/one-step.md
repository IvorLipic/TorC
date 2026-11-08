# Lesson `one-step` — internal note

Not shipped. Library symbols, and what is observed, expected and suspected.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
`nn::Linear::forward` (`x·Wᵀ + b`, `src/nn/linear.cpp`), `relu`, `nn::CrossEntropyLoss::forward` (mean of log-sum-exp minus the
true score; backward gives softmax minus one-hot, divided by the batch size, here 1; `src/nn/losses.cpp`), `Variable::backward`,
`optim::SGD::step` with momentum 0: `param->data() = param->data().sub(grad.mul(lr_))` for each parameter tensor that has a
gradient (`src/optim.cpp`). A parameter entry whose gradient is exactly zero does not change.

## Observed (tools/trace/trace_one_step.cpp → site/data/one-step.json; weights from mnist-mlp.json)
- Training image 24, a one (the first training image read correctly with a probability of the true digit in [0.5, 0.9)).
- Forward: the one gets 0.6065, the three (the rival) 0.3154; loss 0.5001 = −ln 0.6065. 9 of the 32 first-layer neurons and
  10 of the second-layer neurons are off.
- Backward: score gradients are probability minus one for the one (−0.3935) and the probability for the others (0.3154 for the
  three). Output weight from second-layer neuron 1 (value 6.714, the most active) to the one: 6.714 × (−0.3935) = −2.642; to the
  three: 2.117. First-layer weight from pixel 128 (row 4, column 16, value 254/255 = 0.9961) to neuron 15 (gradient 0.7391):
  0.7362. Bias of the one: −0.3935.
- Step (lr 0.001, our choice): `lr_times_grad` is the step size (the schema explains why `change` differs from it by float
  spacing); the weight to the one 0.5099 → 0.5125; to the three 0.4185 → 0.4164. Every update is bit-exact
  (`checks.every_update_exact`). 3,937 of 26,506 parameters change (15 %): weights from black pixels and through neurons that ReLU
  turned off get the gradient zero (the test checks the upper bound from the zero pattern).
- After: the one 0.7985, the three 0.1561, loss 0.2250. Scores after the step (`after.scores`): the one rises by 0.631, the three falls by 0.347, and the wrong scores of 4, 6 and 7 rise a little; every wrong probability falls. 41 parameters with a nonzero gradient do not change, because lr × gradient is below their float32 spacing (`changed_detail`).
- Chain rule (`chain`): second-layer neuron 1 gets 0.5099 × (−0.3935) + 0.4185 × 0.3154 + the rest = −0.0724; first-layer neuron 15 gets 0.2858 + 0.1494 + 0.1354 (from second-layer neurons 23, 20, 0) + 0.1685 for the rest = 0.7391. The producer checks each sum against the recorded gradient.
- The real run: 9,380 AdamW steps of 128 images (`full_run`).

## Expected (mathematics)
Score gradient of cross-entropy for one sample: softmax minus one-hot. A weight's gradient is its input times the gradient at the
output it feeds (chain rule). SGD without momentum: new = old − lr × gradient. The JS port of the forward pass in the test matches
the recorded hidden values, scores and probabilities within 1e-4.

## Suspected discrepancy
None.

## Deviations from the brief
- The brief says "one test digit"; the trace uses a training image, because a training step on a test image contradicts C2.
- The brief says "the gradient of one chosen pixel weight, traced back along its path"; the page traces it back through the first-layer
  neuron (section 3, the three largest parts plus the rest) and also shows two output weights; the learner computes the new value of
  the rival weight (gradient and step).
- The step uses plain SGD with lr 0.001 and a batch of one, not the AdamW of the real run; the page says so in the setup.

## Lesson status
New 2026-11, `template: 3`; revised after the independent review, the pedagogy review and the cold read (H2-H4 and the medium items). Widget: `site/js/widgets/one-step-widgets.js` (eleven stages). Tests: `site/test/one-step.test.mjs`.
Not done: a second review round, human think-aloud, screen reader.
