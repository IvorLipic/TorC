# Lesson `backprop`: How the computer finds every gradient at once

Part 3 `networks`, curriculum v3 (`template: 3`, `loop: 'adjust'`). Brief §6 "backprop": four parts, no more.
Module `site/js/lessons/backprop.js`, graph-lab layout `site/js/lessons/backprop.scenes.js`, widgets
`site/js/widgets/backprop-cost.js` (the nudge counter) and `site/js/widgets/blame-net.js` (the small network), tests
`site/test/backprop.test.mjs`. Data: `data/autograd_trace.json` (scenes `reuse` and `net`, `net_counts`, `reuse_uses`,
`accumulation`), `data/why-gradients.json` (scene `why`, `nudge`) and `data/backprop-timing.json` (a timing, not a trace).

## What the page teaches
1. **The cost of nudging.** A nudge is one forward pass, so nudging needs one forward pass at the start and one per parameter
   (26,506 + 1 for the digit network, anchor `mlp_parameters`). Backpropagation needs one forward and one backward pass; in the
   TorC timing they cost as much as 7 to 9 nudges together, for networks up to 25 times the digit network.
2. **The chain rule** on the notebook model L = (2w − 6)², with the miss defined as prediction minus price (−6 at w = 0), from the
   three nudge ratios of `nudge.stages` (2, 1, about −12 = twice the miss), whose product is the gradient −24. Gears analogy with
   its limit. C7 fixed with the product rule the reader can check: in a product each factor gets the other factor times the
   gradient that arrives, so each use of the miss gets −6 and TorC adds them to −12. Each link's ratio is computed once and reused
   by every parameter behind it.
3. **One value, several uses**: the scene `reuse` of the Backprop lab, embedded (one graph lab per page), with a legend for the
   panel's x̄, c₁ to c₃ and "seed". Why TorC adds: a nudge of the value acts through every use at once. TorC also adds across steps:
   after a NEW forward pass, `L.backward()` adds to the stored gradient (the reason for `zero_grad`, taught in `training-loop`).
4. **The small network** (2 inputs, 2 neurons with ReLU, 1 output, one sample): a neuron's gradient is the prediction's gradient
   times its output weight times the slope of its ReLU (one here); a weight's gradient is its neuron's gradient times its input.
   The transfer quiz tests the new case of slope zero. ∂ and the matrix rule of a layer are in the In TorC panel.

Misconception refuted with the widget's evidence: "every weight gets an equal share of the blame" (the four first-layer weights get
−2.25, 3, −4.5 and 6).

## Observed (traced or executed)
- Scene `net` (`tools/trace/autograd_trace.cpp`): x = (1, 2), W1 = ((0.5, −0.5), (0.25, 0.5)) with W1[i][j] from input i to
  neuron j (`x @ W1`), b1 = (0.5, 0), W2 = (1.5, −2), b2 = 0, t = 2. Forward: h = (1.5, 0.5) (both neurons on, far from the ReLU
  kink), prediction 1.25, miss −0.75, loss 0.5625. Backward (real `Variable::backward`): prediction −1.5, neurons (−2.25, 3),
  W1 (−2.25, 3, −4.5, 6), b1 (−2.25, 3), W2 (−2.25, −0.75), b2 −1.5. No quoted gradient equals ± the miss (tested). The trace's
  finite-difference check agrees for every leaf. `net_counts.parameters` = 9.
- Scene `reuse`: x = 3, m = 9, y = 12; x stores 7. `reuse_uses`: with one leaf per use (y = a·b + c, a = b = c = 3) TorC gives
  3, 3 and 1, so each use's own gradient is a TorC number.
- Scene `why` and `nudge`: local slopes 2, 1, −12 (ratios of node gradients from the real backward pass), product −24;
  e = −6 (= p − t), e.grad = −12; in `L = mul(e, e)` each input slot receives the other factor (−6) times the arriving 1.
- **Accumulation** (`accumulation`, L = w·w at w = 3): after `L.backward()` the gradient is 6; a second `L.backward()` on the SAME
  loss leaves it at 6, because `Variable::backward_with_grad` ends with `tape_.clear()` on the root (`include/torc/autograd.hpp`);
  a NEW forward pass and its `backward()` make it 12 (`accumulate_grad` adds once `has_grad_` is set); `zero_grad()` clears it.
  Within one pass, `backward_with_grad` calls `accumulate_grad` once per input slot of every tape entry, so every use adds.
- `nn::Linear::forward` (`src/nn/linear.cpp`) computes `matmul(x, transpose(W))` plus the bias; `matmul`'s backward
  (`src/autograd.cpp`) computes `grad_output · bᵀ` and `aᵀ · grad_output`, and `transpose`'s backward transposes back, so
  ∂L/∂W = (∂L/∂y)ᵀ x, ∂L/∂x = (∂L/∂y) W, ∂L/∂b = the column sums of ∂L/∂y.
- Transfer quiz: with the input (−2, 0) the first neuron's sum is −0.5, and the engine (checked against TorC on every scene) gives
  zero gradient to W1[0][0], W1[1][0] and b1[0].

### The cost of a backward pass
The page says: "In one TorC timing on one computer, a forward and a backward pass together cost as much as 7 to 9 nudges. That
held for networks up to 25 times the size of the digit network." The numbers are bound to `data/backprop-timing.json`, written by
`tools/trace/time_backprop.cpp` (built by the trace CMake project; NOT run by `run.sh`, and `run.sh --check` only accepts that the
file exists, because a timing depends on the machine; see `docs/trace-schema.md`). Regenerate by hand on a quiet machine:
`cmake --build build/trace --target time_backprop && build/trace/time_backprop > site/data/backprop-timing.json`.

Committed run (the agent's cloud machine, 2026-10-31, clang 18 `-O3`, one sample, best of 5 rounds of about 0.25 s):

| parameters | forward, record on | forward, `NoGradGuard` (a nudge) | forward + backward | backprop in nudges | nudging / backprop |
|---|---|---|---|---|---|
| 26,506 | 110 µs | 74 µs | 471 µs | 6.5 | about 4,100 |
| 118,282 | 747 µs | 297 µs | 2,502 µs | 8.6 | about 13,700 |
| 669,706 | 4,145 µs | 2,137 µs | 16,613 µs | 7.8 | about 86,000 |

Counted against a forward pass with the gradient record on, the backward pass alone cost 2.3 to 3.3 forward passes; counted against
a nudge's forward pass (no record), 5.0 to 6.2. The page uses the comparison that matters for the argument: one forward plus one
backward pass, in nudges. The counter widget draws that cost from the file (`backprop_over_nudge`), at the same scale as the nudges.
Values move from run to run by a few percent; the page rounds to one significant figure.

The general statement (reverse-mode differentiation gives the whole gradient at a cost of a small constant multiple of one
evaluation) is the "cheap gradient principle" of A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008. The page
cites it in the footer as a concept source and quotes no bound from it. **Verify before quoting**: the agent's session had no
network access to the book (or to Baydin et al., JMLR 18, 2018), so no page number or constant was checked.

## Expected mathematics
- Chain rule on L = (2w − 6)²: dL/dw = 2 · 1 · 2(2w − 6) = −24 at w = 0.
- y = x² + x: dy/dx = 2x + 1 = 7 at x = 3 (sum rule; equals the sum of the three contributions).
- Small network: ∂L/∂o = 2(o − t); ∂L/∂h_j = ∂L/∂o · W2_j · relu′(h_j); ∂L/∂W1_ij = x_i ∂L/∂h_j; ∂L/∂W2_j = a_j ∂L/∂o.

## Suspected discrepancy
None. (The first version of this page said that a second `L.backward()` adds to the first; that is false in TorC and was fixed.)

## Deviations and choices
- The miss is the prediction minus the target, named in words; `e` is Euler's number from `nonlinearity` on.
- Neurons, never units (authoring.md). The batch scene `mlp` is not used, because "batch" comes only in `training-loop`.
- The counter shows the three timed networks (digit network, hidden layers of 128 and of 512), not the notebook model, because the
  timing covers only those and a tiny model would make the comparison misleading; when a block does not fit, one pixel stands for
  several passes and the head says how many. The small network is not a preset, so the counter does not answer the cumulative quiz.
- Brief D16 is fixed in `graph-lab.js`, `stepper.js` and `math-window.js`: a `narrow` drawing on a phone; a sideways-scroll line
  ("Swipe" on a coarse pointer, "Scroll" otherwise); a short timeline wraps, a long one scrolls with an edge fade; "Tap Next" on
  touch-only devices.
