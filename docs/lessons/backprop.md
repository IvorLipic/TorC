# Lesson `backprop`: How the computer finds every gradient at once

Part 3 `networks`, curriculum v3 (`template: 3`, `loop: 'adjust'`). Brief §6 "backprop": four parts, no more.
Module `site/js/lessons/backprop.js`, graph-lab layout `site/js/lessons/backprop.scenes.js`, widgets
`site/js/widgets/backprop-cost.js` (the nudge counter) and `site/js/widgets/blame-net.js` (the small network), tests
`site/test/backprop.test.mjs`. Traces: `data/autograd_trace.json` (scenes `reuse` and `net`, `net_counts`) and
`data/why-gradients.json` (scene `why`, `nudge`).

## What the page teaches
1. **The cost of nudging.** A nudge is one forward pass, so nudging needs one forward pass at the start and one per parameter
   (26,506 + 1 for the digit network, anchor `mlp_parameters`). Backpropagation needs one forward and one backward pass.
2. **The chain rule** on the notebook model, from the three nudge ratios of `nudge.stages` (2, 1, about −12), whose product is the
   gradient −24. Gears analogy with its limit (the ratios change with `w`). C7 is fixed: the gradient of the square of the miss is
   twice the miss (−12); TorC treats each use separately (each gives −6) and adds them.
3. **One value, several uses**: the scene `reuse` of the Backprop lab, embedded (one graph lab per page). Why TorC adds: a nudge
   of the value acts through every use at once. The same accumulation adds a second `L.backward()` to the first (the reason for
   `zero_grad`, which `training-loop` teaches).
4. **The small network** (2 inputs, 2 neurons with ReLU, 1 output, one sample): a neuron's gradient is the prediction's gradient
   times its output weight; a weight's gradient is its neuron's gradient times its input. ∂ is introduced here once. The matrix
   rule of a layer is in the In TorC panel, written with ∂.

Misconception refuted with the widget's evidence: "every weight gets an equal share of the blame" (the four first-layer weights get
−2.1, 1.05, −4.2 and 2.1).

## Observed (traced or executed)
- Scene `net` (new, `tools/trace/autograd_trace.cpp`): x = (1, 2), W1 = ((0.5, −0.5), (0.25, 0.5)) with W1[i][j] from input i to
  neuron j (`x @ W1`), b1 = (0.1, −0.2), W2 = (1, −0.5), b2 = 0, t = 2. Forward: h = (1.1, 0.3) (both neurons on, far from the
  ReLU kink), prediction 0.95, miss −1.05, loss 1.1025. Backward (real `Variable::backward`): prediction −2.1, neurons (−2.1, 1.05),
  W1 (−2.1, 1.05, −4.2, 2.1), b1 (−2.1, 1.05), W2 (−2.31, −0.63), b2 −2.1. The finite-difference check of the trace agrees for every
  leaf (max error 9e−6). `net_counts.parameters` = 9 (the tracked numbers of the scene, counted by the producer).
- Scene `reuse`: x = 3, m = 9, y = 12; x receives 3, 3 and 1 and stores 7.
- Scene `why` and `nudge`: local slopes 2, 1, −12 (ratios of node gradients from the real backward pass), product −24;
  e = −6, e.grad = −12.
- `L.backward()` accumulation: `Variable::accumulate_grad` (`include/torc/autograd.hpp`) adds each incoming gradient to `grad_`
  once `has_grad_` is set; `backward_with_grad` calls it once per input slot of every tape entry, so every use adds; `zero_grad()`
  clears it. `matmul`'s backward (`src/autograd.cpp`) computes `grad_output · bᵀ` and `aᵀ · grad_output`; `nn::Linear::forward`
  computes `x · Wᵀ + b`, so for a layer ∂L/∂W = (∂L/∂y)ᵀ x, ∂L/∂x = (∂L/∂y) W and ∂L/∂b = the column sums of ∂L/∂y.
- Transfer quiz: with the input (−1, 0) the first neuron's sum is −0.4, and the engine (checked against TorC on every scene) gives
  zero gradient to W1[0][0], W1[1][0] and b1[0] (`backprop.test.mjs`).

### The cost of a backward pass
The page says that one backward pass costs about as much as a few forward passes, also in a large network (content beyond the
trace files, decision 18). Support: a TorC timing at the pinned ref (38785db), clang 18 `-O3`, one sample, MLPs 784-h-h-10 with
`nn::Linear`, `nn::ReLU` and `nn::MSELoss`, best of 5 × 2,000 repetitions, on the agent's cloud machine (2026-10-30):

| parameters | forward | forward + backward | backward / forward | nudging (params + 1 forwards) / backprop |
|---|---|---|---|---|
| 26,506 | 113 µs | 518 µs | 3.6 | about 5,800 |
| 118,282 | 711 µs | 2,661 µs | 2.7 | about 32,000 |
| 669,706 | 4,159 µs | 17,889 µs | 3.3 | about 156,000 |

The ratio backward/forward stays between about 3 and 4 while the parameter count grows 25 times. The counter widget draws the
backward pass as 3.5 forward squares (`BACKWARD_CELLS`). The timing program (not part of the trace build, so not drift-checked):

```cpp
// per configuration {in, hid, out}: Sequential of Linear(in,hid) ReLU Linear(hid,hid) ReLU Linear(hid,out), seeds 1-3
// x = 0.5 everywhere (1, in), t = 0.1 everywhere (1, out); R = 2000; best of 5
for (...) { Variable y = m(x); Variable L = mse(y, t); }                                       // forward
for (...) { Variable y = m(x); Variable L = mse(y, t); L.backward(); for (p : params) p->zero_grad(); }  // forward + backward
```

The general statement (reverse-mode differentiation gives the whole gradient at a cost of a small constant times one evaluation)
is textbook mathematics (Griewank and Walther, *Evaluating Derivatives*, 2nd ed., SIAM 2008, "cheap gradient principle"; Baydin et
al., "Automatic differentiation in machine learning: a survey", JMLR 18, 2018). Not verified here: the session had no network
access to these sources, so the page cites only the timing above.

## Expected mathematics
- Chain rule on L = (2w − 6)²: dL/dw = 2 · 1 · 2(2w − 6) = −24 at w = 0.
- y = x² + x: dy/dx = 2x + 1 = 7 at x = 3 (sum rule; equals the sum of the three contributions).
- Small network: ∂L/∂o = 2(o − t); ∂L/∂h_j = ∂L/∂o · W2_j · relu′(h_j); ∂L/∂W1_ij = x_i ∂L/∂h_j; ∂L/∂W2_j = a_j ∂L/∂o.

## Suspected discrepancy
None. The JS engine, the trace and the finite differences agree on every value the page quotes.

## Deviations and choices
- The miss is called "the miss" in words, not `e`: the letter e is Euler's number from `nonlinearity` on, and `p` is not used for
  the prediction (one letter per thing).
- "Hidden units" are called neurons, as in `linear-layer`; the batch scene `mlp` is not used, because "batch" is introduced only in
  `training-loop`. The new scene `net` has one sample.
- Brief D16 (graph labs at 390 px) is fixed in `graph-lab.js` and `math-window.js`: a layout may carry a `narrow` drawing (the
  `reuse` layout here has one, so the whole graph fits a phone); other graphs show a "swipe sideways" line; the chip row gets its own
  scrolling row with a fade on the side that has more chips; the start hint says "Tap Next" on a touch-only device.
