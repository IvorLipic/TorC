# Content plan: a complete interactive course on TorC

Status: proposal for owner review. Basis: TorC `main` @ 5416a87 (read-only checkout) plus the current `web` site.
Evidence tags used below: **(V)** = verified in this planning pass by reading the cited code and/or running it against a
`libtorc` built from 5416a87 (clang++-18, -O1); **(NV)** = not verified. Behaviour claims follow CLAUDE.md's split:
*Observed* (code/test), *Expected* (math), *Suspected* (inconsistency; goes to `docs/library-audit/`, never to a lesson page).

## 1. Audience and goals

**Who.** Programmers and STEM students who know basic calculus vocabulary (slope, derivative) and some C++ or NumPy, but have
never seen how a framework turns "compute a loss" into "update the weights". Not for ML researchers; not a PyTorch tutorial.
TorC is small enough (about a dozen source files) that a learner can read the real code after each lesson, and that is the pitch.

**After each part a learner can:**
1. *Big idea* - explain why moving against the gradient lowers a loss, perform one SGD step by hand, and say what `backward()` computed.
2. *Tensors* - predict result shapes of elementwise ops, broadcasting, reductions, `matmul`; say why `transpose`/`reshape` copy in TorC.
3. *Autograd in depth* - predict what `.grad` holds after repeated `backward()` calls, use `detach`/`NoGradGuard`, and fix "graph input expired".
4. *Building blocks* - compute a `Linear` layer, activation and loss by hand; count parameters; choose an initialisation.
5. *Data and training* - write and trace the five-line training loop; explain batching/shuffling; read a loss curve.
6. *Optimizers* - write the update rules of SGD+momentum, Adam, AdamW and predict how each reacts to lr.
7. *Convolution* - compute a `conv2d` output size and one output pixel; explain `dWeight` as a sum over window positions.
8. *Capstones* - read TorC's three examples end to end; explain every shape and the parameter count of the MNIST MLP/CNN.

**Content rules.** Lesson pages teach what TorC *does* with the math beside it; no audit or bug talk (decision 4). Hazards the
learner will hit (lifetime, `zero_grad`, silent `step()` skip) appear as "Why did my X fail?" gotchas with the recorded real
message. Each lesson ends with 3 to 5 *predict, then reveal* checks whose answers come from the trace. Each lesson has an
internal note `docs/lessons/<id>.md` (storyboard, symbols, Observed/Expected/Suspected) that is not shipped.

## 2. Curriculum (v2, owner request 2026-10: follow the usual university order; supersedes the course map below)

The order follows what the common introductory courses share, adapted to what TorC has (no RNNs or attention in TorC, so the
course stops at convolution):
- **Dive into Deep Learning** (d2l.ai): preliminaries (tensors, calculus, autograd) → linear regression → classification → MLPs →
  builders' guide (layers and modules, parameters, initialisation) → CNNs → … → optimisation algorithms.
- **Stanford CS231n**: image classification → loss functions and optimisation (SGD, momentum, Adam) → neural networks and
  backpropagation → CNNs → training neural networks.
- **MIT 6.S191**: perceptron, activation functions, loss, gradient descent, backpropagation, regularisation (dropout, early
  stopping) → sequences → vision (CNNs) → generative models.
- **Karpathy, Neural Networks: Zero to Hero**: a scalar autograd engine and backpropagation first (micrograd), then tensors and
  an MLP, then activations, gradients and initialisation.
TorC Learn takes Karpathy's start (one scalar gradient before tensors) and d2l's order after it.

| # | Part | Lesson (id) | Status | Code beside it | Widgets |
|---|---|---|---|---|---|
| 1 | Learning from a gradient | The gradient (`why-gradients`) | published | `mul`, `sub`, `backward()` | the bowl, graph lab |
| 2 | | Using the gradient (`sgd-step`) | published | `optim::SGD`, `step`, `zero_grad` | loss landscape, zero_grad loop |
| 3 | Building a network | Tensors (`tensors`) | published | `Tensor`, `add` (broadcast), `matmul`, `transpose` | broadcast grid, axis explorer |
| 4 | | Layers (`linear-layer`) | published | `nn::Linear` | matrix grid, signal through six layers |
| 5 | | Nonlinearity (`nonlinearity`) | published | `relu`, `sigmoid`, `softmax` | activation plotter, fit a curve |
| 6 | | Modules (`modules`) | published | `nn::Sequential`, `parameters()` | model builder |
| 7 | Training | Losses (`losses`) | published | `nn::MSELoss`, `nn::CrossEntropyLoss` | loss explorer, gradient comparison |
| 8 | | The training loop (`training-loop`) | published | `data::DataLoader`, `optim::SGD` | batch viewer, training curves |
| 9 | | Better optimizers (`optimizers`) | published | `optim::SGD` (momentum), `Adam`, `AdamW` | optimizer race |
| 10 | | Generalization (`generalization`) | published | train/test loaders, `AdamW` weight decay | train vs test curve |
| 11 | | Digits with an MLP (`mnist-mlp`) | published | `data::MNISTDataset`, the example's model | digit pad, first-layer weights |
| 12 | Images | Convolution (`convolution`) | planned | `nn::Conv2d`, `nn::Flatten` | sliding window, feature maps |
| 13 | | Project: MNIST CNN (`mnist-cnn`) | planned | the CNN example's model | digit pad, feature maps |
| – | Labs | Backprop lab (`backprop-lab`) | published | `mul`, `add`, `backward()` on a reused value | graph lab |

Every lesson pairs the real TorC code (shown beside it on the home page; compiled and run by `tools/snippets/check.mjs`) with
at most two widgets, under the style budget of `docs/authoring.md`. Labs are playgrounds outside the lesson order and may grow.
Planned lessons are specified in the course map below (section 2, v1); their order is the table above.

## 2 (v1). Course map

Sizes: S/M/L = relative effort including trace program, tests and review (section 5). `*` = genuinely new widget. Reused:
**graph** (SVG graph), **stepper** (timeline + checkpoints), **window** (math window/paper work), **inputs** (editable leaves).
Symbols are TorC paths under `include/torc/` and `src/`.

### Part 1 - The big idea
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `why-gradients` | Slope = loss change per unit weight; stepping against it helps for small steps, fails for large; chain rule = multiplied local slopes | none | `Variable::backward`, `mul`, `sub`, `optim::SGD` (autograd.hpp, optim.hpp) | `landscape1d*`, graph, stepper, window | new trace `why`; no new engine ops | M |
| `backprop-lab` (exists) | Forward/backward by hand; reuse, chain rule, tiny net | `why-gradients` (scene 3 also wants Part 2) | `Variable`, `TapeEntry`, `Variable::backward_with_grad`, `accumulate_grad` | existing | existing trace | done |
| `sgd-step` | Do `w <- w - lr*grad` for real; zero_grad; lr too big; momentum preview | `backprop-lab` | `optim::SGD::step/zero_grad` (optim.cpp), `nn::MSELoss`, `nn::Linear` | `contour2d*`, `update-table*`, `loop-timeline*`, graph | trace `optim`; engine `transpose`,`mse`; new `optim.js` | M |

### Part 2 - Tensors
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `tensor-basics` | shape, numel, row-major flat index; `reshape`/`view` copy; TorC has no strides | none | `Tensor(shape)`, `shape()`, `numel()`, `operator[]`, private `flat_index` (tensor.cpp: `idx = idx*shape[i] + indices[i]`, V), `reshape`, `shape_product` | `memory-layout*` | trace `tensor` (fill 0..n-1, read back through `operator[]`) ; pure JS formula | S |
| `broadcasting` | Predict result shape or error of any elementwise op; know how its gradient is summed back | `tensor-basics` | `broadcast_shape` (utils.hpp), `Tensor::add/sub/mul/div` (+scalar), `reduce_sum_to_shape` (autograd.cpp) | `shape-explorer*`, graph | trace table (shapeA,shapeB)->shape or `ShapeError` text (V: "Cannot broadcast shapes (2, 3) and (2)"); engine already has `broadcastShape` | M |
| `reductions` | `sum/mean/max/min` over all or one axis; which axis collapses; argmax tie = first | `broadcasting` | `Tensor::sum/mean/max/min(axis)`, autograd `max(a,axis)` | `axis-explorer*` (shares `shape-explorer`), window | engine: `sum/mean/max/min(axis)`, `neg`, `div` | M |
| `matmul` | Shape rule `(m,k)@(k,n)`, batch broadcast, `transpose`, contiguous `slice`; grads `G Bt` and `At G` | `reductions` | `Tensor::matmul` (tile 32, i-p-j order, V), `torc::matmul/transpose/slice` | `matmul-grid*`, `shape-explorer`, graph | engine: `transpose`, `slice`; trace batch-broadcast case | M |

### Part 3 - Autograd in depth
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `grad-accumulation` | `.grad` is added to, never overwritten; `zero_grad`; second `backward()` on the same root does nothing (V) | `sgd-step` | `Variable::backward/zero_grad/has_grad`; tests `GradMapAccumulation.DiamondDependencyAccumulates` | `grad-ledger*`, stepper | trace calls log (V: x=3,y=x*x -> grad 6 -> 6; two losses sharing h -> 6 then 18) | S |
| `graph-control` | `requires_grad` propagation, `detach`, `NoGradGuard`, `backward(grad_output)` for non-scalars, guarded in-place `fill` | `grad-accumulation` | `Variable::detach/fill/NoGradGuard/grad_enabled` (thread_local); tests `DetachNoGradAutograd`, `InPlaceGuardAutograd` | graph, window | trace (V: detach case x.grad = 9); no new ops | S |
| `graph-lifetime` | "Why did `backward()` throw?" Keep tracked Variables alive; do not nest `loss(model(x), y)`; a second `model(x2)` clears the first graph | `graph-control` | `TapeEntry::input_lifetimes`, `VariableLifetime`, `Module::forward_cache_`; tests `GraphLifetime` | `scope-timeline*` | trace of 4 to 6 fixed snippets with real `what()` (V: "autograd graph input expired before backward(); keep all graph Variables alive" for both the temporary and the reused-module case) | S |
| `gradient-checking` | Finite differences as a debugging tool; why h cannot be too small in float32 | `matmul` | tests' eps=1e-4/atol=1e-2 (docs/AUTOGRAD.md) | `fd-error-plot*` | trace: error vs h sweep in float32 (the lesson's real content; JS doubles would hide it) | M |
| `ref-op-rules` (reference page, not a lesson) | One card per op: forward, backward, shape rule | - | generated from the engine op registry | window | free: renders `FORWARD_RULE/BACKWARD_RULE` | S |

### Part 4 - Building blocks
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `linear-layer` | Spec in section 3.3 | `matmul`, `sgd-step` | `nn::Linear`, `kaiming_normal_std`, `fill_kaiming_normal` | `shape-explorer`, `matmul-grid`, `init-signal*` | trace `linear` | M |
| `activations` | Why a nonlinearity (two Linears without ReLU collapse to one); ReLU/Sigmoid/Softmax(axis) values and slopes; dead ReLU | `linear-layer` | `nn::ReLU/Sigmoid/Softmax(axis)`; `Tensor::softmax(axis)` subtracts the max (V); legacy `softmax()` flattens | `activation-plotter*` | engine: `exp`,`log`,`sigmoid`,`softmax(axis)` | M |
| `losses` | MSE and cross-entropy by hand; CE gradient = (softmax - onehot)/N; targets are class indices stored as floats (V) | `activations` | `nn::MSELoss`, `nn::CrossEntropyLoss` (rank-2 logits, rank-1 or `[n,1]` targets) | `loss-explorer*` | engine: fused `mse`, `cross_entropy` (library uses custom backward closures, V) | M |
| `modules-sequential` | A model = parameters + `forward`; `Sequential`; parameter counting; identify parameters by name or shape, not position | `losses` | `Module::register_parameter/named_parameters/parameters/operator()`, `Sequential::add` | `model-builder*` (shape + param count, real error strings) | trace table (V: MLP 26 506 params in 6 tensors; CNN 9 098) | S |

### Part 5 - Data and training
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `dataloader` | Dataset vs loader; batches, last partial batch, shuffle per epoch | `modules-sequential` | `data::Dataset/TensorDataset/SyntheticRegression/CSVDataset/MNISTDataset`, `DataLoader::next_batch/has_next/reset` | `batch-viewer*` | trace: index order from a seeded run (ship it; `std::shuffle` is stdlib-specific) | M |
| `training-loop` | Trace every line of forward -> loss -> `backward` -> `step` -> `zero_grad`; epochs vs steps | `dataloader` | examples/linear_regression/linear_regression.cpp loop | `loop-timeline`, `train-playground*` (live, JS) | trace: C++ history for preset; JS port reproduces it | L |
| `linear-regression` | Whole pipeline on `y=3x+1+noise`; loss floor = noise variance | `training-loop` | `data::SyntheticRegression` (x ~ U(-1,1), y = w x + b + N(0,s), V) | `train-playground` | C++ run (V: w=2.993, b=1.012, final loss 0.00818, floor 0.01) | M |

### Part 6 - Better optimizers
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `momentum` | Velocity formula, overshoot and damping, when it helps | `training-loop` | `SGD::step` (`v = mu*v + g; p -= lr*v`, V), test `SGD.StepWithMomentumAccumulatesVelocity` | `optimizer-race*`, `contour2d` | `optim.js`; velocity is private in `SGD`, so conformance is on parameters only | M |
| `adam-adamw` | m, v, bias correction, `eps`; first step is about `lr*sign(g)` (V); decoupled decay | `momentum` | `Adam::step`, `AdamW::step` (`t = step_++`, `step_` starts at 1, V) | `optimizer-race` | `optim.js` | M |

### Part 7 - Convolution
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `conv2d-forward` | Window, stride, padding, `H_out = (H+2P-K)/S+1` (28 -> 14 -> 7, V); `Conv2d` = conv + bias via broadcast; Flatten | `matmul`, `broadcasting` | `Tensor::conv2d`, `torc::conv2d`, `nn::Conv2d`, `nn::Flatten` | `conv-window*` | engine: `conv2d` forward (naive loops), `reshape` | L |
| `conv2d-backward` | `dWeight` and `dInput` as scatter-add over window positions | `conv2d-forward` | conv2d backward closure (autograd.cpp), tests `Conv2dAutograd` | `conv-window` backward mode | engine: `conv2d` backward | M |

### Part 8 - Putting it together
| id | Outcome | Needs | Symbols | Widgets | Data / ports | Size |
|---|---|---|---|---|---|---|
| `mnist-mlp` | Read `Linear(784,32)->ReLU->Linear(32,32)->ReLU->Linear(32,10)` + `AdamW` + `CrossEntropyLoss`; classify digits in the browser | `adam-adamw`, `losses` | examples/mnist_mlp/mnist_mlp.cpp | `digit-pad*`, `weights-gallery*` (first-layer rows as 28x28 images), loss/accuracy plots | precomputed: seeded training run, weights (~100 KB), 20 test digits + C++ logits; JS forward port | L |
| `mnist-cnn` | Same data, fewer parameters (9 098 vs 26 506, V); see kernels and feature maps | `conv2d-forward`, `mnist-mlp` | examples/mnist_cnn/mnist_cnn.cpp | `digit-pad`, `weights-gallery`, feature-map viewer | precomputed weights; JS conv forward port | L |

### Part 9 - Under the hood (optional, last)
`performance` (M): worth exactly one lesson, because it explains *why* TorC's numbers look the way they do: cache-blocked `matmul`
(`Mc=Nc=Kc=32`, V), 8-lane AVX2 inner loop guarded by `TORC_HAS_AVX2` (`src/simd_ops.hpp`, V), and "copy not view" (`transpose`
copies; the README comparison lists it as 96x slower than `torch.t()`, a figure that predates the latest rerun). Widgets: `loop-order*` (animated tile/loop visit order computed
from the real loop bounds) and a benchmark bar chart quoting **one dated table** of docs/BENCHMARKS.md. Do not hand-quote the
README table (it disagrees with the latest rerun, section 6). Judgement: include, but last, and only after a fresh single-machine rerun.

### New-widget notes
Build on one shared SVG helper `plot.js` (axes, curves, scatter, a sampled heat/contour grid); no charting library. Plots carry a
screen-reader-only table (decision 9; the earlier visible "show as table" toggle was removed). The load-bearing ones are `landscape1d`, `contour2d`,
`optimizer-race`, `shape-explorer` (reused by 5 lessons), `conv-window` and `train-playground`.

## 3. Detailed specs: the first three lessons

Shared setup (from tools/trace/): each lesson gets its own trace program emitting `site/data/<topic>.json` in the existing node
schema (`scenes[]`, `nodes[]`, `finite_difference`) plus topic-specific tables, under the envelope in section 4.

### 3.1 `why-gradients`

**Objectives.** (1) Read a derivative as "loss change per unit weight change". (2) Predict the direction and rough size of an
improvement step, and why a big step fails. (3) See the chain rule as multiplied local slopes, which `backward()` automates.

**Example (V).** One weight, one datum: `p = w*x`, `e = p - t`, `L = e*e` with x=2, t=6, so `L(w) = (2w-6)^2`, minimum at w=3.
Built with real `mul`, `sub`, `mul`. At w=0: p=0, e=-6, L=36, dL/dw = -24 (library). Expected: `dL/dw = 8(w-3)`; at the other
probes the library gave L/grad = 33.64/-23.2 (w=0.1), 16/-16 (w=1), 12.96/-14.4 (w=1.2), 0/0 (w=3), 36/24 (w=6), 70.56/33.6 (w=7.2).

**Storyboard.**
1. *A knob and a score.* Slider for w; live p, e, L and a dot on the L(w) curve. Predict: "turn left or right?" then reveal.
2. *Slope.* Tangent line at the dot; slider for h shows the secant slope `(L(h)-L(0))/h` with paper work: h=1 gives -20, 0.5 gives -22,
   0.1 gives -23.6, 0.01 gives -23.96, then a badge "TorC's `backward()` says -24". Units are stated: loss per unit of w.
3. *Step against the slope.* Button "step" with lr: `w <- w - lr*slope`. With lr=0.05: w=1.2, L=12.96. The tangent *promised* 36-24*1.2=7.2;
   actual 12.96 (the promise holds only locally). Table of 6 steps: w = 0, 1.2, 1.92, 2.352, 2.6112, 2.7667; L = 36, 12.96, 4.666, 1.680, 0.605, 0.218.
4. *Step size.* Presets: lr=0.1 (2.4, 2.88, 2.976...), lr=0.125 (lands on 3 in one step; flagged as special to this bowl),
   lr=0.25 (bounces 0, 6, 0, 6 with L pinned at 36), lr=0.3 (36, 70.56, 138.3, 271.1: diverges). The page shows the live factor
   `1 - 8*lr` ("distance to 3 is multiplied by this each step"): stable iff |1-8 lr| < 1, i.e. lr < 0.25 (Expected, derived; the trace confirms the five presets).
5. *Chain rule as gears.* The existing graph/stepper on scene `why`: w -> p -> e -> L. A "nudge w by delta" control shows each stage's change
   next to its local slope: x=2, 1, 2e=-12; product -24. Delta=0.01 gives p +0.02, e +0.02, L -0.2396 (about -24*0.01; difference is the delta^2 term, V). Backward step on `e*e` shows
   two contributions of -6 summing to -12, a callback to the `reuse` scene.
6. *Recap and checks.* Three sentences; bridge to `backprop-lab` ("the same thing for 1000 weights").

**Trace additions** (`tools/trace/trace_why.cpp` -> `site/data/why-gradients.json`): scene `why` (nodes w,x,t,p,e,L, constants x,t, FD check);
`grid`: w in [-1,7] step 0.25 with L and grad from real forward+backward; `secants` for h in {1,0.5,0.1,0.01,0.001}; `trajectories` for lr in
{0.05,0.1,0.125,0.25,0.3} x 8 steps using the real `optim::SGD` on a one-element Variable (V: equals manual `w -= lr*g`).
**Engine/explain:** no new ops (mul, sub exist). New: `landscape1d` widget fed by `buildGraph` so the curve and the graph share code; `explain.js` gets a
"secant to tangent" paper-work helper.

**Tests.** (a) JS `buildGraph/runBackward` equals `grid` at every point (rel 1e-5). (b) JS iteration `w -= lr*g` reproduces each trajectory (rel 1e-5).
(c) Identity `grad == 8(w-3)` on all grid points. (d) Secant slopes are monotone toward -24. (e) Stability predicate matches the sign of divergence in the five
trajectories. (f) Every number quoted in prose (36, -24, 12.96, 7.2, -12, -0.2396) is bound to a `data-trace` path and the test fails if the trace changes. (g) KaTeX parses all paper work.

**Misconceptions to address.** The gradient is a *direction and rate*, not a distance; "smaller lr is always better / bigger always faster" (both false: 0.125 is perfect here, 0.25 never converges);
the tangent line is a local promise; "the chain rule is a formula to memorise" (it is ratios multiplying); "gradient points at the minimum" (it points uphill locally; in 1-D it is just a sign).

### 3.2 `sgd-step`

**Objectives.** Perform one parameter update by hand with real numbers; say what `step()` reads and writes; explain `zero_grad`; see lr instability and the momentum formula.

**Semantics (Observed, V).** `SGD(params, lr, momentum=0)` throws `TorcError` if lr <= 0 or momentum < 0. `step()` skips any param with `!has_grad()` (silently).
Without momentum: `p = p - lr*grad`. With momentum: `v = momentum*v + grad; p = p - lr*v` (velocity starts at zero; same formula the design doc states).
`zero_grad()` calls `Variable::zero_grad` (`has_grad_=false`). `step()` does not zero gradients. Signature takes `const std::vector<Variable*>&` (design doc says non-const; doc drift).

**Example (V, identical in library and a NumPy replica).** `nn::Linear(1,1)` with w=0, b=0 set through `named_parameters()`; x=[1,2,3]^T, t=[3,5,7]^T (true line y=2x+1); `MSELoss`; lr=0.05.
Forward y=[0,0,0]; e=[-3,-5,-7]; L=(9+25+49)/3=27.667. Backward: dL/dy_i=2e_i/N=[-2,-3.333,-4.667]; gw = sum(dL/dy_i x_i) = -2-6.667-14 = -22.667; gb = -10.
Update: w=0-0.05*(-22.667)=1.1333, b=0.5; new L=5.4885 (80% drop). Next steps (w,b,L before the update): (1.6378, 0.7233, 1.0897), (1.8621, 0.8234, 0.2172), (1.9618, 0.8687, 0.0441).

**Storyboard.**
1. *The graph you already know* (graph scene `sgd`: x, W, b, Wt=transpose, xW=matmul, y=add, t, L=mse). Run Forward then Backward in the stepper; gradient badges appear on W and b.
2. *Who touches what.* Diagram: `backward()` fills `.grad`; `step()` reads `.grad`, rewrites `.data`; `zero_grad()` clears `.grad`. A tiny code panel highlights the line being explained (C++ from linear_regression.cpp).
3. *The update on paper.* Math window: rule `w <- w - eta*gbar_w`, then `0 - 0.05*(-22.667) = 1.1333`; same for b. Parameter table (before | grad | step | after) for each of W, b.
4. *Look at the landscape.* `contour2d` over (w,b), computed live from the engine; arrow shows the move from (0,0) to (1.133,0.5); the dot's loss falls from 27.67 to 5.49. Press Step repeatedly; trail and table grow (5 steps shown above).
5. *Try the learning rate.* lr slider: 0.1 reaches L=0.33 after one step (w=2.267, overshoots the true 2), 0.15 oscillates but shrinks, 0.2 diverges (27.67, 41.08, 61.01, 90.59: V). Critical lr from the Hessian: eigenvalues 11.09 and 0.240, so lr < 2/11.09 = 0.180 (Expected, derived in Python, not claimed by the library).
6. *The zero_grad trap.* `loop-timeline` with a toggle for `zero_grad()`. Without it, three iterations give gw = -22.67, -45.33, -68.00 (V: accumulation, not replacement). Also: calling `step()` before any `backward()` changes nothing and raises no error (V).
7. *Momentum preview.* mu slider (formula in the window). With lr=0.05, mu=0.5 the loss runs 27.67, 5.49, 0.174, 2.447, 1.616, 0.177, 0.058 (V against library): faster at first, then overshoot, then settling. mu=0.9 overshoots badly (18.0 at step 3). Pointer to `momentum`.

**Trace additions** (`tools/trace/trace_optim.cpp` -> `optim.json`): scene `sgd` built from real `transpose`, `matmul`, `add`, `MSELoss`, with a parity check against `nn::Linear` (output and grads equal);
`runs[]` = `{optimizer, lr, momentum, steps:[{w,b,loss,gw,gb}]}` for SGD lr {0.05, 0.1, 0.15, 0.2} and (lr 0.05, mu 0.5/0.9), 10 steps each;
`no_zero_grad` run (3 iterations); `skipped_step` record; 25 `(w,b)->loss` samples for the surface conformance test.
**Engine/explain:** new ops `transpose` (backward = transpose of grad) and fused `mse` (backward `2(x-t)/N`, from `MSELoss`); new module `optim.js` with `sgdStep(state, params, grads, {lr, momentum})` as the
library's exact formulas; `explain.js` gets `updateWindow` (rule, numbers, before/after table).

**Tests.** JS reproduces every `runs[]` row within 1e-5 relative (10 steps; if drift exceeds this, apply `Math.fround` after each op - tolerance to be measured, NV);
parity flag true; `mse` and `transpose` match the `sgd` scene gradients; accumulation test k*g; skipped-step test; stability predicate `lr < 2/lambda_max` agrees with the divergence in `runs`; prose numbers bound to the trace; KaTeX.

**Misconceptions.** The optimizer does not compute gradients; `step()` is not "learning" by itself; `zero_grad` is not optional; lr is not "how fast it is correct"; the loss need not fall monotonically with momentum; parameters are updated, inputs are not.

### 3.3 `linear-layer`

**Objectives.** From (in, out, batch) predict every shape; compute `y = x W^T + b` by hand; read row k of W as output neuron k; derive `gW`, `gb`; count parameters; explain Kaiming init.

**Semantics (Observed, V).** `Linear(in, out, init_std=0, seed=random_device)`: weight `{out,in}`, bias `{out}` (zeros). If `init_std <= 0` weights ~ N(0, sqrt(2/in)) (`kaiming_normal_std(fan_in=in)`), else N(0, init_std).
`forward(x) = add(matmul(x, transpose(weight)), bias)` (src/nn/linear.cpp), intermediates parked in `forward_cache_`. Bias `{out}` is broadcast over the batch, so `gb` is the column sum of the output gradient.
Wrong input width raises `ShapeError` "matmul inner dimensions must match, got (2, 2) and (3, 2)": the message shows `x` and the *transposed* weight (V). Rank-3 inputs: NV, do not claim.

**Example (V; module output and gradients equal a hand-built graph exactly).** x=[[1,2,3],[-1,0,2]] (2x3), W=[[0.5,-1,2],[1,0,-0.5]] (2x3), b=[0.1,-0.2]. W^T=[[0.5,1],[-1,0],[2,-0.5]]; xW^T=[[4.5,-0.5],[3.5,-2]]; y=[[4.6,-0.7],[3.6,-2.2]].
Loss MSE vs t=[[4,0],[3,-2]]: errors [[0.6,-0.7],[0.6,-0.2]], L=0.3125; dY=e/2=[[0.3,-0.35],[0.3,-0.1]]; gW=dY^T x=[[0,0.6,1.5],[-0.25,-0.7,-1.25]]; gb=[0.6,-0.45]. With loss=sum(y) and x trainable, dX rows are [1.5,-1,1.5] (column sums of W).
Counts: Linear(784,32) = 25 120 params; MLP 26 506 in 6 tensors.

**Storyboard.**
1. *Shape explorer.* Steppers for in, out, batch; diagram x{B,in}, W{out,in}, b{out}, y{B,out}; live parameter count `in*out+out`. A "break it" button shows the recorded real error above.
2. *One neuron = one row.* Hover y[i,k]: highlights row i of x and row k of W with `dot + b_k` paper work (`matmul-grid`).
3. *Three ops under the hood.* Graph `W -> Wt -> xWt -> y` plus `b`; stepper and windows. Badge: "TorC's `Linear` output equals this hand-built graph", because `Linear` hides its intermediates (the trace asserts exact equality of output and gradients).
4. *Backward.* Add `mse`; the window shows `gW = dY^T x` and why `gb` = column sums (broadcast bias; callback to `backprop-lab` scene 3).
5. *Why initialise like that?* Weight histogram (empirical std 0.0509 for 784x32 vs sqrt(2/784)=0.0505; 0.2613 vs 0.25 for 32x32, V). `init-signal`: 6 layers of Linear(256,256)+ReLU on N(0,1) input, std of activations per layer:
   Kaiming 0.83, 0.83, 0.84, 0.91, 0.85, 0.82; init_std 0.01: 0.094 ... 1.7e-6; init_std 1.0: 9.4 ... 1.7e6 (V; vanishing/exploding signal). Discrete choices only (it is a precomputed table, labelled as such).
6. *Check yourself.* Predict shapes for `Linear(784,32)`; predict `y` for edited `b`; why are weights {out,in}?

**Trace additions** (`tools/trace/trace_linear.cpp` -> `linear.json`): scene `linear` (manual graph + parity flags for output and grads); `shape_errors[]` with exact `what()` strings; `init_stats[]` for (in,out,seed) with mean/std; `signal[]` for the three inits; `param_counts[]` (MLP, CNN).
Because `std::normal_distribution` output is stdlib-specific, the JSON ships the actual weights and never asks JS to reproduce an RNG (NV across libc++/libstdc++).
**Engine/explain:** reuse `transpose`, `matmul`, `add`, `mse`; add transpose rule text; hover mode for `matmul` paper work.

**Tests.** Engine equals the `linear` scene (values and grads); parity flags true; for all `param_counts`, formula `sum(in*out+out)`; `init_stats` std within 5% of sqrt(2/in) when n >= 1000 (expected relative error about 1/sqrt(2n) < 1%);
`shape_errors` match `/inner dimensions must match/`; `dX` equals W column sums; FD check present; KaTeX for all strings.

**Misconceptions.** Weights are `{in,out}` (they are `{out,in}`); `Linear` includes an activation; zero weights are fine (identical neurons receive identical gradients; only the *bias* is zero by default); bigger init learns faster; bias is per input element; the input needs a gradient (usually not).

## 4. Site architecture for a full course

Principles kept from decisions.md: static, no framework, KaTeX vendored, tested JS port, real-run numbers. Additions:

**Pages and URLs (all relative so `/TorC/` works).** `site/index.html` (course map + progress), `site/lessons/<id>/index.html` (a ~15-line committed shell per lesson that loads `../../js/lesson-boot.js` with its id),
`site/ref/ops/` (generated op cards), `site/ref/api/` (API index). No `<base>`, no root-absolute URLs; `check_site.mjs` greps for `href="/`, `src="/`, `fetch("/`. State in the hash (`#step=3&sel=h`); the old `#reuse/3` links are redirected for `backprop-lab`.

**Course metadata.** Hand-written `site/course.json`: parts, lessons `{id,title,outcome,prereqs[],symbols[],trace[],est_minutes}`. Home page and prev/next nav render from it at runtime; the API index is built client-side from `symbols`. Generated data lives only in `site/data/`.
**Lesson format.** `site/js/lessons/<id>.js` exports `{sections:[{kind:'text'|'widget'|'gotcha'|'predict'|'check', ...}]}` where text is a function `(D) => html` over the loaded trace `D`, so Node can render every lesson with the real JSON and test it (no jsdom). Numbers in prose use `T('path')`, which throws if the path is missing; a test fails on unbound numerals marked `.num`.
**Shared modules** (extract from `app.js`/`engine.js`/`explain.js`): `core/dom.js`, `core/state.js` (hash), `core/store.js` (progress, wrapped in try/catch), `core/katex.js`; `ops/<op>.js` registry entries `{forward, backward, texForward, texBackward, why}` so the engine, explain and the op-card page are generated from one place, and a test requires *every registered op to appear in some trace scene* (no op without ground truth); `optim.js`; `widgets/*.js`; `widgets/plot.js`.
**Trace envelope.** Every JSON: `{schema:"torc-trace/1", library:{ref, compiler, flags}, generator, kind, ...}`. `tools/trace/run.sh --check` regenerates against `tools/trace/TORC_REF` and fails on drift (already decided). Tolerances: 1e-5 relative for short chains, to be measured for optimizer trajectories (C++ is float32, JS is double; AVX2/FMA may differ in the last bits).
**Progress.** `localStorage` key `torc:v1` -> `{lessonId:{visited, checks:{id:bool}, done}}`; "mark done" is manual; learner-edited inputs are not persisted by default. No accounts, no analytics. Everything works with storage blocked.
**Search.** Skip full-text search until more than 15 lessons exist. Ship instead: a filter box on the course map (title/outcome/symbol) and the API index ("where is `SGD::step` explained?"), which is also the code-reading bridge.
**Precomputed vs live.** Live (JS port, editable, badge "recomputed in your browser"): anything built from the ported ops and the `optim.js` rules. Precomputed (C++ only): MNIST training, init-signal tables, shuffles, float32 error sweeps, benchmark numbers. Every widget states which one it is.

**Recommended `web` layout**
```
CLAUDE.md  LICENSE
docs/            content-plan.md architecture.md authoring.md decisions.md
  lessons/<id>.md        internal behaviour note + storyboard + DoD checklist (not deployed)
  library-audit/
tools/
  trace/         run.sh [--check]  TORC_REF  CMakeLists.txt  trace_util.hpp
                 autograd_trace.cpp trace_why.cpp trace_optim.cpp trace_linear.cpp ...
  train/         mnist_export.cpp (seeded, exports weights/history/test digits; links torc only)
  check_site.mjs
site/
  index.html  course.json  lessons/<id>/index.html  ref/
  js/core/ ops/ widgets/ lessons/ lesson-boot.js
  css/  data/ (generated only)  vendor/katex/  test/
.github/workflows/  pages.yml  verify.yml
```
`pages.yml` currently hard-codes seven site files; replace that list by `node tools/check_site.mjs` (manifest-driven from `course.json`).
Revisit decision 1 ("no build step") only if the 15-line lesson shells start to drift; `check_site.mjs` can assert they equal the template.

## 5. Quality bar, effort, order

**Per-lesson definition of done.** (1) `docs/lessons/<id>.md` with symbols and Observed/Expected/Suspected, claims tagged V/NV. (2) Trace program + JSON + schema check; `run.sh --check` clean.
(3) Engine/explain tests vs the C++ run, FD check, KaTeX `throwOnError`, prose-number binding test. (4) `node --test site/test/` green; browser check at 360, 768 and 1280 px and under a `/TorC/` prefix server.
(5) Accessibility: full keyboard path, focus kept on re-render, `aria-live` for step text, `prefers-reduced-motion` honoured, AA contrast, no colour-only encoding, table view for every plot.
(6) Independent read-only review (CLAUDE.md method step 5) plus one outside reader doing the lesson cold and answering the checks. (7) No audit talk on the page. Report exactly what ran and what did not.

**Effort** (S = one specialist session plus review; M = 2 to 3; L = 4 or more; section 2 gives each lesson's size). Platform extraction is L. Roughly: Part 1 new = 2M; Part 2 = 1S+3M; Part 3 = 3S+1M; Part 4 = 3M+1S; Part 5 = 1M+1L+1M; Part 6 = 2M; Part 7 = 1L+1M; Part 8 = 2L; Part 9 = 1M.

**Build order and milestones**
- **M0 Platform (L).** Extract core modules, `course.json`, home page, lesson shell, `ops/` registry, trace envelope, CI check. Gate: today's lab runs unchanged as `backprop-lab` under `/TorC/lessons/backprop-lab/`, old tests green.
- **M1 Vertical slice (owner's three).** `why-gradients`, `sgd-step`, `linear-layer`. Builds `landscape1d`, `contour2d`, `optim.js`, `shape-explorer`, `transpose`, `mse`. Gate: review + outside reader; this slice fixes the lesson format for everything after.
- **M2 Foundations.** Part 2 and Part 3 (they unblock everything else and are mostly S/M).
- **M3 Training core.** `activations`, `losses`, `modules-sequential`, `dataloader`, `training-loop`, `linear-regression`.
- **M4 Depth.** `momentum`, `adam-adamw`, `conv2d-forward`, `conv2d-backward`.
- **M5 Capstones.** `mnist-mlp`, `mnist-cnn` (needs the MNIST run, risk 3).
- **M6 Optional.** `performance`, polish, light theme.
A reasonable "v1.0" is M0 to M3 (about 20 pages); M4 to M6 ship as they finish.

## 6. Risks and open questions for the owner (max 8)

1. **Order: concept-first or tensors-first?** Recommend concept-first as mapped (why -> lab -> SGD, then tensors), with scene 3 of `backprop-lab` marked "easier after Part 2".
2. **Port scope.** Porting every op to JS multiplies tests. Recommend porting only what editable widgets need (list in section 2), each with a C++ ground-truth scene, and keeping MNIST, init tables, shuffles and float32 sweeps precomputed.
3. **MNIST run.** No MNIST CSVs are bundled; examples use `random_device` seeds, so committed CSVs are not reproducible; the library has no state-dict API, so export must read `named_parameters()` from layers kept as raw pointers before `Sequential::add`. Recommend: you run `tools/train` once locally with fixed seeds, commit weights (<= 150 KB each) and 20 test digits; accuracy quoted only from that run (the committed CSVs give per-class means of about 0.90 for the MLP and 0.94 for the CNN; overall accuracy is NV). MNIST licence/attribution: NV, check before shipping images.
4. **Hand-drawn digits.** Canvas drawings differ from MNIST preprocessing; the model will mis-predict. Recommend centring by centre of mass, labelling the demo "approximate", and making the 20 real test digits the primary demo.
5. **Library changes the site would like.** Stable `parameters()` order (now an `unordered_map` iteration, V), seedable `DataLoader` default, named access to `Sequential` children. Recommend none now: work around, list in `docs/library-audit/`.
6. **Performance lesson.** README's benchmark table and docs/BENCHMARKS.md "latest rerun" disagree (for example transpose 256x256: 424 us vs 1253 us; matmul 256: 4283 vs 5166 us). Recommend including it last, after one fresh single-machine rerun, quoting one dated table.
7. **Library is still moving** (ROADMAP: LayerNorm, embeddings, attention, optimizer validation pending). Recommend pinning `TORC_REF`, stamping each lesson "numbers from TorC @ref", and adding lessons only for merged features; no "coming soon" placeholders on the public map.
8. **Theme, language, analytics.** Recommend dark-first now, light theme in M6, English only, no analytics (nothing to consent to, nothing to maintain).

**Doc drift noticed on `main` (no action taken; report only):** `docs/DESIGN.md` gives `SGD(std::vector<Variable*>&, ...)` but `include/torc/optim.hpp` takes `const std::vector<Variable*>&`; ROADMAP step 5.13 is unchecked although `nn::Conv2d`, `Flatten` and the CNN example exist (AGENTS.md says implemented); `docs/AUTOGRAD.md` repeats a sentence fragment in the move-construction paragraph.
