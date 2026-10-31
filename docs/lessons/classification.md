# Lesson note: `classification` (internal, not shipped)

Behaviour note for the lesson author. Library = TorC `38785db` (`main` = v0.1.0 + PR #5 + PR #6).
Data: `site/data/classification.json` (producer `tools/trace/trace_classification.cpp`; schema in `docs/trace-schema.md`). Review
round 1 added three derived fields: `best_line` (140 of 200 training points, 70 %, by an exhaustive search; the predict's reveal),
`vote` (the worked vote of the 4-neuron model at (0, 0) and (1.4, 0)) and `typical_right_prob` per checkpoint (e^(−train_loss)).
Naming (lead): "neuron" for a hidden unit, "probability" for the output of softmax. Tags: **V** = verified (by the real library run, or by `site/test/classification.test.mjs` on the recorded
data), **NV** = not verified.

## 1. Library symbols the lesson touches

| Symbol | Where | What it does here |
|---|---|---|
| `nn::Sequential`, `add(std::unique_ptr<Module>)` | `include/torc/nn.hpp` | runs its modules in order; the real-line snippet (and `course.json` `code`) builds `Linear(2, 8) -> ReLU -> Linear(8, 2)`; `tools/snippets/check.mjs` compiles it and checks 42 parameters and a `(3, 2)` output (V) |
| `nn::Linear(in, out, init_std = 0, seed)` | `include/torc/nn/linear.hpp:11` | weight `(out, in)`, bias `(out)`; `init_std 0` is the Kaiming default, the bias starts at 0 (so every unit line passes through the origin before training, V in the data) |
| `nn::ReLU` | `include/torc/nn/activations.hpp` | `max(0, z)`: a unit is "off" where `w1·x + w2·y + b <= 0` |
| `nn::CrossEntropyLoss` | `src/nn/losses.cpp:63` | log-sum-exp minus the target logit, mean over the rows. The page calls it "the mean surprise at the right colour" and does not name it (named in `losses`) |
| `torc::softmax(a, 1)`, `torc::sigmoid`, `torc::sub` | `src/autograd.cpp` | `prob1` is column 1 of softmax of the two scores; `sigmoid_identity` shows softmax[1] = sigmoid(s1 − s0) for six pairs, `abs_diff` 0 (V) |
| `optim::Adam` (lr 0.05, full batch, 1,000 steps) | `src/optim.cpp` | the training; the page does not name the optimizer (Adam is taught in `optimizers`) |

## 2. Observed (from the committed trace, checked by the test)

- **Data.** 200 training and 200 test points, label `i % 2`. Orange (class 1) radius ≤ 0.9, blue (class 0) radius in [1.1, 1.8].
  The blue angles leave no gap of 45°, so any half-plane that holds the whole disc holds blue points (V).
- **Best straight line** (`best_line`, derived): 140 of 200 training points (70 %); the test repeats the search in JS (V).
- **0 hidden neurons** (`Linear(2, 2)`, 6 parameters). Train accuracy 0.495 → 0.505 → 0.525 → 0.525 (step 0, 25, 100, 1000);
  `T.human pct` shows 53 %. At step 1000 every grid probability is in (0.35, 0.65): "the shading fades". The 0.5 level of the
  recorded grid lies on `derived.decision_line` at every checkpoint, and `prob1 = sigmoid(w1 x + w2 y + b)` of that line (V).
- **2 hidden neurons** (12 parameters). Train mistakes per 100: 23 at step 100 and at step 1000 (the `longer` quiz). The orange
  region reaches the edge of the plane at both checkpoints (an open band in the screenshot) (V).
- **4 hidden neurons** (22 parameters, the cumulative quiz). At step 1000: train accuracy 1, every neuron's bias < 0 (all off at
  the centre), every neuron's output weight favours blue (`W2[1][k] − W2[0][k] < 0`), the bias difference (14.58) favours orange.
  The four neuron lines lie 0.47 to 0.62 from the centre, so they **cross inside the disc** (radius 0.9); the 50 % boundary lies
  further out (on the x axis beyond ±0.9), where the votes of the neurons that are on outweigh the bias (V). Worked vote at (1.4, 0):
  neurons 1 and 4 are on (outputs 0.658, 2.461), votes −5.003 and −24.05, total −14.47 = the library's score difference (V).
- **8 hidden neurons** (42 parameters). The typical probability of the right colour (`typical_right_prob`, the probability whose
  surprise is the mean surprise) is 0.48 before training and 0.9998 after 1000 steps (never shown as 1 or 100 %). The 0.5 level never reaches the edge (closed). Unit 6 (index 5) is on for all 200 points at the end, unit 4
  (index 3) for 11: not every unit is a "ring detector"; the page claims the vote picture only for the 4-unit model.
- **New points (C2).** Only `models[3].checkpoints[3].test_correct` (8 units, end) is bound on the page: 200/200 = 100 %. The four
  models are compared on training numbers only, and the text says so. One run per model (seeds `100 + H`, `200 + H`), labelled (C5).

## 3. Expected mathematics

- Two scores `s0, s1`: softmax gives `p1 = e^s1 / (e^s0 + e^s1) = 1 / (1 + e^-(s1 − s0)) = σ(s1 − s0)`. The boundary `p1 = 0.5` is
  `s1 = s0`. With no hidden layer `s1 − s0` is linear in (x, y), so the boundary is a straight line.
- With one ReLU layer, the score difference is linear inside each region cut out by the neuron lines, so the boundary is straight
  inside each region and can change direction only on a neuron line. The page says only "more neurons give more lines, so the
  boundary can bend in more places" (review round 1 dropped the stronger sentence, which confused a reader against the 2-neuron
  picture). The drawn boundary comes from straight-line interpolation on the 41 × 41 grid, so its corners are rounded.
- The trained 1-layer model is unsure everywhere (all grid probabilities in (0.35, 0.65)) and gets 53 %, below the best line's 70 %:
  training lowers the mean surprise, not the number of mistakes, and here that ends near 50 % everywhere (observed, one run).
- The loss is the mean of `−ln p(right class)`; `100·e^(−L)` (`T.human prob`) is the geometric mean of the probability of the right
  colour, which the page calls "a typical training point".

## 4. Suspected discrepancies

None found. Not verified: whether two ReLU units in the plane can ever enclose a bounded orange region in general; the page only
states what this run shows (an open band).

## 5. Choices

- Title "Two classes and the decision boundary": the brief's "Two classes: decision boundaries" has a colon, which the prose linter
  rejects in a heading.
- "logit" is not used on this page (the page says "score"); the glossary entry `logit` now names `losses` as its lesson.
- Palette: the analogy "neuron as a weighted vote" only, with the limit that fits this lesson (a vote that grows with the distance
  from the neuron's line, not a yes/no vote). The `probability` refresher was cut for the 1,400-word budget (it counts
  toward the total).
