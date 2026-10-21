# TorC Learn: pedagogical review and rework brief

**For:** the next session that works on the `web` branch (TorC Learn, https://ivorlipic.github.io/TorC/).
**Reviewed:** `origin/web` @ `3b06118` (13 published lessons and the Backprop lab), 2026-10-21.
**Owner's question:** How good is this material *for learning*? The target reader has just finished high school and wants to
understand how ML and deep learning work. The site should be the best resource for that reader.
**Status:** v3. Revised over two rounds of independent review: a cold read by a simulated target learner, a learning-design critique
(twice) and a fact check of every claim against the `web` checkout (twice). All accepted corrections are folded in.

---

## 0. Read this first

**Where this file lives.** This brief was written on a branch cut from `main`. The `web` branch must never merge `main`
(`CLAUDE.md` on `web`). So **copy this file into `web` as `docs/pedagogy-review.md`** in your first commit, and refer to it there.

**Rules that still apply.** `CLAUDE.md`, `docs/authoring.md`, `docs/decisions.md` and `docs/architecture.md` on `web`. This brief
asks to change several of them. Do not change them silently. Record each change as a new numbered decision that cites this
brief, and get the owner's approval where §12 says so.

**Deploys are live.** `.github/workflows/pages.yml` deploys `site/` on every push to `web`. It runs only `check_site` and the
Node tests. The prose linter, trace-drift check and snippet check run in `verify.yml` in parallel, so a commit can go live before
those checks fail. Two safe ways to work:
- **Preferred:** ask the owner for a work branch (for example `web-v3`). Merge it to `web` once per finished part.
- **If not approved:** every *new* lesson stays `"status": "draft"` (reachable by URL, not listed) until its part is done.
  A *rewrite* of a published id cannot be hidden this way, because `lessons/<id>/` always serves the current module. Build the
  rewrite under a temporary draft id (for example `why-gradients-v3`), and swap the module into the real id in one commit when its
  part is done. Then switch the part to `published` and renumber `course.json` in that same commit.

**Refer to lessons by id, never by number.** This brief proposes new numbering, so numbers are ambiguous. Ids are stable.

**Ids of the recommendations.** R1–R12 are design changes, D1–D20 are defects, C1–C10 are conceptual errors in lesson
content. Cite them in commits.

---

## 1. Verdict

### What is excellent and must stay
- **The correctness pipeline.** Every number comes from a real run of TorC, and tests bind prose numbers to trace data. The JS
  engine is checked against the C++ output, KaTeX is validated, and accessibility is audited (205 Node tests pass). Almost no
  learning site has this. It is the site's unique selling point: *real library, real numbers*.
- **Short, user-paced sections**, each with a heading that names its topic.
- **Predict-then-reveal prompts in every lesson.** The best of them are among the best questions anywhere, for example
  `nonlinearity` "what does the model without ReLU learn?", `tensors` "(2) + (2, 3)?" and `convolution` "parameters of `Conv2d(1, 8, 3)`?".
- **Strong widgets.** The bowl with tangent and lr presets. The matmul grid with "paper work". The "copy neuron 1 into neuron 2"
  button. The with/without-ReLU curve fit. The optimizer race. The digit pad, which runs a real forward pass.
- **Moments that click** for the target learner (from the cold read): the bowl as a parabola, "if you increase `w` by a small
  amount, `L` changes by about −24 times that amount", the five-line training loop, "the loss does not go to zero because the data
  has noise", "Adam divides the average gradient by its typical size", the sliding-window arithmetic, and the digit pad.

### What keeps it from being the best resource for this learner
The site is optimized for **verifiable correctness and lintable style**, not for **understanding**. Every quality gate checks
something other than "did the learner understand it": trace binding, the STE linter, the word budget, the independent reviewer.
The result reads like a precise, well-tested tour of the TorC API with the maths beside it.

1. **It is aimed at the wrong reader.** `docs/content-plan.md` §1 targets "programmers and STEM students who know basic calculus
   vocabulary (slope, derivative) and some C++ or NumPy". The target reader has no C++, no matrices, no partial derivatives and
   little Σ. The home page leads with C++ cards. Lessons use `Variable w(Tensor({0.0f}, {1}), true);`, `std::make_unique` and
   `param->data() = …` in the main text.
2. **There is no "whole game" and no "why".** No page says what machine learning is, or what a model, a prediction, a target or a
   loss is in general. "Loss" is defined properly only in `losses` (lesson 7): "A loss turns the output of a model into one number
   that says how wrong the model is." The cold reader wrote: *"This is the sentence I needed in lesson 1."* "Sample" and "dataset"
   are defined only in `training-loop`.
3. **Numbers are narrated instead of ideas explained** (the owner's observation, confirmed). 53 sentences carry two or more
   bound numbers, about 4 per lesson, and 15 carry three or more. Most lessons are near the 25-number budget (`sgd-step` 25,
   `losses` 24, `why-gradients` 23, `nonlinearity` 23, `optimizers` 20). The cold reader confirms the owner. "36 to 12.96",
   "27.67", "5.489", "1442", "1.815" and "0.00008427" were "just numbers scrolling by". The momentum numbers in `sgd-step` actively
   taught a wrong lesson: *"momentum makes the loss go up, so it's risky"*. Numbers *did* help when the learner could check them or
   when they made a point (§4).
4. **Several lessons teach wrong conclusions** through unfair comparisons, peeking at the test set, and demos that support the
   misconception they should refute (§5, C1–C10).
5. **The core of deep learning is not on the main path.**
   - Backpropagation is an optional lab "not part of the lesson order". `why-gradients` §4 states the chain rule.
     `linear-layer` §3 states the matrix rules for the layer's backward pass but does not derive them.
   - No page says *why* backprop exists. Nudging each of 26 506 weights one at a time would cost 26 506 extra forward passes.
     One backward pass costs a small multiple of one forward pass (roughly two to three times), whatever the number of weights.
   - Classification is never shown visually: no decision boundary anywhere.
   - The gradient is never called "the slope" in the text (only the bowl widget's readout says "slope"), and its derivative
     `8(w − 3)` is never written.
6. **TorC engineering sits on lesson pages.**
   - `modules` is half API detail: `parameters()` order from an `unordered_map`, and the forward-cache lifetime rule.
   - Other lessons carry more of the same:
     - `reshape` copies;
     - `transpose({1, 0})` argument order;
     - "TorC does not flip the kernel";
     - "`conv2d` is a plain loop";
     - `step()` skips parameters without a gradient;
     - error strings.
   - `losses` §1 shows an audit finding outright, the MSE shape issue from `docs/library-audit/v0.1.0-findings.md`. That directly
     breaks decision 4.
7. **It assumes notation it never teaches.** The cold reader stalled on all of these:
   - ←, Σ with index subscripts (`y_{ik} = Σ_j x_{ij} W_{kj}`), ∂, the overline gradient, ᵀ, ⌊ ⌋, hats and Greek letters;
   - "log" (school "log" is base 10, so the reader got 0.477 instead of ln 3 = 1.099);
   - "curvature", "variance", "standard deviation", "regression", "channels", "logit", "overflow", "MNIST".
   Two kinds of plot are never explained either: contour maps and log axes.
8. **Checks often test recall or look-up.**
   - Some predictions give away their own answer, because the sentence just before says it or the heading names it.
   - Some checks cannot be answered from what was taught: `linear-layer` x̄; the `sgd-step` c-zero check needs a gradient the page
     never states; the `losses` log base.
   - One check is a pure number look-up: `mnist-cnn` "which digit is worst?".
   - "I got it" is a self-report, and no check revisits an earlier lesson.
9. **It is short and it stops abruptly.**
   - 13 lessons × 6–10 min = 96 min for "from a gradient to a CNN".
   - The last page ends: "This is the last lesson of the course." Nothing connects to the modern models (ChatGPT) that motivate
     this reader. The same loop of loss, gradient and optimizer is what pretrains them.

**Bottom line.** The engineering foundation is a real advantage: keep all of it. The content layer needs a rework of audience,
order, examples and checks. Most widgets, all of the trace tooling and most of the facts can be reused.

### The twelve changes, in priority order

| # | Change | § |
|---|---|---|
| R1 | Re-target the course at the high-school graduate. State the prerequisites: "algebra and graphs; we teach the rest". | 2 |
| R2 | Apply the "numbers serve claims" rule with role-tagged bindings. Translate losses into human units. Put trajectories in widgets. | 4 |
| R3 | Fix the conceptual errors C1–C10: fair comparisons, no test-set peeking, refute misconceptions with evidence. | 5 |
| R4 | Add `what-is-learning`, the whole game, with the digit pad and a hand-fitting game. Plant train and test there. | 6 |
| R5 | Put backpropagation on the main path, opened with the cost argument. Teach the gradient as a vector. Add a 2D classification lesson. | 6 |
| R6 | The concept text must be completable without reading C++. Keep 4–5 "this is the real line" moments, put the rest in an "In TorC" panel, and move TorC engineering to an appendix track. | 6 |
| R7 | Use a new lesson template: a core plus a palette. Open with a question, put the prediction before the text that answers it, close with the answer. | 9 |
| R8 | Keep STE's structure rules. Add *specified* teacher moves and a list of banned LLM tics. Add an analogy bank with stated limits. | 8 |
| R9 | Introduce notation just in time and keep one notation guide. Add maths refreshers, a glossary, and a first-use legend for each plot type. | 8 |
| R10 | Make checks commit-first, with feedback for each wrong option. Add transfer, cumulative and self-explanation checks. Bump the progress store to v2. | 9 |
| R11 | Add the `one-step` capstone (one real training step followed end to end), "same thing in ChatGPT" asides and the epilogue `from-digits-to-chatgpt`. | 6 |
| R12 | Change the quality gates. Add a pedagogy reviewer and an AI cold reader (a smoke test), plus a human think-aloud protocol. Replace the budgets that reward the wrong thing. | 10 |

---

## 2. The learner (R1)

**Persona.**
- **Who:** aged 17–19, just finished high school.
- **Knows:** algebra; functions and their graphs (lines, parabolas); exponents and logs (school log = base 10); basic
  probability; maybe one term of calculus (derivative as slope, shaky); maybe a little Python.
- **Does not know:** vectors, matrices, Σ with subscripts, partial derivatives, standard deviation, C++, what a tensor is.
- **Motivation:** "AI is everywhere; how does it actually work, and roughly how does ChatGPT work?" They quit silently when lost
  or bored.

**Consequences for the whole course.**
- **State the prerequisites** on the home page and in `what-is-learning`: "You need algebra and graphs of functions. We explain
  everything else when you need it, including derivatives and matrices."
- **Never use a symbol or term before it is introduced**, in course order. A test enforces this (§10).
- **Never ask the reader to differentiate a formula symbolically** before the backprop lesson. Get slopes by nudging: "raise `w`
  by 0.01 and the loss changes by about −0.24, so the slope is about −24". The trace already has `nudge` and `secants` data.
  Only `nudge.stages[].local_slope` is used today (`why-gradients.js:91`, `why-gradients.scenes.js:5`), and
  `site/js/widgets/landscape-model.js` already exports a tested `nudge()` (`why-gradients.test.mjs:105`).
- **C++ is depth, not the entry point** (R6). The reader must be able to finish the course without reading C++. "Real library,
  real numbers" stays visible through 4–5 short "this is the real line" moments, each with a plain-English comment:
  - the `SGD::step` update line;
  - `loss.backward()`;
  - the five-line training loop;
  - a `Sequential` model;
  - the digit model.
  Never show `Variable w(Tensor({0.0f}, {1}), true);` to a novice.
- **Meaningful data first**: "two notebooks cost 6 euros", a line through points, coloured points in 2D, handwritten digits. Keep
  abstract toys (the narrow valley) as second examples.
- **Update `docs/content-plan.md` §1** to this persona. Keep the old audience as a secondary one, served by the "In TorC" layer
  and the appendix.

---

## 3. Teaching principles (research → rule a reviewer can check)

Sources are in §13. Effect sizes are given where the meta-analyses report them, and the brief does not claim more than they say.

| Principle | What the evidence says | Rule for TorC Learn |
|---|---|---|
| Whole game first (Perkins; fast.ai) | Meeting an authentic whole version of the activity early motivates the parts | `what-is-learning` shows the digit pad and the full loop (data → model → loss → adjust) before any part. Every lesson shows a small loop diagram with its own part lit |
| Prequestions and predict-observe-explain (Richland, Kornell & Kao 2009; Carpenter & Toftness 2017; Crouch et al. 2004) | Answering before instruction improves learning of the answered material. In physics demonstrations, students who predicted first understood better than students who only watched | The prediction comes **before** any heading or text that contains its answer. The learner **commits** to an answer. At least one wrong option must look plausible |
| Productive failure (Kapur; Sinha & Kapur 2021 meta-analysis) | Attempting a rich problem before instruction helps conceptual learning when the attempt is real and well supported. The effect depends on how faithfully it is done | Use it once, deliberately: the learner fits a line by hand before the algorithm does (`what-is-learning`). Do not label every quiz "productive failure" |
| Concreteness fading (Fyfe, McNeil, Son & Goldstone 2014) | Concrete → pictorial → symbolic beats concrete-only or abstract-only for transfer | Each idea goes: a thing you drag → the same with numbers → the symbol → (optionally) the TorC code |
| Multiple representations (Ainsworth 2006) | Several representations help only if learners are shown how they map onto each other | Colour-link a symbol across the formula, the plot and the code (`w` is the same colour everywhere). Label curves directly on the plot, not in a legend under it |
| Coherence and seductive details (Mayer; Rey 2012 meta-analysis) | Interesting but unneeded material hurts learning | Remove library quirks and error strings from the concept text. Use analogies only from the curated bank, and not for every idea |
| Signalling and spatial contiguity (Mayer) | Cues that show structure, and labels next to what they label, help | Each lesson opens with "Questions this lesson answers" and closes with "The answer". Each key sentence is bold. Explain each plot type at its first use |
| Pretraining (Mayer) | Knowing the names and traits of a system's parts before the main explanation helps | Name the parts on a labelled diagram (the network picture, the loop picture), with glossary hovers. No generic "words you will meet" list on every page |
| Personalization (Ginns, Martin & Marsh 2013) | Conversational style helps modestly (about d = 0.30 for retention and 0.54 for transfer). The effect is weaker in lessons longer than 35 minutes. A 2024 replication found no gain from style alone | Write "you" for instructions and predictions. Do not expect style to rescue weak content (§8) |
| Worked examples, fading, expertise reversal (Sweller; Kalyuga) | Novices learn more from studying worked examples than from cold problem-solving. Support should fade as skill grows, and full worked examples can hurt experts | One fully worked example per lesson, then the learner does the last step. Fewer worked steps in later parts |
| Self-explanation (Bisra et al. 2018 meta-analysis, g ≈ 0.55) | Prompting learners to explain in their own words improves learning | One "explain it in one sentence" prompt per lesson, with a model answer to compare |
| Retrieval, spacing, interleaving (Dunlosky et al. 2013) | Practice testing and distributed practice are the two techniques with high utility | One check per lesson revisits an earlier lesson. Each part ends with a short mixed review that recommends, never blocks |
| Refutation texts (Schroeder & Kucera 2022 meta-analysis) | Stating a misconception, then refuting it with an explanation, changes beliefs more than stating the correct idea alone | Each lesson names one misconception and refutes it **with the widget's own evidence**, not by assertion |
| Explorables (Victor; Case) | A bare sandbox does not teach. Build each mechanic separately, then combine. The text must work without the widget | Each widget gets 1–3 guided tasks, and the text states each task's observation |

---

## 4. The "x goes to y in z steps" problem (R2)

### Diagnosis
The owner is right. Examples from the pages:

> With `lr` = 0.05, one step moves `w` from 0 to 1.2. The loss falls from 36 to 12.96. *(why-gradients)*
> With μ = 0.5, the loss is 0.174 after two steps, and 1.09 without momentum. At step three it rises to 2.447. *(sgd-step)*
> Under AdamW, it falls from 3 to 1.815 in 50 steps. *(optimizers)*
> The test loss is lowest after step 200, at 0.05669. Then it rises to 0.1281, and the training loss ends at 0.00008427. *(generalization)*

Why they fail:
1. **The learner has no reason to care.** A number means something only if the learner predicted it, computed it, or can
   compare it with something they already understand. In `sgd-step` the three data points are never even listed, so nothing
   there can be checked.
2. **They use up working memory** that a novice needs for the idea.
3. **They say *that* something happened, not *why*.**
4. **Loss values have no unit a human can feel.** Rounding 0.05669 to 0.057 does not help. "Loss 0.057" still means nothing.
5. **They are over-precise.** `fmt` prints 4 significant figures, which is the direct source of "0.00008427".

Numbers that **did** help the cold reader:
- `0 − 0.05 × (−24) = 1.2` (when the arithmetic was shown);
- the MSE worked example 0.1667;
- the learned weight 3.002 and bias 1.013 against the true 3 and 1;
- 9204 → 9666 correct (but as percentages it would have been easier).

### Root cause
`CLAUDE.md` asks that "Numbers on lesson pages come from `tools/trace` output" and "Prose that quotes a number needs a test". The
`T()` binding plus a budget of "≤ 25 numbers" made bound numbers look like proof of quality. The honesty rule is right. The budget
turned it into an incentive.

### The rule, v2 (enforceable)
1. **Every prose number has a role**, given in its binding: `T(path, {role})`. Untagged bindings fail the tests.
   - `reveal`: inside a predict or check reveal, answering the prompt.
   - `worked`: valid only inside an element marked `class="worked"` (a calculation whose inputs are all visible in that block).
     Tested. At most 3 worked blocks per lesson, so the tag cannot become an escape from the limit.
   - `setup`: a constant of the scenario (x = 2, t = 6, the lr presets), from the trace's `setup` block or a run's settings.
   - `anchor`: listed in a course-wide `site/anchors.json` of trace **paths** (never typed values, per `CLAUDE.md`), at most about
     15, each reused in at least two lessons. For example: 784 pixels, 28 × 28, 10 classes, 60 000 images, about 97 % accuracy,
     26 506 vs 9 098 parameters, the bowl's bottom at `w` = 3. 784 and "10 classes" have no trace field today: add `setup.pixels`
     and `setup.classes` to the MNIST traces.
   - `contrast`: half of a pair (A vs B) in one sentence that states the comparison in words first.
   - `human`: a loss translated into a human unit:
     - MSE → typical miss in the target's units (√MSE, "off by about 0.24 on average");
     - cross-entropy → probability given to the right answer (e^−L, "66 % on the right class");
     - accuracy → mistakes per 100.

     Derived values come from `T.human(path, 'rmse' | 'prob' | 'per100')`, which is tested. They are never typed.
2. **Formatting by role, done inside `T()`.**
   - Prose defaults to 2 significant figures. Exact values only for integers and in `worked` and `reveal` blocks.
   - One thousands separator (decide: thin space or comma).
   - Percentages for accuracy.
   - Widgets keep their full precision.
   - Leave `fmt` (`site/js/ops/texutil.js`) alone for TeX, and add a prose formatter in `makeT` (`site/js/core/trace.js`).
3. **Mechanical limits.**
   - At most 2 non-`worked` numbers per sentence.
   - No bare loss value outside `worked` or `reveal`; use `human` instead.
   - At most 10 numbers per lesson outside `worked`, `setup` and `reveal`.
   - Numbers in code blocks and widgets do not count toward the 10.
   - Apply these limits per lesson as each lesson is rewritten, using an allow-map in the test. Ten lessons fail the limit today.
4. **Reviewer heuristic.** Delete the number. If the sentence loses no meaning, leave it deleted.
5. **Close the loophole.** Number words that state a *measured result* of a run ("nineteen more passes bring it to 97 %") must be
   bound. Counting words in a setup or a comparison ("two notebooks", "ten times smaller") are fine.

### Before and after

| Before | After |
|---|---|
| "With `lr` = 0.05, one step moves `w` from 0 to 1.2. The loss falls from 36 to 12.96. Each step is shorter than the one before, because the gradient gets smaller near the lowest point." | "One step with learning rate 0.05 is new `w` = 0 − 0.05 × (−24) = 1.2. Subtracting a negative number adds, so `w` goes up toward 3. *Your turn:* at `w` = 1.2 the gradient is −14.4. Where does the next step land? *(reveal: 1.92. The step shrank from 1.2 to 0.72, because the bowl is flatter there.)*" (every digit bound, see Appendix B) |
| "Under AdamW, it falls from 3 to 1.815 in 50 steps." | "Each step keeps 99 % of the weight, because 1 − lr·λ = 1 − 0.1 × 0.1 = 0.99. After 50 steps that is 0.99⁵⁰ ≈ 0.6, so the 3 has become about 1.8." (TorC fact for the "In TorC" panel: a parameter that gets *no* gradient is skipped by `step()` and does not decay.) |
| "After the first epoch, the model reads 9204 of the test images correctly. After the last epoch, it reads 9666 correctly." | "After one pass through the training images, the network gets about 8 of every 100 new digits wrong. After 20 passes it gets about 3 wrong. The later passes remove more than half of the remaining mistakes." |
| "The test loss is lowest after step 200, at 0.05669. Then it rises to 0.1281, and the training loss ends at 0.00008427." | "The training loss keeps falling almost to zero. The validation loss falls at first, then turns and climbs. From that turn on, the curve follows the noise of its ten training points instead of the sine curve." (and C2: this curve must be validation, not test) |
| "It takes 0 steps. After the first epoch, `has_next()` returns `false`, so the inner loop does not run. TorC gives no error, and the loss stays at 0.8618." | (This check moves to the appendix: it is TorC API knowledge.) In `training-loop`, ask instead: "Why does the loader shuffle the samples again at each epoch?" |

---

## 5. Conceptual errors in the lesson content (R3)

These teach a wrong idea. Fix them in any version, before or during the redesign.

| # | Lesson | Problem | Fix |
|---|---|---|---|
| C1 | `training-loop` §3 | The batch-size prediction compares batch 10 against batch 100 *after the same number of epochs*. The answer, "batch 10 … took 300 steps", teaches "small batches are better". What the run actually shows is that more steps beat fewer steps. | Compare at equal steps *and* at equal samples seen. State the cost of each: a small batch means more steps but a noisier gradient estimate. Explain the "S" in SGD: the gradient of a batch is an *estimate* of the gradient over all the data. |
| C2 | `generalization`, `mnist-mlp` | Both say "use the test set only once, at the end". Yet both choose the stopping point or diagnose overfitting from the test curve ("The test loss is lowest after step 200"; test accuracy after every epoch). | Rename these curves "validation" and record a validation split in the trace. Otherwise say explicitly: "we look at the test set here only to show you". |
| C3 | `why-gradients` | "The gradient does not tell you how far away the lowest point is" is asserted. But in this bowl the gradient *is* proportional to the distance (8(w − 3)), so the widget supports the misconception. | Add a second bowl with the same minimum and a different steepness ("4 notebooks cost 12": L = 16(w − 3)², gradient −96 at w = 0). Same distance, four times the gradient. Needs a new trace scene. |
| C4 | `optimizers` | The race runs Adam at lr 0.2 and SGD at lr 0.019, and the predict answer is "Adam ends lowest" with no caveat. This builds "Adam is always best". | Say: "Adam got a ten times larger learning rate here. No optimizer wins on every problem." Better: show each optimizer at its own best learning rate from a small sweep. |
| C5 | all runs | Every run uses one seed, yet the prose states its results as general facts. | Say once per part: "a different random start gives slightly different numbers". For the headline comparisons (MLP vs CNN, batch sizes, optimizers), record 3 seeds and show the range. |
| C6 | `sgd-step` | Momentum appears in lesson 2 with numbers that rise ("at step three it rises to 2.447"). The learner concludes that momentum makes things worse. The motivating case, the narrow valley, only comes in `optimizers`. | Teach momentum once, in `optimizers`. |
| C7 | `why-gradients` §4 | "The error `e` appears twice in `L = e·e`. So `e` gets two contributions of −6." This confuses a learner who knows d(e²)/de = 2e = −12. | Say: "`e·e` is e², whose slope is 2e = −12. TorC gets the same answer a different way. It treats each use of `e` separately (each gives −6) and adds them. The Backprop lesson shows why adding is right." |
| C8 | `convolution` (planned motivation) | "A pattern detector that slides does not care where the stroke is" mixes up equivariance and invariance. An informal probe on the 20 shown test images (16 correct before the shift; a biased sample, because four were chosen as misreads): a 3 px shift right leaves MLP 10 and CNN 15 correct; 3 px left, 7 and 12; 3 px down, 8 and 9; 4 px right, 3 and 10. TorC's CNN ends in Flatten → Linear, so it is **not** shift-invariant. Also, the digit pad re-centres drawings by centre of mass, which undoes any shift drawn there. | Teach "the feature map moves with the digit", which holds exactly only for shifts that are multiples of the stride. Record shifted-image accuracy of both models on the full 10 000 test images in the trace, and show it honestly, with a test. |
| C9 | `mnist-cnn` | "The CNN, with the settings of the two TorC examples": the trace trains 5 epochs, but `examples/mnist_cnn` uses 10. The page itself says "The TorC example trains for more epochs". The comparison is also confounded (5 vs 20 epochs, different learning rates). | Say plainly: "This comparison is not perfectly fair: different numbers of passes and step sizes. The CNN still wins, with a third of the parameters." Remove the false "settings of the two examples". |
| C10 | `generalization` | The code block shows `DataLoader train(train_set, 64…)` and `AdamW(params, 1e-3f, 0.01f)`, but the run on the page uses 10 samples, lr 0.05, weight decay 0, 0.1 and 1, and 20 000 steps. The check answers "The model overfits" as a yes/no fact. Students treat overfitting as binary and over-diagnose it (Valdenegro-Toro & Sabatelli). | Show the experiment's real settings. Teach overfitting as a matter of degree, and refute "more parameters ⇒ overfitting" (the MNIST MLP has many parameters *and* many images). |

Other content issues (smaller):
- **Unexplained "why"s:**
  - why the error is squared;
  - why `backward()` *accumulates* gradients (it reads like a bug);
  - why identical neurons are bad (missing step: they get equal gradients and stay twins forever);
  - why a tiny or huge signal spread at initialization matters (gradients vanish or explode, so the network cannot learn);
  - why the 2 in √(2/in) (ReLU zeroes about half the values);
  - why shuffle;
  - why softmax uses exponentials (to make every score positive);
  - why the bias needs broadcasting (to add one bias to every row).
- **Magic numbers:**
  - "a learning rate above about 0.1803 is too large" (`sgd-step`, from a Hessian eigenvalue);
  - "below two divided by the steep curvature, here 0.02" (`optimizers`);
  - "we know its curvature" (`why-gradients` check). "Curvature" is never defined.
- **Symbol overloading:**
  - `p` = prediction (`why-gradients`) and then parameter (`sgd-step`);
  - the gradient is `g`, then `p̄`;
  - the batch size is `B`, then `N`;
  - "graph" means a plot and also the computation graph, in the same lesson;
  - the bowl widget says "slope" while the prose says "gradient";
  - `sgd-step` prose says `W` and "step", while its widgets say `w` and "update".
- **Forward references:**
  - MSE in `sgd-step` (taught in `losses`);
  - "training loop" in `sgd-step`;
  - ReLU in `linear-layer`'s init plot (taught in `nonlinearity`);
  - Adam in `modules` code (taught in `optimizers`);
  - batch and sample in `linear-layer` (defined in `training-loop`);
  - 784 in `linear-layer` (explained as 28 × 28 only in `mnist-mlp`).
- **Unread pictures:** the 32 first-layer weight images in `mnist-mlp` come with no instruction on what to look for.
- **Wording:** "the model reads 9666 correctly" is odd; use "classifies" and percentages.

---

## 6. Structure and the new curriculum (R4, R5, R6, R11)

### Problems with the current order
Today's order: `why-gradients`, `sgd-step`, `tensors`, `linear-layer`, `nonlinearity`, `modules`, `losses`, `training-loop`,
`optimizers`, `generalization`, `mnist-mlp`, `convolution`, `mnist-cnn`, plus the `backprop-lab` lab.

The problems are covered in §1 and §5:
- no whole game;
- forward references;
- a lesson on the C++ API (`modules`);
- backprop off the main path;
- no classification picture;
- a new toy problem in every lesson;
- no ending.

The cold reader's most likely quit points:
- `sgd-step` (contour plot and numbers they cannot check);
- `linear-layer` §3 (matrix gradient rules, with no loss defined in that lesson);
- `modules` (C++ plumbing).

### Curriculum v3

Design choices, each supported by the reviews:
- **Backprop belongs right after the learner has a network with many weights.** The question "how does it get all those
  gradients?" exists only then. That is the order 3Blue1Brown and Nielsen use.
- **No standalone linear-algebra lesson.** A maths-only lesson before the payoff is where this learner drops out. Introduce the
  dot product inside "the neuron" (a weighted sum *is* a dot product). Introduce the matrix product as "a layer is many neurons at
  once". Both get just-in-time refreshers.
- **Plant generalization early.** `what-is-learning` scores the hand-fitted line on hidden points, because "does it work on new
  data?" is the definition of learning.
- **ChatGPT asides throughout**, not only at the end. Examples: cross-entropy is the next-word loss; softmax over the vocabulary;
  AdamW trains most LLMs; parameter counts.
- **Each part ends at a real stopping point**: "you can stop here and explain X".
- **Time target:** about 3–4 hours for the core path, 12–18 minutes per lesson, with depth in optional panels. Part 0 plus Part 1
  must deliver a complete payoff in under 45 minutes, so those three lessons aim at the low end (about 12–15 minutes each).

★ = new id, ◆ = existing id, substantially rewritten, ○ = existing id, mostly kept. **Keep each existing id on the lesson that
inherits most of its content** (ids hold URLs, progress, test files and SPEC keys; see §12).

**Part 0. The whole game**
- ★ `what-is-learning`: **What does it mean for a machine to learn?** Open with the digit pad (already built). Then the
  hand-fitting game:
  - drag a line through points;
  - each miss is drawn as a square whose area adds to the loss (MSE, made visible);
  - try to make the loss small (this is the deliberate productive-failure moment);
  - press "let gradient descent take over" and watch it run live;
  - score both lines on hidden points.

  Name the four parts (data, model, loss, adjust) on the loop diagram that returns in every lesson. Promise what the course
  delivers, including "what is different in ChatGPT".

**Part 1. Learning means going downhill**
- ◆ `why-gradients`: **The gradient.** The gold-standard section in Appendix B shows the target. It has:
  - the notebook story;
  - a slope found by nudging (the nudge-lens widget, using the existing `nudge` and `secants` trace data);
  - one worked step and one step the learner does;
  - the steep-bowl refutation (C3);
  - the bowl with "fog mode";
  - `L.backward()` as a "real line" moment.

  The chain rule moves to `backprop`.
- ◆ `sgd-step`: **Gradient descent with many weights.**
  - The line `y = w·x + b` with its data points **listed**.
  - The contour map, with a first-use legend: "each colour is the loss for one pair (w, b); dark is low".
  - Find the gradient the same way as before: nudge `w` alone, then `b` alone. Two weights take two nudges. The list of the two
    slopes is the gradient, drawn as an arrow that points uphill.
  - Fog mode in 2D.
  - Why a learning rate can be too big in the steep direction, as a picture rather than a magic 0.1803.
  - The `SGD::step` line as a "real line" moment.

  Momentum and `zero_grad` leave this lesson.

**Part 2. Neural networks**
- ◆ `linear-layer`: **The neuron and the layer.**
  - One neuron = a weighted sum plus a bias, drawn as circles and arrows, with a dot-product refresher.
  - A layer = many neurons = one matrix product (the matmul grid, 0-indexed).
  - Shapes and parameter counts.
  - The bias is added to every row; this is the place to explain broadcasting in words.

  The backward matrix rules move to the "In TorC" panel; initialization moves to `deep-nets`.
- ◆ `nonlinearity`: **Bending: activation functions.**
  - Two linear layers collapse into one. Show it with plain numbers first: f(x) = 2x + 1, g(x) = 3x + 4, so g(f(x)) = 6x + 7.
  - The **hinge builder**: each hidden ReLU unit is a hinge; switch units on and off and see their sum change shape. This needs
    the trace to record W1, b1, W2 and b2 at each checkpoint.
  - Sigmoid stays. Softmax moves to `losses`.
- ★ `classification`: **Two classes: decision boundaries.**
  - Coloured points in 2D.
  - With 0 hidden units the boundary is a straight line. With 2, 4 or 8 hidden units it bends, and each hidden unit's line is
    drawn in the plane.
  - **The loss.** TorC has only `MSELoss` and `CrossEntropyLoss` (softmax), and no binary cross-entropy. The trace trains with
    `CrossEntropyLoss` on two scores. The page shows "probability of orange". For two classes, softmax of the two scores equals
    the sigmoid of their difference, so sigmoid is introduced here as that probability. Introduce the loss in words, as
    "surprise at the right colour". `losses` then generalizes it to ten classes and shows why it beats squared error.
  - Say once: "TorC finds the gradients for us. The lesson `backprop` shows how."
  - Precomputed presets from a new trace program (recommended), or live JS training (owner decision, §12).
- ◆ `losses`: **Many classes: softmax and cross-entropy.**
  - Softmax turns scores into shares of 100 %; it uses exponentials so that every score becomes positive.
  - Cross-entropy as "surprise", written as **ln**, with a refresher.
  - Why it beats squared error for classes (keep the comparison widget).
  - Ten digits as ten classes.
  - ChatGPT aside: the same softmax, over a vocabulary of roughly 100 000 to 200 000 tokens in current models (cite the source;
    GPT-2 had 50 257).

  MSE is now taught in `what-is-learning`. The MSE shape pitfall goes to the appendix.
- ★ `backprop`: **How the computer finds every gradient at once.** Four parts, no more:
  1. The cost argument, as the payoff of the nudging thread: "With 26 506 weights that is 26 506 nudges, each a full forward
     pass. That is the reason backprop exists." The **nudge-vs-backprop counter** widget shows it.
  2. The chain rule on the notebook model, as gears: 2 × 1 × (−12) = −24, taken from `why-gradients` §4, with C7 fixed.
  3. The "Reuse" graph scene from the Backprop lab, embedded (one graph lab per page).
  4. One small network (2 inputs, 2 hidden units, 1 output) backward with real numbers: "blame flows back along the weights".

  The matrix rule `W̄ = ȳᵀx` (today's `linear-layer` §3) goes to the "In TorC" panel or the `tensors` appendix. This learner
  never needs it.
- ★ `deep-nets`: **Why deep networks are hard to start.** Short. Vanishing gradients with sigmoid. Initialization: the "copy
  neuron" twins *with* the missing reason (equal gradients keep them equal forever), and the six-layer signal plot, now after
  ReLU, with the reason for the 2 in √(2/in).

**Part 3. Training in practice**
- ○ `training-loop`: **The training loop.**
  - Batches and epochs.
  - The S in SGD (a batch's gradient is an estimate).
  - Why shuffle.
  - A fair batch-size comparison (C1).
  - The five-line loop as a "real line" moment.
  - `zero_grad` in the "In TorC" panel, with the reason: TorC adds each new gradient to the stored one. That lets one
    `backward()` per use add up, so you must clear it between steps.
- ○ `optimizers`: **Better optimizers.**
  - Momentum, taught once, with the ball analogy.
  - Adam as "a step size per weight, scaled by the typical size of its recent gradients".
  - AdamW.
  - The C4 caveat.
  - Label the race curves directly on the plot, and replace the monospace readout table with a loss-vs-step plot.
- ○ `generalization`: **Does it work on new data?**
  - Train, validation and test (C2), with the **validation-stop game**: pick the stopping step from the validation curve, commit,
    then see the test loss once.
  - Overfitting as a matter of degree (C10).
  - More data helps.
  - Define "variance" before the noise-floor check, or rephrase that check without the word.

**Part 4. Images**
- ○ `mnist-mlp`: **Digits with an MLP.**
  - Mistakes per 100.
  - Say what to look for in the weight images.
  - A **confusion matrix** and a **grid of the mistakes**, with "why might 4 and 9 get confused?". This needs a 10 × 10 confusion
    matrix recorded in `trace_mnist_mlp.cpp` and a re-run.
  - "Channels" explained (red, green, blue), ahead of `convolution`.
- ★ `one-step`: **One training step, followed end to end.** The capstone of the course, and the strongest use of "real library,
  real numbers". One test digit through the real MNIST MLP:
  1. Forward: pixels → hidden activations → ten scores → probabilities → surprise.
  2. `backward()` gives the gradient of one chosen pixel weight, traced back along its path.
  3. `step()` changes that weight by lr × gradient.
  4. The same image runs again, and the probability of the right digit has risen a little.
  5. "Now repeat this 9 380 times."

  Every number causes the next, so this is the opposite of number narration. Needs a trace that records one batch's forward
  values, the gradients of a few named weights, and the probabilities before and after the step.
- ◆ `convolution`: **Convolution.**
  - Motivation: an MLP has a separate weight for every pixel. A kernel reuses the same small pattern detector everywhere, so it
    needs few weights, and its feature map moves with the digit (C8).
  - The sliding window, explained with a picture before the ⌊ ⌋ formula.
  - Channels.
  - The parameter-count prediction.
  - "Does not flip the kernel" goes to "In TorC".
- ○ `mnist-cnn`: **Digits with a CNN.**
  - Fix C9.
  - Replace the look-up check with a reasoning question.
  - End with a recap of the whole course and the bridge to the epilogue.

**Part 5. Where this goes next**
- ★ `from-digits-to-chatgpt`: concepts only, no TorC code, and honest that TorC stops at CNNs.
  - Text becomes tokens, and tokens become vectors (embeddings).
  - **Next-token prediction with cross-entropy, where the label comes for free from the text itself** (self-supervision). This
    one idea connects everything in the course.
  - Attention as "each word takes a weighted average of the others, with weights it learns".
  - Pretraining uses exactly this course's loop, at enormous scale. Later stages (fine-tuning, learning from human feedback) use
    other objectives.
  - End with a curated list: 3Blue1Brown's neural network series, Karpathy's *Zero to Hero*, Nielsen's *Neural Networks and Deep
    Learning*, d2l.ai.

**Appendix track: "Using TorC"** (outside the lesson order, like the labs; needs `kind: "appendix"`, see §11).
- ◆ `tensors`: storage, `reshape` copies, the full broadcasting rules, reductions and axes, batched `matmul`. Keep the broadcast
  grid and the axis explorer.
- ◆ `modules`: `Module`, `Sequential`, `named_parameters()`, why `parameters()` has no fixed order, and the forward-cache rule.
- ★ `torc-pitfalls`: "Why did my X fail?". Covers:
  - `zero_grad` and gradient accumulation (with the `sgd-step` loop widget);
  - graph lifetime;
  - `step()` skipping parameters that have no gradient;
  - removing `loader.reset()`;
  - the MSE shape pitfall (move it here or to `docs/library-audit/` only; decide);
  - `CrossEntropyLoss` label errors;
  - recorded error strings.

**Labs.** Keep `backprop-lab` as a free playground. Later, add a "train a tiny network" playground.

**Throughline.** Parts 0–2 reuse the same few datasets: the notebook price; one line-fitting set, which can come from
`training.json` x in [−1, 1) or from a new, meaningful dataset with a new trace; and one 2D classification set. That way each new
idea arrives on familiar data.

---

## 7. Lesson-by-lesson notes (current pages)

For the rewrite. Each entry gives what to keep, what to fix, and the misconception to refute. Ids are as today.

- **`why-gradients`.**
  - Keep: the bowl and tangent; the lr presets 0.25 (bounces) and 0.3 (diverges); "if you increase `w` by a small amount, `L`
    changes by about −24 times that amount"; the check "why does lr 0.125 not work for a real network?" (a good reasoning
    question, but define "curvature" or rephrase).
  - Fix:
    - Open with the story, not C++.
    - Call the gradient "the slope of the loss curve" once (allow this in `authoring.md`).
    - Get −24 by nudging.
    - Explain ← ("replace `w` by").
    - Write `lr` upright, so it does not read as l × r.
    - Drop the formula `w_{k+1} − 3 = f(w_k − 3)` from the main text (its "8" is never derived); keep the widget and one sentence.
    - Fix C3 and C7.
    - The "predict lr 0.25" question needs the gradient at `w` = 6, which is not given. Give it, or ask only for the first step.
  - Misconceptions: "the gradient points to the minimum"; "a smaller learning rate is always better".
- **`sgd-step`.**
  - Keep: the contour trail; "four times the learning rate is not four times faster" (but move it before the sentence that gives
    the limit).
  - Fix:
    - List the three data points.
    - Fix C6.
    - Remove the magic 0.1803.
    - Unify `W`/`w` and step/update.
    - Explain the contour map and the log axis at first use.
    - Fix c-zero (it needs the one-step gradient, −22.67, to be stated).
    - Fix the two different "after ten steps" losses (0.001145 after 10 updates in §2; 0.001184 after 9 updates in §4).
  - Misconceptions: "the optimizer computes the gradient"; "the loss must fall at every step".
- **`tensors`** (to the appendix).
  - Keep: "(2) + (2, 3)?" (it targets a real misconception); the broadcast grid; the axis explorer.
  - Fix:
    - "The number of a dimension is also called its axis" reads as "how many". Say "the position of a dimension".
    - Give one worked 2 × 2 matrix product before the index formula.
    - Show a 3D example (an image).
- **`linear-layer`.**
  - Keep: the matmul grid and its paper work; "copy neuron 1 into neuron 2"; the signal plot (moves to `deep-nets`).
  - Fix:
    - D1, D3, D4.
    - Add the network picture.
    - Explain why `W` is (out, in) and is transposed.
    - The x̄ predict is not answerable for this persona; drop it from the main path (the matrix rule goes to "In TorC").
    - Give the reason the twins are bad.
    - Explain 784.
  - Misconceptions: "W is (in, out)"; "a neuron is a little brain"; "zero weights are a fine start".
- **`nonlinearity`.**
  - Keep: the collapse argument (add the plain-number version); the no-ReLU fit with the flat-line reveal (best question in the
    course). Its prompt hint ("think about the best straight line through a curve that is symmetric around zero") gives the answer
    away; drop the hint.
  - Fix: add the hinge builder. Say the consequence of a small sigmoid gradient ("that weight barely learns"). Move softmax. Move
    "overflow" to "In TorC".
  - Misconception: "ReLU itself is the curve".
- **`modules`** (to the appendix). Reuse the parameter-count viewer in `linear-layer` and `mnist-mlp`.
- **`losses`.**
  - Keep: the CE vs squared-error gradient comparison; the uniform-logits check (with ln stated).
  - Fix:
    - D6.
    - Say "ln".
    - Introduce ∂, or keep d/d (see §8 notation).
    - Explain "logit" in words.
    - The predict is half given away by the summary at the top; move the summary to the end (§9).
- **`training-loop`.**
  - Keep: the batch viewer; the commented five-line loop (the clearest thing in the course; show its picture already in
    `what-is-learning`); the noise-floor section (3.002 / 1.013 vs 3 / 1 helped).
  - Fix: C1, D5. Define "regression" and "synthetic". Give a reason to shuffle. Move the `reset()` check to the appendix.
- **`optimizers`.**
  - Keep: the narrow valley; "Adam divides the average gradient by its typical size"; the first-step check (≈ lr for both
    gradients).
  - Fix:
    - C4.
    - "Curvature" (define it or avoid it).
    - Explain the hats on m and s in words before the formula, or drop the hats from the main text.
    - The weight-decay purpose comes only in the next lesson; say "the next lesson shows why this helps".
    - Label the curves directly; replace the table.
- **`generalization`.**
  - Keep: the degree-15 fit; underfitting with weight decay 1; the noise-floor idea.
  - Fix: say "a polynomial of degree 15" (the persona knows polynomials) instead of "`nn::Linear` on the powers of `x`". Also C2,
    C10, "variance", and the predict that its own heading gives away.
- **`mnist-mlp`.**
  - Keep: the digit pad, the misread images, the weight gallery (with guidance).
  - Fix: D9; say what MNIST is; mistakes per 100; the confusion matrix; C2 (overfitting from train vs validation, not test alone).
- **`convolution`.**
  - Keep: the sliding window with its arithmetic; edge-kernel maps; the 6 280 vs 80 prediction.
  - Fix: C8; a picture before ⌊ ⌋; "channels"; the code block that hides its shape comments (D14); `B` vs `N`.
- **`mnist-cnn`.**
  - Keep: the trained feature maps; the comparison (with C9).
  - Fix: C9; replace "which digit does the CNN read worst?" (a look-up) with "why would a CNN confuse 7 and 1 less often than an
    MLP?" or similar; a recap and bridge.
- **`backprop-lab`.** Keep it as a lab. Its line "the value gets a gradient from each place; TorC adds them" is the explanation
  that `why-gradients` §4 lacked; reuse it in `backprop`.

---

## 8. Voice, notation and analogies (R8, R9)

### Voice: keep STE's structure and add specified teacher moves
Decision 12 brought in STE *because* of an earlier complaint: the pages "read like LLM output". Do not drop it wholesale. The
typical LLM tics (rhetorical questions everywhere, punchlines, decorative metaphors, "we" and "let's") would come back.

**Keep:**
- complete sentences;
- one idea per sentence, ≤ 25 words;
- define every term before use;
- one name for one thing;
- active voice;
- no slogans and no "X, not Y" punchlines;
- headings that state the topic.

**Change:**
- **Allowed teacher moves:**
  - At most one real question per section, answered by the next paragraph. Opening questions and prediction prompts don't count
    toward this.
  - "You" for instructions, tasks and predictions.
  - Analogies only from the bank below, each with its limit stated at first use.
  - A concrete situation before the abstraction ("Two notebooks cost 6 euros").
- **Banned tics**, added to the linter as a word list:
  - "let's", "here's the thing", "it turns out", "simply", "magic", "crucially", "notice that", "the key insight", "under the
    hood", "in other words", "think of it as", "imagine", "essentially", "at its core", "the heart of", "powerful", "elegant",
    "beautiful", "not just X but Y", "That's it", "This is where X comes in";
  - "This lesson shows" as an opener.
- **Flag for the reviewer, do not fail the linter:** "just" used as a hedge ("just multiply"); three-item lists used for rhythm; a
  sentence that only restates the paragraph as a punchline. These go in Appendix A.
- **Linter patterns to allow:** the minus sign (U+2212) and dashes inside maths are not punctuation dashes; label colons such as
  "*Task:*", "*Reveal:*" and "*Where it breaks:*" at the start of a block are allowed.
- **Punctuation:** keep the bans on semicolons and on dashes as punctuation. Allow a colon before a formula, list or code (as
  today).
- **`authoring.md` "one name" list:**
  - Allow "slope" once, to *define* the gradient ("the gradient is the slope of the loss curve at the current point"). After
    that, say "gradient".
  - Allow "knob" only inside the lesson-0 analogy, with the explicit mapping "the knobs are the weights".
- **The real test** is the owner's focus group and the human think-alouds (§10), not a principle.

### Notation (decide once, record in `authoring.md`)
- **Gradient.**
  - In Part 1 prose: dL/dw, read aloud as "how much the loss changes per unit change of w".
  - Introduce ∂ once, in `backprop`: "the same as d, used when the loss depends on many weights".
  - Keep the overline only inside the graph-lab maths windows, defined there.
  - This keeps the overline notation in `explain.js` and `ops/*` unchanged (their index subscripts still change under D12). The
    prose stops mixing three notations.
- **Indices.**
  - Tensor positions count from 0, everywhere (prose, widgets, TeX in `ops/*`, `linear-math.js`).
  - Counts of epochs, steps and "the first neuron" in prose count from 1.
  - Fix D4 and D12 under this rule.
- **Symbols in one guide:** ←, Σ, subscripts, ᵀ, ⌊ ⌋, hats, the overline, and the Greek letters with their names (μ "mu",
  β "beta", λ "lambda", ε "epsilon", σ "sigma").
  - **The guide:** one page, `site/notation/`, or a refresher kind.
  - **First use:** each symbol links to the guide when it first appears.
  - **Logs:** "log" always means ln, and the site says so at first use.
- **One letter per thing:** batch size is always `B`; the prediction `y` (or `ŷ` once defined); the parameter `w`/`W`. Never
  reuse `p` for two things.

### Maths refreshers (just in time, collapsible, 100–200 words with a picture each)
derivative as slope (found by nudging) · Σ notation · vectors and the dot product · the matrix product (a worked 2 × 2) · exp and
ln · probability and "adds up to 1" · mean, variance and standard deviation · the normal distribution · how to read a contour map ·
how to read a log axis.

### Glossary
One source of truth (`site/glossary.json`). A hover or tap definition at the first use of each term in each lesson. A test checks
that every glossary term a lesson uses is linked at its first use.

### Analogy bank (use at most one per section; state the limit)

| Idea | Analogy | Where it breaks |
|---|---|---|
| Gradient | Standing on a hillside in thick fog: you feel the slope under your feet, but you cannot see the valley | It tells steepness and direction *here*, not where or how far the bottom is |
| Learning rate | Stride length, scaled by how steep the ground is | A stride that is safe on gentle ground overshoots on steep ground |
| Momentum | A ball rolling downhill keeps part of its speed | A real ball has friction and mass; here μ sets how much speed is kept per step |
| Adam | Each weight gets its own step size, scaled by the typical size of its recent gradients | It is not "smart". It can still diverge, and it needs its own learning rate |
| Chain rule | Gears: if B turns 2 times as fast as A and C turns 3 times as fast as B, then C turns 6 times as fast as A | Real derivatives change from point to point; gears have fixed ratios |
| Neuron | A weighted sum (for classification: a weighted vote) plus a bias that shifts the threshold | It is not a brain cell; it only multiplies, adds and applies one function |
| ReLU units | Hinges: a sum of hinges bends a straight line into a curve | It gives straight pieces only. With enough hinges it approximates any continuous curve on an interval. 8 hinges give at most 8 bends |
| Softmax | Turning raw scores into shares of 100 % | Shares say how sure the model is, not that it is right |
| Cross-entropy | Surprise when the right answer is revealed: 100 % sure and right is zero surprise | Being sure *and wrong* costs without bound. Only the probability of the right answer counts, not how the rest is spread |
| Overfitting | Memorizing last year's exam answers instead of learning the subject | The model does not choose to memorize. It fits noise because nothing stops it, and it is a matter of degree |
| Weight decay | A slow leak that pulls each weight toward zero unless the data holds it up | In TorC a weight with *no* gradient is skipped and does not leak |
| Convolution kernel | A small stencil slid over the image, asking "is my pattern here?" at every spot | The feature map moves with the image; the final answer of a CNN can still change when the digit moves |

---

## 9. Lesson template, checks and widgets (R7, R10)

### The template: a core plus a palette

**Core, in every lesson:**
1. **"Questions this lesson answers"**: 1–3 questions the persona would ask, each set in a concrete situation. This replaces
   the TL;DR at the top, which today answers the questions before the learner has asked them.
2. **A committed prediction**, placed *before* any heading or text that contains its answer.
3. **One "aha" interaction** with a guided task. The text also states the observation, so the page works without the widget.
4. **One worked example** in which the learner does the last step.
5. **One misconception refuted** with the widget's own evidence.
6. **"The answer"**: two or three sentences that answer the opening questions. This is where the old summary goes. Then the next
   question, which the next lesson answers.

7. **One transfer question** (a new situation, the same idea) and **one self-explanation prompt** with a model answer.
8. **One cumulative check** from an earlier lesson, from the third lesson on.

**Palette, 0–2 per lesson:** an analogy with its limit; a maths refresher; an "In TorC" panel; a "same thing in ChatGPT" aside;
an end-of-part review (it recommends going back and never blocks).

**Exceptions:** `from-digits-to-chatgpt` and the appendix pages need no worked example and no widget-based refutation.

A 14-item checklist ticked on every lesson would produce a formula, and a formula reads like LLM output. Use the core plus
judgement instead.

### Checks
- **New section kind `quiz`.** It replaces "Show the answer" + "I got it".
  - Answer types: multiple choice with 3–4 options, where each wrong option matches a real misconception and has feedback that
    names it; a numeric answer with a tolerance; or "drag the marker to where you think `w` lands".
  - The reveal opens after the attempt.
  - Keyboard and screen-reader behaviour are specified and tested.
  - It works with storage blocked.
- **Progress store schema v2** (`torc:v2`). Record an attempt and its correctness, not a self-report. Migrate v1 "I got it" flags
  as "seen", never as "correct". `site/js/core/store.js` `sanitize` today keeps only booleans, and `core.test.mjs` asserts this.
- **Question quality rule:** no prompt may be answerable by re-reading the sentence just before it. Every number in a prompt that
  the learner needs must be on the page (for example, `sgd-step` c-zero needs the gradient −22.67).

### Widgets
- Each widget has 1–3 guided tasks with the expected observation stated in the text.
- **Readouts:**
  - 2–3 significant figures;
  - a meaning label ("loss 1.2, lower is better");
  - only what the task needs;
  - no visible multi-column tables (decision 9 stands; the `optimizers` race readout breaks it today).
- **Labels and links:** label curves directly on the plot, and colour-link symbols across the formula, the plot and the code.
- **Controls:** replace the "≤ 2 widgets, ≤ 2 controls" cap (decision 7 and `authoring.md`) with "every widget earns its place
  with a guided task". Some lessons need three small widgets.

**The "aha" widgets, with data sources:**

| Widget | Lesson | Data today | Work needed |
|---|---|---|---|
| Nudge lens (drag Δw, predicted vs actual change, zoom until straight) | `why-gradients` | `why-gradients.json` `nudge`, `secants` | Widget only |
| Fog mode (hide the curve; step blind; then lift the fog), in 1D and 2D | `why-gradients`, `sgd-step` | `why-gradients.json` grid/trajectories, `optim.json` runs | Widget only |
| Steep vs shallow bowl, same minimum | `why-gradients` | none | New trace scene (C3) |
| Gradient arrow on the contour map | `sgd-step` | `optim.json` runs and surface (`contour2d.js` already draws a step arrow) | Small widget change |
| Hand-fitting game with residual squares, then "let gradient descent take over", scored on hidden points | `what-is-learning` | `training.json` data (t = 3x + 1, x in [−1, 1)) and `runs[].epochs` | Widget and JS MSE/SGD (testable against the trace). A meaningful dataset needs a new trace |
| Hinge builder | `nonlinearity` | `nonlinearity.json` has only predictions per checkpoint | Trace change: record W1, b1, W2, b2 at each checkpoint |
| 2D classification playground (0, 2, 4, 8 hidden units; each unit's line drawn) | `classification` | none | New trace program with boundary grids per checkpoint (recommended). Live JS would need ports of sigmoid/softmax/CE and Adam, plus a trace scene for each new op (`ops.test.mjs`) |
| Nudge-vs-backprop cost counter | `backprop` | parameter counts in `modules.json` / `linear.json` | Widget only (arithmetic) |
| Small network backward with real numbers | `backprop` | `autograd_trace.json` scene `mlp` | Check the scene fits 2-2-1; else a new scene |
| One step followed end to end | `one-step` | `mnist-mlp.json` weights and 20 test images | Trace change: one batch's forward values, named-weight gradients, probabilities before and after the step |
| Validation-stop game | `generalization` | `generalization.json` has train/test only | Trace change: a validation split |
| Confusion matrix and mistake grid | `mnist-mlp` | only `test_per_class.correct/total` | Trace change: 10 × 10 matrix and the misclassified indices; full MNIST re-run |
| Feature map follows the shift, plus measured robustness | `convolution` | `mnist-cnn.json` weights; `cnnFirstMaps` in `digit-widgets.js` | Shifts as multiples of the stride. Robustness numbers from a new trace field with a test. Bypass the pad's re-centring |
| Multi-seed ranges for headline comparisons | several | single seeds | Trace changes (C5) |

Every new trace program needs a `TRACES` line in `tools/trace/run.sh` and an entry in `TRACE_TARGETS` in
`tools/trace/CMakeLists.txt`. Regenerating needs clang ≥ 18 or GCC ≥ 14, network access (TorC v0.1.0, MNIST) and several
minutes for the MNIST runs. `verify.yml` checks all of it for drift.

### Home page
- Lead with the promise and a live demo (the digit pad), the prerequisites, and an honest total time.
- Move the C++ cards into a "Show the TorC code" toggle on each card. This revises decisions 10 and 11 (§12). If any
  `course.json` `code` changes, its `SPEC` in `tools/snippets/check.mjs` must change too.
- Keep the progress ticks and the filter. Add the appendix section.
- **Light theme.** The shell template sets `<meta name="color-scheme" content="dark">` (`tools/check_site.mjs`), so all shells
  must be rewritten with `--write-shells`. Raise the minimum caption size in widgets.

---

## 10. Quality gates and process (R12)

- **The current gates do not measure learning.** The `independent-reviewer` agent's dimensions are about correctness. Its
  "writing" dimension does ask for undefined terms, but nothing asks "would the target learner understand this?".
- **Add `.claude/agents/pedagogy-reviewer.md`** (read-only). Its checklist is Appendix A, which is the core template plus C-type
  checks: fair comparisons, no test peeking, misconceptions refuted with evidence.
- **Add `.claude/agents/cold-reader.md`.**
  - It role-plays the §2 persona. It reads `build/prose/<id>.md` (from `tools/prose/extract.mjs`) and the phone and desktop
    screenshots.
  - It answers every prompt *before* reading the reveal, and reports every confusion point.
  - **This is a smoke test, not evidence.** The cold read for this brief found real problems (the log base, symbol overloading,
    quit points), but it is not a learner.
- **Human protocol**, the real gate:
  - Test each part with 3–5 people from the target group, thinking aloud while one person observes.
  - Before and after each lesson, ask three short questions, including "explain the main idea in one sentence".
  - Record where each person stopped.
  - The site is static, with no backend and no analytics, so this protocol is the only learning data there is. ROADMAP M3 lists
    "outside cold reader per lesson" as not done.
- **Budgets.** Replace the 800-word cap with a time target of 12–18 minutes, plus the rule "every paragraph explains a why, sets
  up a task, or answers a question". Tests still need something measurable, so keep a soft word cap of 1 400 per lesson and the
  number rules of §4.
- **Tests to add:**
  - every lesson has a "Questions" opener and an "Answer" closer;
  - glossary terms are linked at first use;
  - no symbol from the notation list appears before its introduction, in course order;
  - role-tagged number bindings (§4);
  - no LaTeX escape bugs (grep the source for a single backslash before `;`, `,`, `!`, `:` or a space; D1);
  - the closing "next lesson" line matches `neighbors()` in `site/js/core/course.js` (D2);
  - the visible UI strings in widgets and `lesson-boot.js` are linted too (D15).

**Files and tests that must change** (so the implementing session is not surprised):

| Change | Files and tests |
|---|---|
| Voice and linter rules | `site/test/helpers.mjs` `proseProblems`; `tools/prose/check.sh` and its step in `verify.yml`; `CLAUDE.md` writing section (owner approval); `docs/authoring.md` writing rules and the "one name" list; decision 12; `.claude/agents/independent-reviewer.md` dimension 8 |
| Number rule and formatting | `site/js/core/trace.js` `makeT`; `site/js/core/sections.js` `boundNums`; `site/test/lessons.test.mjs` (the `T.text` equality and the ≤ 25 budget); the per-lesson budget tests in `why-gradients`, `linear-layer` and `sgd-step` `.test.mjs`; the per-lesson "no hand-typed digits" tests in all 13 `site/test/<id>.test.mjs` (allow anchors); tests that pin removed bound paths (for example `linear-layer.test.mjs`, `sgd-step.test.mjs`, `why-gradients.test.mjs`); `engine.test.mjs` only if `fmt` changes |
| Template (Questions/Answer instead of TL;DR) | `lessons.test.mjs:107` (TL;DR right after the hero), `:117-118` (≤ 60 / ≤ 100 words), `:121` (≤ 4 numbered sections); `why-gradients.test.mjs:149` (≤ 4 items); the `tldr` renderer in `sections.js` (heading "Summary"); `helpers.mjs` `lessonStats` and `sectionWords`; decision 7 |
| `quiz` kind and store v2 | `site/js/core/sections.js` `SECTION_KINDS` and `validateLesson`; `site/js/lesson-boot.js`; `site/js/core/store.js` (`STORAGE_KEY` `torc:v1`, `sanitize`); `core.test.mjs:61`; `lessons.test.mjs` question budget (≤ 1 predict + 2 checks) |
| Appendix part | `site/js/core/course.js` (`validateCourse` rejects unknown kinds; `isLabPart`, `curriculum`, `labs` and `neighbors` need a three-way kind; header comment `:4-8`); `core.test.mjs:82` (asserts the rejection); `site/js/home.js`; `docs/architecture.md:36`; prerequisites in `course.json` (for example `losses.prereqs` is `["modules"]`). `tools/check_site.mjs` needs no change |
| "In TorC" panel wording | `lessons.test.mjs:110` bans "recorded / recomputed / printed by / in your browser" |
| Widget and UI string linting (D15) | `tools/prose/extract.mjs`, `site/test/helpers.mjs` |
| A lesson numbered 0 | `curriculum()` numbers from 1 (`course.js`, `core.test.mjs`). Easiest: no "0"; start at 1 |
| Lessons without trace data (`from-digits-to-chatgpt`, refreshers, glossary) | `lessons.test.mjs` footer check needs a lesson `footer`. `CLAUDE.md` "source of truth" needs a rule for content beyond TorC: cite sources, keep numbers out or cite them, label conceptual content as such (owner approval) |
| Removing momentum and `zero_grad` from `sgd-step` | `sgd-step.test.mjs:91-121` pins the six runs and the presets, `:177` pins `runs.4.steps.3.loss` (2.447) and `runs.3.final.loss` (1442); the `course.json` `code` for `sgd-step` and its SPEC |
| New traces | `tools/trace/run.sh`, `tools/trace/CMakeLists.txt`, `site/data/`, tests per file |

---

## 11. Defects

Phases: **A** = D1–D3, D5, D6, D10, D13–D15, D17, D19, D20, and also D4 and D12 (`authoring.md` already says positions count
from 0). **B** = D8, D9 (they need the prose formatter). **C** = D7, D11 (with C10's code change), D16, D18.
Of the conceptual errors, the wording parts of C4, C7, C9 and C10 are Phase A. The rest go with their lesson's rewrite.

| # | Where | Defect | Fix |
|---|---|---|---|
| D1 | `site/js/lessons/linear-layer.js:56` | The shape formula renders as `x ;·; Wᵀ ;+; b ;=; y`. In a single-quoted JS string, `'\;'` is a bare `;`, which KaTeX parses without error (so the test missed it). | Write `\\;`. Add the escape-grep test. No other broken escapes were found. |
| D2 | `site/js/lessons/sgd-step.js:115` | "The next lesson, Layers", but the next lesson is Tensors (the nav button on the same page says Tensors). | Link `../tensors/`. Better: generate the line from `neighbors()`, or test it. |
| D3 | `site/js/lessons/linear-layer.js:104-107` | Ends with "Go back to the course map." instead of naming the next lesson. | Name Nonlinearity and link `../nonlinearity/`. |
| D4 | `site/js/widgets/matmul-grid.js:46-52, 73, 87`; `site/js/widgets/linear-math.js:2, 42-44`; `linear-layer.js:101-102`; `site/test/linear-layer.test.mjs:129, 137` | The grid and its TeX count from 1 (`i=1`, `k=1`, `\sum_{j=1}`, "Copy neuron 1 into neuron 2"). `linear-math.js:2` even documents "1-based, as in the rest of the site's paper work". But `authoring.md` and `tensors` say positions count from 0. The check "increase `b[2]` … the second column" is right only under 1-based indexing. | Apply the index rule of §8: tags `i=${i}`; `I = i`, `K = k`, `\sum_{j=0}^{n-1}`; button "Copy the first neuron into the second"; check `b[1]`; update the comment and tests. |
| D5 | `site/js/lessons/training-loop.js:46` (`cf` at `:11`) | Code shows `0.10000000149011612f`: the float32 value from `training.json` is printed as a double. | Print the shortest precision p (1–9) with `Math.fround(Number(v.toPrecision(p))) === Math.fround(v)`. |
| D6 | `site/js/lessons/losses.js:51` | The MSE shape issue ("3.333 instead of 0.1667") is an audit finding (`docs/library-audit/v0.1.0-findings.md`) on a lesson page, so it breaks decision 4 directly. | Remove from the lesson now. Later, put it in the appendix `torc-pitfalls` or keep it in the audit docs only (decide). |
| D7 | `site/js/lessons/modules.js:64-67` | `parameters()` order from an unordered map is a library quirk on a lesson page. | Moves with `modules` to the appendix (Phase C). Until then, leave it. |
| D8 | lessons vs home page | Thousands formatting differs. Home (hand-written comments in `course.json`) says "26,506" and "9,098"; lessons say "26506", "25450", "60000"; `matmul-grid.js:33` says "1,000,000". | A prose formatter in `makeT` (not `fmt`, which tests pin and which feeds KaTeX, where a comma needs `{,}`); one separator; update the tests that pin shown strings. |
| D9 | `mnist-mlp.js:66, 71, 85`; `mnist-cnn.js:53, 81` | Accuracy shown as counts. No trace field holds a percentage. | A bound derived value (precedent: `sci()` in `linear-layer.js`, with per-lesson tests), or a new trace field. |
| D10 | `mnist-cnn.js:53` | "with the settings of the two TorC examples" is false (5 epochs traced vs 10 in the example). | See C9. |
| D11 | `generalization.js:43-45`, `course.json` code | The code shown does not match the experiment. | See C10. |
| D12 | `site/js/ops/matmul.js:18`, `ops/elementwise.js:18`, `ops/relu.js:14`; `site/js/widgets/training-widgets.js:96` | More 1-based indices in the graph-lab TeX. The training readout says "epoch 1 … batch 0 of 4" while the code loops from `epoch = 0`. | Apply the §8 index rule. |
| D13 | `sgd-step.js:68` and `:92` | Two "after ten steps" losses (0.001145 after 10 updates; 0.001184 after 9 updates), never explained. | Use the same quantity, or drop both (§4 rule). |
| D14 | `convolution` "A small CNN" code block (808 px in a 702 px box) and the `mnist-cnn` model block (737 px in 702 px), at 1366 px | The shape comments are hidden unless the reader scrolls sideways. | Put the comments on their own lines, or shorten them. |
| D15 | `matmul-grid.js:33` (semicolon), `:95` (mid-sentence colon); `contour2d.js:131, 250` (semicolons); `site/js/lesson-boot.js:32` (em dash, "✓ Done — click to undo"); `site/js/widgets/inputs-popover.js:7` ("recomputed in your browser") | Visible UI strings break the writing rules, and one uses provenance wording that `lessons.test.mjs` bans on lesson pages. Neither `proseProblems` nor `extract.mjs` covers widget strings, although `CLAUDE.md` says the rules apply to them. | Fix the strings. Extend the extraction to widget and UI strings. |
| D16 | graph labs at 390 px (`why-gradients` chain lab, `backprop-lab`) | The step-timeline chips strip is clipped ("← BACKW…"). The graph needs horizontal scrolling (640 px in 356 px), so `L` is off-screen at the start. The hint "Press Next or → to start" shows on phones. | Wrap the chips or make them scrollable with a cue. Fit the graph to the width, or start the view at the active node. Use a touch-specific hint. |
| D17 | `modules.js:56` | `optim::Adam` appears in code three lessons before Adam is taught. | Use `optim::SGD`, or move the lesson (Phase C). |
| D18 | `optimizer-widgets.js` race readout | A visible monospace table, against decision 9. | A loss-vs-step plot with direct labels. |
| D19 | `sgd-step` widgets | The prose says `W` and "step"; the widgets say `w` and "update" (`contour2d.js:131, 149`; `descent.js:29, 37`). | One name per thing. |
| D20 | `why-gradients` bowl readout | Says "slope" while the prose says "gradient". | "gradient". Or define "slope" once, per §8. |

---

## 12. Work plan for the next session

### Owner approvals needed before Phase B (ask in one message, list them)
1. The audience change (R1). This changes the pitch in `content-plan.md` §1.
2. The voice change: STE structure stays, but `CLAUDE.md`'s "load asd-ste100" rule and decision 12 change (R8).
3. The concept layer, the "In TorC" panel and the appendix track. `modules` and `tensors` leave the main path (R6).
4. The home page: demo first, code behind a toggle. This revises decisions 10 and 11.
5. Decision 7 (the TL;DR becomes "Questions" and "The answer"; ≤ 2 widgets and ≤ 2 controls replaced) and the new budgets.
6. A rule for content beyond TorC (`from-digits-to-chatgpt`, refreshers, glossary): sources are cited, and numbers are either left
   out or cited.
7. Adding the `pedagogy-reviewer` and `cold-reader` agents.
8. A work branch (`web-v3`), or temporary draft ids on `web` (§0).
9. Live JS training vs precomputed presets for the playgrounds (recommend: precomputed presets first; live training later, under
   decision 2 with conformance tests).
10. Small choices: the thousands separator (thin space or comma); where the MSE pitfall lives (appendix or audit docs only).

### Phase A. Quick fixes (one session, safe on `web`)
- Copy this brief to `web` as `docs/pedagogy-review.md`.
- Fix D1–D3, D4, D5, D10, D12, D13, D14, D15, D17, D19, D20 and the wording parts of C4, C7, C9 and C10 (caveats, true settings).
- Remove the D6 text.
- Add the escape-grep test and the next-lesson test.
- If C10 changes the `course.json` `code` of `generalization`, update its SPEC in `tools/snippets/check.mjs` and run
  `node tools/snippets/check.mjs --torc <TorC at v0.1.0>`.
- Run `node --test site/test/*.test.mjs`, `tools/prose/check.sh` and `node tools/check_site.mjs`. Run `tools/trace/run.sh --check`
  if the toolchain and network are available (Phase A changes no traces, and CI runs it). Check pages at 1366 × 768 and
  390 × 844.
- Push.

### Phase B. Decisions, docs and the vertical slice
- Record the approved decisions.
- Update `content-plan.md` §1 and §2, `authoring.md` (voice, notation, index rule, number roles, template, budgets), `CLAUDE.md`
  (with approval) and `.claude/agents/`.
- Build the shared machinery:
  - the `quiz` kind and store v2;
  - role-tagged `T()` and the prose formatter (this also fixes D8 and D9);
  - the "Questions" and "The answer" sections;
  - the "In TorC" panel (a `<details>` inside text sections is enough);
  - the refresher component, glossary and notation guide;
  - `kind: "appendix"`.
- **The vertical slice:** `what-is-learning`, `why-gradients`, `sgd-step` (Part 0 and Part 1), on the work branch or under
  temporary draft ids (§0).
- Run the pedagogy reviewer and the cold reader on them, then **at least three human readers** from the target group.
- Adjust the rules in this brief from what they show, then continue.

### Phase C. The rest of the main path, part by part
- **Part 2:** `linear-layer`, `nonlinearity`, `classification`, `losses`, `backprop`, `deep-nets`.
- **Part 3:** `training-loop`, `optimizers`, `generalization`.
- **Part 4:** `mnist-mlp`, `one-step`, `convolution`, `mnist-cnn`.
- **Part 5:** `from-digits-to-chatgpt`.
- **Appendix:** `tensors`, `modules`, `torc-pitfalls`.
- With each part: its trace work (§9 table), its C-fixes, and a human test before publishing.

### Phase D. Spacing and polish
- End-of-part reviews and cumulative checks.
- The light theme.
- Firefox and Safari.
- A real screen reader.
- The digit pad on a touch device.

### Ids, URLs and progress
- **Ids that stay:** `why-gradients`, `sgd-step`, `linear-layer`, `nonlinearity`, `losses`, `training-loop`, `optimizers`,
  `generalization`, `mnist-mlp`, `convolution`, `mnist-cnn`, and `tensors` and `modules` (moved to the appendix).
- **New ids:** `what-is-learning`, `classification`, `backprop`, `deep-nets`, `one-step`, `from-digits-to-chatgpt`,
  `torc-pitfalls`.
- **No id is removed.** So no redirects are needed and stored progress keeps working. If an id is ever removed, keep a static
  page at its URL that links to the new place.
- When a lesson changes a lot, keep its "done" flag (it is per id), but its check ids may change. Store v2 treats unknown check
  ids as absent.

### Acceptance criteria (per lesson)
- A human reader from the target group can answer the lesson's opening question in their own words afterwards.
- No term, symbol or notation appears before its introduction in course order (test).
- Every prose number has a role. At most 10 non-`worked` numbers. No sentence has three or more numbers unless allow-listed (test).
- Every widget has a guided task, and the text states the observation.
- The core template of §9 is met (exceptions as listed there).
- No unfair comparison, no test-set peeking, and claims from a single seed are labelled.
- The pedagogy reviewer reports no High findings.
- All existing gates pass: trace drift, Node tests, site check, prose linter, accessibility audit.

---

## 13. Sources

Learning science:
- R. E. Mayer, *Multimedia Learning* (3rd ed., 2020); summary handout:
  [Mayer 2020](https://ugc.futurelearn.com/uploads/files/7d/d6/7dd6188d-c343-4311-b064-ac98d2c95abc/Multimedia_Principles._R._E._Mayer__2020.pdf).
- P. Ginns, A. Martin, H. Marsh (2013), "Designing instructional text in a conversational style: a meta-analysis",
  *Educational Psychology Review*: [record](https://researchers.westernsydney.edu.au/en/publications/designing-instructional-text-in-a-conversational-style-a-meta-ana/);
  2024 replication: [Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1405308/epub).
- D. Perkins (2009), *Making Learning Whole*: [Harvard Project Zero](https://pz.harvard.edu/resources/making-learning-whole-how-seven-principles-of-teaching-can-transform-education).
- E. Fyfe, N. McNeil, J. Son, R. Goldstone (2014), "Concreteness fading in mathematics and science instruction: a systematic
  review", *Educational Psychology Review* 26: [summary](https://pc.cogs.indiana.edu/?p=885).
- L. Richland, N. Kornell, L. Kao (2009), "The pretesting effect", *J. Exp. Psych.: Applied*; S. Carpenter, A. Toftness (2017),
  "The effect of prequestions on learning from video presentations", *J. Applied Research in Memory and Cognition*.
- C. Crouch, A. Fagen, J. Callan, E. Mazur (2004), "Classroom demonstrations: learning tools or entertainment?", *American Journal
  of Physics* 72.
- M. Kapur, productive failure; T. Sinha, M. Kapur (2021), "When problem solving followed by instruction works: evidence for
  productive failure", *Review of Educational Research* (meta-analysis); [Springer study on observing failure](https://link.springer.com/doi/10.1007/s11251-020-09528-z).
- S. Ainsworth (2006), "DeFT: a conceptual framework for considering learning with multiple representations", *Learning and
  Instruction* 16.
- G. D. Rey (2012), "A review of research and a meta-analysis of the seductive detail effect", *Educational Research Review* 7.
- J. Sweller, cognitive load theory and the worked-example effect; S. Kalyuga, the expertise-reversal effect;
  [notes on the worked example effect](https://notes.andymatuschak.org/Worked_example_effect).
- K. Bisra, Q. Liu, J. Nesbit, F. Salimi, P. Winne (2018), "Inducing self-explanation: a meta-analysis", *Educational Psychology
  Review* 30.
- J. Dunlosky, K. Rawson, E. Marsh, M. Nathan, D. Willingham (2013), "Improving students' learning with effective learning
  techniques", *Psychological Science in the Public Interest* 14.
- N. Schroeder, A. Kucera (2022), "Refutation text facilitates learning: a meta-analysis of between-subjects experiments",
  *Educational Psychology Review* 34.

Explanation design:
- N. Case, [notes on explorable explanations](https://blog.ncase.me/explorable-explanations); B. Victor, "Explorable
  Explanations" (2011, worrydream.com); [Red Blob Games notes](https://redblobgames.com/x/2306-educ432).
- C. Olah, S. Carter (2017), ["Research Debt"](https://distill.pub/2017/research-debt/), Distill.

ML education:
- M. Valdenegro-Toro, M. Sabatelli, ["Machine Learning Students Overfit to Overfitting"](https://proceedings.mlr.press/v207/valdenegro-toro23a.html).
- [Data-driven teaching and 9th graders' ML conceptions (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/41511):
  workflow knowledge improved, but ideas about how models work inside persisted. Verify before quoting.
- [Lessons learned teaching ML and NLP to high school students (AAAI 2020)](https://mlanthology.org/aaai/2020/norouzi2020aaai-lessons).

Resources to study (not to copy):
- 3Blue1Brown, *Neural networks* series (visual intuition; order: network → gradient descent → backprop).
- A. Karpathy, *Neural Networks: Zero to Hero* (build from scratch; micrograd).
- M. Nielsen, *Neural Networks and Deep Learning* (narrative prose).
- TensorFlow Playground (decision boundaries).
- CNN Explainer.
- MLU-Explain.
- d2l.ai.
- fast.ai (top-down).

Some research citations (Richland, Carpenter, Crouch, Sinha & Kapur, Ainsworth, Rey, Bisra, Dunlosky, Schroeder & Kucera) were
added from the reviewers' and the author's knowledge, not from fetched pages. Check the details before quoting them on the site.

---

## Appendix A. Pedagogy reviewer checklist

Answer each with evidence (quotes) and rank findings High, Medium or Low.

1. Does the lesson open with questions the persona would ask, in a concrete situation?
2. Is the prediction committed, and placed before any text or heading that answers it? Is a wrong option plausible?
3. Is every term and symbol introduced before use, in course order? Is the notation consistent (one letter, one name)?
4. Does each idea go picture → numbers → symbol → (optional) code? Does the text work without the widgets?
5. Does every number have a role, and does it meet §4?
6. Does every widget have a guided task, with the observation stated?
7. Is there a worked example with a learner step?
8. Is a misconception refuted with evidence from the page?
9. Are comparisons fair? Is the test set left untouched? Are single-seed claims labelled? (C-type errors)
10. Is TorC engineering kept out of the concept text?
11. Does the lesson close by answering its questions, and point to the next question?
12. Would a cold reader be able to explain the main idea in one sentence afterwards? Quote where they would stall.
13. Style a linter cannot see: "just" as a hedge, three-item lists for rhythm, punchline sentences, decorative metaphors.

## Appendix B. Gold-standard example: `why-gradients`, core section

This shows the target quality. It replaces "The model" and §1–2 of today's page.

**Bindings.** Every digit on the page must be bound, including quiz options and reveals. Today's test
`why-gradients.test.mjs:166` fails on any hand-typed digit. In the text below, `{{path|role}}` marks a `T()` binding with its
role (§4). Real paths in `why-gradients.json`, each prefixed `files.why-gradients.`:

| Value | Path |
|---|---|
| 2, 6, 3 | `setup.x`, `setup.t`, `setup.w_min` |
| w = 0, L = 36, gradient −24 | `probes.0.w`, `probes.0.L`, `probes.0.grad` |
| nudge 0.01; new loss 35.76; change −0.2396; predicted −0.24 | `nudge.delta`, `nudge.stages.2.after`, `nudge.actual_dL`, `nudge.predicted_dL` |
| nudge sizes 1, 0.5, 0.1, 0.01, 0.001 and their slopes −20 … −23.99 | `secants.0–4.h`, `secants.N.slope` |
| lr 0.05; 1.2; −14.4; 1.92 | `trajectories.0.lr`, `trajectories.0.states.1.w`, `trajectories.0.states.1.grad`, `trajectories.0.states.1.w_next` |
| lr 0.25; 6; back to 0 | `trajectories.3.lr`, `trajectories.3.states.1.w`, `trajectories.3.states.2.w` |

The keys are `states`, not `steps`. There is no `nudge.L_after` or `nudge.dL`. These values are not in the trace yet: the step
sizes 1.2 and 0.72, the distances 1.8 and 1.08, the shares 60 %, 100 % and 140 %, and the 0.02 / 5.98 of the nudge. Add trace
fields for them, or bind computed values with a test (precedent: `sci()` in `linear-layer.js`). The four-notebook numbers come from
the new steep-bowl scene (★).

**Arithmetic, checked twice.** L = (2w − 6)² = 4(w − 3)², so dL/dw = 8(w − 3). One step with lr keeps the fraction 1 − 8·lr of the
distance to 3: 0.6 for lr 0.05, −1 for lr 0.25, −1.4 for lr 0.3. The steep bowl is L = (4w − 12)² = 16(w − 3)², so its gradient
at w = 0 is −96.

> ### Questions this lesson answers
> How does a computer know which way to change a weight? How big a change should it make?
>
> ### The price of a notebook
> Two notebooks cost 6 euros. The model must learn the price of one notebook. It has one weight, `w`, its guess of the price.
> Its prediction for two notebooks is 2 × `w`. The **loss** measures how wrong the prediction is. It is the miss, squared:
> (2`w` − 6)². The model starts with the guess `w` = {{probes.0.w|setup}}. It predicts 0 euros, misses by 6, and the loss is
> {{probes.0.L|worked}}.
>
> You can find the price by dividing 6 by 2. A network with 26 506 weights has no such shortcut. So we practise the general
> method on a problem with a known answer.
>
> **[quiz: numeric, before any explanation, tolerance 0.01]** Raise `w` to 0.01. The prediction is 0.02, and the miss is 5.98.
> What is the new loss?
> *Reveal:* 5.98² = 35.76. The loss fell by about 0.24. That is 24 times the nudge.
>
> ### The gradient
> Try a nudge ten times smaller, and the loss changes about ten times less. The ratio stays near −24. This ratio is the
> **gradient**. It tells how much the loss changes for each unit that you move `w`, at the point where you are. It is the slope
> of the loss curve at that point. A negative gradient tells you that a larger `w` gives a smaller loss.
>
> **[widget: nudge lens]** *Task:* set the nudge to 0.5, then 0.1, then 0.01. *What you will see:* the curve and its straight
> tangent line come closer together. So the prediction "gradient × nudge" becomes more exact.
>
> ### One step
> Training changes `w` by the gradient times a small number, in the opposite direction. That small number is the
> **learning rate**, lr.
>
> new `w` = `w` − lr × gradient = 0 − {{trajectories.0.lr|worked}} × ({{probes.0.grad|worked}}) = **{{trajectories.0.states.1.w|worked}}**
>
> A minus sign before a negative number adds. So `w` goes up, toward the true price.
>
> `L.backward();` is the line that makes TorC find the gradient and store it in `w.grad()`.
>
> **Your turn [quiz: numeric, tolerance 0.01].** At `w` = 1.2 the gradient is {{trajectories.0.states.1.grad|setup}}. Where does
> the next step land?
> *Reveal:* 1.2 − 0.05 × (−14.4) = 1.92. The first step moved `w` by 1.2, and this step moves it by only 0.72. The steps get
> shorter by themselves, because the curve is flatter near the bottom. Each step keeps 60 % of the distance to 3. The distance
> goes from 3 to 1.8 to 1.08.
>
> ### A tempting idea that is wrong
> "A gradient of −24 means that the answer is far away." Suppose instead that four notebooks cost 12 euros. The true price is
> still 3, and the start is still 0, so the distance is the same. But the gradient at the start is now
> {{steep.probes.0.grad|contrast}}★, four times larger. The miss is larger here, and each nudge of `w` now changes the prediction
> four times as much. Both make the ground steeper. The distance to the answer is unchanged. So the gradient tells you how steep
> the ground is where you stand. It does not tell you how far away the bottom is.
>
> *Analogy.* You stand on a hillside in thick fog. You can feel which way the ground slopes under your feet, and how steeply.
> You cannot see the valley. *Where it breaks:* a hiker remembers the path. Plain gradient descent does not. The lesson
> "Better optimizers" adds that memory.
>
> **[widget: bowl with fog mode]** *Task 1:* turn the fog on and reach the bottom in as few steps as you can with lr 0.05. Then
> lift the fog. *Task 2:* before you press Step with lr 0.25, decide where `w` will land. *What you will see:* it lands as far
> past the bottom as it started before it. With lr 0.25, each step keeps 100 % of the distance but jumps to the other side. With
> lr 0.3 it keeps 140 %, so the jumps grow.
>
> **[In TorC, collapsed]** `SGD::step()` does the subtraction above, in one line:
> `param->data() = param->data().sub(grad.mul(lr_));`
>
> ### The answer
> The computer cannot see the whole loss curve. At its current point, it measures the gradient. The gradient says which way is
> downhill and how steep it is. Then the computer takes a step against the gradient, scaled by the learning rate. If the
> learning rate is too large, each step overshoots by more than it gains. So you try a few learning rates and watch the loss.
> *Next question:* what if the model has thousands of weights, not one?

Why this meets the bar:
- The learner commits to an answer first, and computes it rather than guessing.
- It answers the obvious objection ("why not just divide?").
- Every number is a setup value, part of a worked calculation, or a contrast.
- No symbolic differentiation is needed.
- The misconception is refuted with evidence, and the obvious misreading of that evidence is blocked.
- The text works without the widgets, and each widget has a task.
- The one "real line" moment (`L.backward()`) stays visible. The implementation line is in the panel.
- There is one analogy, and it states its limit.
- It answers both opening questions and asks the next one.
- It is about 700 words. With the gradient-as-vector material moved to `sgd-step`, the whole lesson fits 12–15 minutes.
