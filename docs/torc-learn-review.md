# TorC Learn: pedagogical review and rework brief

**For:** the next session working on the `web` branch (TorC Learn, https://ivorlipic.github.io/TorC/).
**Reviewed:** `origin/web` @ `3b06118` (13 published lessons + Backprop lab), 2026-10-21.
**Question asked by the owner:** how good is this material *for learning*? The target is a person who just finished high school
and wants to understand how ML/DL works, and the site should be the best resource for them.

Status: DRAFT v1 (will be revised after independent reviews).

---

## 0. How to use this document

1. Read §1 (verdict) and §11 (work plan) first. They are the contract.
2. The `web` branch's own rules still apply: read `CLAUDE.md`, `docs/authoring.md`, `docs/decisions.md` on `web` before editing.
   Several of the changes below **revise owner decisions** (7, 9, 12 in `docs/decisions.md`). Do not silently override them:
   record each revision as a new numbered decision, citing this review, and get the owner's OK where §11 marks it.
3. §10 lists concrete defects that can be fixed immediately without any design discussion.
4. Every numbered recommendation has an id (`R1`, `R2`, ...) so commits and PRs can reference it.

---

## 1. Verdict

**What is excellent and must be kept.**
- The correctness pipeline is unusually good: every number comes from a real run of the library, prose numbers are bound to trace
  data with tests, the JS engine is checked against C++ output, KaTeX is validated, a11y is audited. Almost no learning site has this.
- Lessons are segmented into short, user-paced sections, each with a heading that names its topic (Mayer's segmenting principle).
- Predict-then-reveal prompts exist in every lesson. This is the single most effective pattern on the site.
- Several widgets are genuinely good: the bowl with tangent and step presets, the matmul grid with the "paper work" panel, the
  with/without ReLU curve fit, the optimizer race, the digit pad with a real forward pass.
- It never lies. Claims are precise and verified.

**What keeps it from being the best resource for a high-school graduate.**
The site is optimized for *verifiable correctness* and *lint-able style*, not for *understanding*. Every quality gate (trace binding,
STE linter, word budget, independent reviewer) checks something other than "did the learner understand it". The result reads like a
precise, well-tested tour of the TorC API with the math next to it. Concretely:

1. **Wrong audience.** `docs/content-plan.md` §1 targets "programmers and STEM students who know calculus vocabulary and some C++ or
   NumPy". The owner's target (a high-school graduate) has no C++, no matrices, no partial derivatives, no Σ fluency. The first
   thing they see in lesson 1 is `Variable w(Tensor({0.0f}, {1}), true);`.
2. **No "whole game" and no "why".** There is no lesson that says what machine learning is, what a model is, what "learning from
   data" means, or why anyone wants a gradient. Lesson 1 starts with "This lesson uses a model with one weight."
3. **Numbers narrated instead of ideas explained** (the owner's observation, confirmed). About 130 sentences across 13 lessons
   carry two or more numbers, roughly ten per lesson ("With `lr` = 0.05, one step moves `w` from 0 to 1.2. The loss falls from 36
   to 12.96."). Most of them prove the site is accurate. They do not help the learner understand. See §4.
4. **The writing standard works against learning.** ASD-STE100 is a controlled language for aircraft maintenance manuals. It
   produces flat, voice-less prose with no analogies, no motivation and no story. Mayer's *personalization principle*
   (conversational style beats formal style, one of the larger effects in multimedia-learning research) points the other way. See §7.
5. **The core idea of deep learning is not on the main path.** Backpropagation through a network gets one section in lesson 1 and an
   optional lab outside the lesson order. Classification is never shown visually (no decision boundary anywhere). The gradient as a
   *vector* over many parameters is never taught; lesson 2 silently goes from one parameter to two.
6. **Library engineering on lesson pages.** A whole lesson (`modules`) and parts of others teach TorC specifics: `parameters()`
   order from an `unordered_map`, the forward-cache lifetime rule, `zero_grad` accumulation, `reshape` copying, MSELoss not checking
   shapes. These matter to someone using TorC. They are noise for someone learning how DL works, and they cost scarce attention.
7. **Too short to teach the subject.** Thirteen lessons at 6 to 10 minutes each is about 95 minutes for "from a gradient to a CNN".
   A realistic, deep first pass for this learner is 6 to 10 hours. The 800-word cap and the 2-widget cap force density, and
   density is the enemy of a novice (cognitive load).
8. **Checks test recall more than understanding**, and "I got it" is self-reported. Several checks ask for a number that is only
   on the page ("Which digit does the CNN read worst?" → "7, with 998 of 1028"). None revisit earlier lessons (no spacing, no
   interleaving).
9. **It stops abruptly at a CNN.** A 2026 high-school graduate wants to know how ChatGPT works. The course never connects what it
   taught to modern models, even though the same loop (loss, gradient, optimizer) is exactly what trains them.

**Bottom line.** The engineering foundation is a real competitive advantage. The content layer needs a rework of audience, structure
and voice, not polish. The good news: most widgets, all of the trace tooling and most of the facts can be reused.

**Top 10 changes, in priority order** (details in later sections):

| # | Change | Section |
|---|---|---|
| R1 | Re-target the course at the high-school graduate; state prerequisites ("algebra and graphs; we teach the rest") | §2 |
| R2 | Adopt the "numbers serve claims" rule; remove number-narration from prose; move trajectories into widgets | §4 |
| R3 | Replace ASD-STE100 with a plain, warm, conversational teacher voice (keep short sentences and defined terms) | §7 |
| R4 | Add lesson 0 "What does it mean for a machine to learn?" (the whole game, a demo first) | §5 |
| R5 | Put backpropagation and the gradient-as-vector on the main path; add a 2D classification lesson with decision boundaries | §5 |
| R6 | Make concepts the main layer and TorC code a second layer ("In TorC" panel); move TorC engineering to an appendix track | §5, §6 |
| R7 | Every lesson opens with a question or a problem, uses concreteness fading (picture → numbers → symbols → code) | §3, §7 |
| R8 | Real checks: commit an answer before reveal, feedback per wrong answer that names the misconception, cumulative review | §8 |
| R9 | Add just-in-time math refreshers (derivative, Σ, vectors/dot product, matrix product, exp/log, probability) and a glossary | §8 |
| R10 | Add an epilogue "From digits to ChatGPT" and change the QA process so pedagogy is reviewed, not only correctness | §5, §9 |

---

## 2. Who the learner is (R1)

**Persona.** Age 17 to 19, just finished high school. Knows: algebra, functions and their graphs, exponents and logarithms, basic
probability, maybe one term of calculus (derivative as slope, shaky). Does not know: vectors and matrices, Σ notation beyond a
glimpse, partial derivatives, C++, what a tensor is. May know a little Python. Motivation: "AI is everywhere; how does it actually
work?" They will quit when they feel lost or bored, and they will not tell you which.

**Consequences for the whole course.**
- State the prerequisites on the home page and in lesson 0: "You need algebra and graphs of functions. Everything else, including
  the derivative and matrices, is explained when it is needed."
- Never use a symbol before it is introduced in words. ∂, Σ, overline-gradient notation, ᵀ, ⌊ ⌋, `σ`, `\hat m` all appear today
  without an introduction suited to this learner.
- C++ is optional depth, not the entry point (R6). The learner should be able to complete the whole course without reading C++
  and still understand how DL works. Readers who want to see the real code open the "In TorC" panel.
- Use concrete, meaningful data before abstract data: house prices, exam scores vs study hours, cats vs dogs as 2D points,
  handwritten digits. Abstract toys (`t = 2x + 1`, a narrow quadratic valley) are fine as second examples.
- Update `docs/content-plan.md` §1 ("Who") to this persona and keep the old audience as a secondary one served by the TorC layer.

---

## 3. Teaching principles to adopt (research basis → rule for this site)

Each principle below has a concrete rule that a reviewer can check. Sources in §12.

| Principle | Evidence (short) | Rule for TorC Learn |
|---|---|---|
| **Play the whole game first** (Perkins, *Making Learning Whole*; fast.ai's top-down approach) | Learners who meet a whole, authentic version of the activity early are more motivated and see why the parts matter | Lesson 0 shows a working digit recognizer and the full loop (data → model → loss → adjust) before any part is taught. Each later lesson says where it sits in that loop |
| **Concreteness fading** (Fyfe, McNeil, Son, Goldstone 2014) | Concrete → pictorial → abstract beats concrete-only or abstract-only for transfer | Each concept appears in this order: a picture or a thing you drag; then the same with numbers; then the symbol; then (optional) the TorC code |
| **Pretraining** (Mayer) | Learners do better when they know the names and traits of key parts before the main explanation | Each lesson starts with a 2 to 4 item "Words you will meet" list with one-line plain definitions and an icon or picture; a site-wide glossary with hover |
| **Coherence** (Mayer) | Removing interesting-but-unneeded material improves learning | Remove library quirks, error strings and implementation details from the concept layer (they go to "In TorC" or the appendix) |
| **Signaling** (Mayer) | Cues that show the structure help | Each lesson has a visible "the question" at the top and "the answer" at the end; key sentence of each section is bold; the training-loop diagram reappears with the current part highlighted |
| **Personalization** (Mayer; conversational style) | Conversational, second-person wording beats formal wording | Write to "you", ask questions, use "we" for shared exploration; see §7 |
| **Segmenting** (Mayer) | User-paced chunks beat a continuous presentation | Keep the short sections; allow more of them (§6 budget) |
| **Prediction / generation before explanation** (productive failure, Kapur; predict-observe-explain) | Attempting a problem before instruction improves conceptual understanding when learners have enough prior knowledge to try | Keep "Before you try it", but require a committed answer (choice, number or drag) before the reveal; place predictions *before* the explanation, not after it |
| **Worked examples, then faded practice** (cognitive load theory) | For novices, studying a worked example beats solving the same problem cold; fade support as fluency grows | Each lesson has one fully worked example (every step shown), then a near-identical one with a step missing, then one the learner does |
| **Retrieval, spacing and interleaving** | Recalling earlier material later, mixed with new material, improves long-term retention | Each lesson's checks include one question from an earlier lesson; a short "review" set appears every three lessons |
| **Refutation of misconceptions** | Naming a misconception, then showing why it is wrong, changes beliefs more than just stating the correct idea | Each lesson names 1 to 2 common misconceptions and refutes them with the widget (list in §6) |
| **Explorables are explanations, not sandboxes** (Bret Victor, Nicky Case) | A bare sandbox does not teach; build each mechanic separately, then combine; the text should work even if the widget is skipped | Every widget comes with 1 to 3 guided tasks ("Set lr to 0.3. What happens to the dot?") and an expected observation in the text; build mechanics one at a time |

---

## 4. The "x goes to y in z steps" problem (R2)

**Diagnosis.** The owner is right. Sentences like these appear about ten times per lesson:

> With `lr` = 0.05, one step moves `w` from 0 to 1.2. The loss falls from 36 to 12.96. *(why-gradients)*
> With μ = 0.5, the loss is 0.174 after two steps, and 1.09 without momentum. At step three it rises to 2.447. *(sgd-step)*
> Under AdamW, it falls from 3 to 1.815 in 50 steps. *(optimizers)*
> The test loss is lowest after step 200, at 0.05669. Then it rises to 0.1281, and the training loss ends at 0.00008427. *(generalization)*
> After the first epoch, the model reads 9204 of the test images correctly. After the last epoch, it reads 9666 correctly. *(mnist-mlp)*

Why they do not help:
- The learner has no reason to care about 12.96 or 1.815. A number is meaningful only if the learner predicted it, computed it, or
  can compare it with something they already understand.
- They cost working memory. Each number is a thing to hold while reading the next sentence, and novices have little to spare.
- They replace the explanation. "The loss falls from 36 to 12.96" says *that* it worked; it does not say *why*.
- Over-precision signals noise: 0.00008427, 0.001145, 0.05669. Four significant digits on a loss value carries no meaning for this reader.

**Root cause in the process.** `CLAUDE.md` says "Numbers on lesson pages come from `tools/trace` output... Prose that quotes a
number needs a test that checks it." The `T()` binding and the budget ("≤ 25 numbers bound in the text") make a number on the page
look like proof of quality. Authors fill prose with bound numbers to show that the page is grounded. The rule is good for honesty
but it has become an incentive.

**New rule ("numbers serve claims").** A number may appear in lesson prose only if one of these holds:
1. The learner is asked to predict or compute it, and it appears in the reveal.
2. It is an anchor the learner will remember and reuse: 784 pixels, 10 classes, about 97 % accuracy, 26 506 parameters vs 9 098.
3. It makes a comparison where the size of the difference is the point, and the sentence states the point in words first:
   "With momentum the loss drops much faster at first (0.17 after two steps instead of 1.09), but then it overshoots."
Everything else (trajectories, per-step values, final losses of runs) goes into the widget readout, where the learner sees it
while doing something. Round to 2 to 3 significant figures in prose; use percentages for accuracy; use a thin space or comma for
thousands consistently (the home page says "26,506", lessons say "26506").

Keep `T()` for every number that remains; the honesty guarantee stays. Lower the prose budget to **≤ 10 bound numbers per lesson**
(today ≤ 25) and add a test heuristic: flag any sentence with three or more bound numbers for a human look.

**Before/after examples** (the next session should rewrite every such sentence in this spirit):

| Before | After |
|---|---|
| "With `lr` = 0.05, one step moves `w` from 0 to 1.2. The loss falls from 36 to 12.96. Each step is shorter than the one before, because the gradient gets smaller near the lowest point." | "Press **Step** a few times and watch the dot. Each step is shorter than the one before. Why? The step is the learning rate times the slope, and the slope gets flatter as you approach the bottom. So gradient descent slows down by itself near the minimum, without anyone telling it to." |
| "Under AdamW, it falls from 3 to 1.815 in 50 steps." | "Weight decay shrinks every weight a little at each step, like a slow leak. A weight that the loss does not care about drifts toward zero; a weight that matters is held up by its gradient." (the 3 → 1.815 curve is drawn in a widget) |
| "After the first epoch, the model reads 9204 of the test images correctly. After the last epoch, it reads 9666 correctly." | "After one pass through the training images the network already gets 92 % of new digits right. Nineteen more passes bring it to about 97 %. Most of the learning happens early; the rest is slow polishing." |
| "The test loss is lowest after step 200, at 0.05669. Then it rises to 0.1281, and the training loss ends at 0.00008427." | "Look at the two curves. The training loss keeps falling almost to zero. The test loss falls at first, then turns around and climbs. From that turning point on, the model is memorizing the noise in its ten training points, not learning the curve." |
| "It takes 0 steps. After the first epoch, `has_next()` returns `false`, so the inner loop does not run. TorC gives no error, and the loss stays at 0.8618." | "None. The loader has already handed out every batch, so the inner loop never runs again, and the model silently stops learning after the first epoch. No error tells you." |

---

## 5. Structure and progression (R4, R5, R6, R10)

### 5.1 Problems with the current sequence

Current order: 01 The gradient · 02 Using the gradient · 03 Tensors · 04 Layers · 05 Nonlinearity · 06 Modules · 07 Losses ·
08 The training loop · 09 Better optimizers · 10 Generalization · 11 Digits with an MLP · 12 Convolution · 13 Digits with a CNN ·
(Lab) Backprop lab.

- **No lesson 0.** Nothing says what ML is, what a model, data, a prediction or a loss is in general, or why we want gradients.
- **Used before taught** (prerequisite violations):
  - Lesson 2 uses "the mean squared error, `nn::MSELoss`" and a matrix-shaped `W·x + b`; MSE is taught in lesson 7, matrices in lesson 3.
  - Lesson 2 teaches momentum; lesson 9 teaches it again. Momentum is premature in lesson 2 (no noisy or narrow-valley loss yet).
  - Lesson 4's init-signal widget uses ReLU ("After each layer, a ReLU sets the negative values to zero"); ReLU is lesson 5.
  - "Neuron" is defined in lesson 4 as "a row of W". The intuitive neuron (weighted vote + threshold) never appears.
  - "Classifier" is first defined inside the softmax section of lesson 5. Classification as a task is never introduced.
  - ∂ appears in lesson 7 with no introduction; lesson 1 uses d/dw. Overline notation for gradients appears in lessons 2 and 4.
  - Lesson 4 derives `W̄ = ȳᵀx` and `x̄ = ȳW`. This is the hardest math on the site and it comes before the learner has trained anything with a layer.
- **The gradient is never taught as a vector.** Lesson 1 has one weight; lesson 2 has two and shows a contour plot, but no section
  says "with many parameters, the gradient is a list of slopes, one per parameter, and together they point uphill".
- **Backpropagation is off the main path.** The heart of "how DL works" is the Backprop lab, explicitly "not part of the lesson order".
- **No visual classification.** The course jumps from fitting lines and curves straight to MNIST. The most intuitive picture in all of
  deep learning (coloured points in 2D and a boundary that bends as hidden units are added, as in TensorFlow Playground) is missing.
- **`modules` is an API lesson.** Its four sections are: what a module is, Sequential, `parameters()` order from an unordered map,
  and the forward-cache lifetime rule. Two of four are TorC limitations.
- **Every lesson has a new toy problem** (a bowl, two different lines, x², a sine polynomial, a valley, digits). There is no
  throughline that lets the learner build on a familiar example.
- **No ending.** The last lesson ends with "This is the last lesson of the course." Nothing connects to modern DL.

### 5.2 Proposed curriculum (v3)

Main track = concepts, each lesson with an optional "In TorC" panel showing the real code. Lesson count goes from 13 to about 18;
most new lessons reuse existing widgets and traces. ★ = new lesson, ◆ = substantially rewritten, ○ = mostly kept.

**Part 0. The whole game**
- ★ **0. What does it mean for a machine to learn?** Open with the digit pad (already built). Then a hand-fitting game: drag a line
  through points (hours studied vs exam score), see a "how wrong" number (the loss) update live, try to make it as small as you can.
  Name the four parts: data, model, loss, adjust. Show the loop diagram that recurs in every lesson. Promise: "By the end you will
  know exactly what happens inside this digit recognizer, and what is different in ChatGPT."

**Part 1. Learning is making the loss small**
- ◆ **1. Models and losses.** A model is a function with knobs (parameters). The loss measures how wrong it is on the data.
  Introduce MSE here, visually (squares on the residuals). Reuses the hand-fitting game.
- ◆ **2. The gradient** (today's lesson 1, rewritten). Derivative as slope, with a just-in-time refresher. The bowl widget.
  No C++ in the main text.
- ◆ **3. Gradient descent** (today's lessons 1 §2-3 + 2, merged). Step against the slope; learning rate too small, right, too big.
  Then **two parameters**: the gradient as an arrow on a contour map (reuse the `contour2d` widget), "a list of slopes that
  together point uphill". Drop momentum and `zero_grad` from here.

**Part 2. How the computer finds the gradient**
- ◆ **4. The chain rule and computation graphs** (today's lesson 1 §4 + Backprop lab scenes 1-2 moved onto the main path). Gears
  analogy: if p moves 2× as fast as w, and L moves −12× as fast as p, then L moves −24× as fast as w. Reuse of a value adds
  contributions. `backward()` appears here as "the computer does this for every parameter at once".

**Part 3. Neural networks**
- ★ **5. Vectors, the dot product and the matrix product** (just enough linear algebra; today's `tensors` §4 + the matmul grid).
  Dot product as a weighted vote / similarity score. Matrix product as "many dot products at once". Shapes as a bookkeeping tool.
  Broadcasting and reductions move to the "In TorC" panel or the appendix.
- ◆ **6. The neuron and the layer** (today's `linear-layer` §1-2). One neuron = weighted sum + bias, drawn as a node with
  incoming arrows; then a layer = many neurons = one matrix product. Parameter counting. Initialization moves to lesson 10.
- ◆ **7. Nonlinearity** (today's lesson 5, extended). Keep "two linear layers collapse". Add the key intuition that is missing
  today: each ReLU neuron makes one bend; a sum of bends can draw any curve. Show the hidden neurons' individual hinge functions
  in the curve-fit widget and how they add up.
- ★ **8. Classification** (today's lesson 7 §2-3 + new widget). Points of two colours in 2D. A neuron with sigmoid draws a straight
  boundary; with hidden ReLU neurons the boundary bends (a small Playground-style widget, trained live in JS, or precomputed from
  TorC runs). Softmax turns scores into shares; cross-entropy measures surprise; why it beats squared error for classes.
- ◆ **9. Backpropagation through a network** (today's lesson 4 §3 + Backprop lab scene 3). The layer's backward pass, done once
  with numbers, then as the matrix rule. Vanishing gradients with sigmoid (from today's lesson 5 §2).
- ◆ **10. Starting the weights** (today's lesson 4 §4). Why not zeros (symmetry, the "copy neuron 1 into neuron 2" button), why
  not too big or too small (the six-layer signal plot, now after ReLU is taught).

**Part 4. Training in practice**
- ○ **11. The training loop** (today's lesson 8). Batches, epochs, the noise of small batches. `zero_grad` belongs here, in the
  "In TorC" panel, as the reason each iteration clears the gradient.
- ○ **12. Better optimizers** (today's lesson 9). Momentum taught once, here, with the ball-rolling analogy; Adam; AdamW.
- ○ **13. Generalization** (today's lesson 10). Add the memorize-vs-understand analogy and a validation-set widget, not only text.

**Part 5. Images**
- ○ **14. Digits with an MLP** (today's lesson 11). Accuracy in percent; add "what the hidden neurons respond to" and a
  confusion matrix (which digits get mixed up and why).
- ◆ **15. Convolution** (today's lesson 12). Motivate it: shift a digit 3 pixels and watch the MLP's answer change, while a
  pattern detector that slides does not care where the stroke is. Then kernels, stride, padding, channels.
- ○ **16. Digits with a CNN** (today's lesson 13).

**Part 6. Where this goes next**
- ★ **17. From digits to ChatGPT** (concept only, no TorC code; honest that TorC stops at CNNs). Text as tokens, tokens as vectors
  (embeddings), predicting the next token with cross-entropy, attention as "each word looks at the others and takes a weighted
  average", and the punchline: it is trained with exactly the loop of lesson 11, at a vastly larger scale. Ends with a curated
  "where to go next" list (3Blue1Brown's neural network series, Karpathy's *Zero to Hero*, Nielsen's *Neural Networks and Deep
  Learning*, d2l.ai).

**Appendix track: "Using TorC"** (outside the main order, like the labs today). Today's `modules` lesson, tensor storage and
`reshape` copies, broadcasting rules in full, `zero_grad` and gradient accumulation, graph lifetime ("why did `backward()`
throw?"), label checks in `CrossEntropyLoss`, MSELoss shape pitfall. These pages can keep today's precise, API-first style.

**Labs**: keep the Backprop lab as a free playground; add a "Train a tiny network" playground later.

**Throughline.** Use one recurring example in Parts 0 to 3 (for example hours studied → exam score for regression, and
pass/fail from two features for classification), so the learner meets each new idea on familiar data before seeing it elsewhere.

**Time.** Plan for 10 to 20 minutes per lesson, about 4 to 6 hours in total. State honest times on the cards.

---

## 6. Lesson-by-lesson notes

For each published lesson: keep (what works), fix (what hurts learning), and the misconception to refute.

**01 The gradient (`why-gradients`)**
- Keep: the bowl widget with the tangent line; the lr presets including 0.25 (bounces) and 0.3 (diverges); "the gradient does not
  tell you how far away the lowest point is" (a great refutation); the chain-rule product with numbers.
- Fix: opens with C++ before the idea; the model has no meaning (why multiply by 2 and target 6?). Give it a story ("a price
  per item; you saw 2 items cost 6"). "Local derivative" needs a picture. The §3 formula `w_{k+1} - 3 = f (w_k - 3)` is a detour
  into convergence analysis; replace with the widget and one sentence. The "lr = 0.125 lands exactly" check teaches a fact that
  the lesson itself calls special to this bowl; replace with a transfer question.
- Misconceptions: "the gradient points to the minimum"; "a smaller learning rate is always better".

**02 Using the gradient (`sgd-step`)**
- Keep: the contour landscape with the step trail; the "four times the learning rate is not four times faster" prediction (excellent).
- Fix: uses MSE and `W·x + b` before they are taught; momentum and `zero_grad` belong in later lessons; "a learning rate above about
  0.1803 is too large" is an unexplained magic number (Hessian eigenvalue). Teach the gradient as a vector (arrow on the contour).
  The closing link says "The next lesson, Layers" but the next lesson is Tensors (§10).
- Misconceptions: "the optimizer computes the gradient"; "the loss must fall at every step".

**03 Tensors (`tensors`)**
- Keep: the shape-from-the-right broadcasting rule with the error message as a predict; the axis explorer.
- Fix: for this learner, "tensor" needs a picture ladder (number → list → table → stack of tables, with an image as the
  example of a 3D tensor). Row-major storage, `reshape` copying and `transpose({1, 0})` argument meaning are TorC engineering
  (appendix). Matrix product needs intuition (dot product as weighted vote) before the index formula.
- Misconception: "a tensor is something physics-y and deep"; "shape (2) and (2, 1) are the same".

**04 Layers (`linear-layer`)**
- Keep: the matmul grid with paper work; "copy neuron 1 into neuron 2" (the best symmetry demonstration I have seen); the
  six-layer signal plot.
- Fix: the hardest content of the course (backward matrix rules, Kaiming init) is packed into a 7-minute lesson before
  nonlinearity. Split per §5.2 (lessons 6, 9, 10). The formula in §1 renders with literal semicolons (§10). The grid labels count
  from 1 while the course says positions count from 0, and the check "increase `b[2]`" refers to a bias that does not exist under
  0-indexing (§10). The page ends with "Go back to the course map" instead of the next lesson.
- Misconceptions: "W is (in, out)"; "a neuron is a biological thing inside the computer"; "zero weights are a fine start".

**05 Nonlinearity (`nonlinearity`)**
- Keep: the algebra that collapses two linear layers; the with/without-ReLU fit with the horizontal-line reveal (memorable).
- Fix: show *why* ReLU lets the model bend: plot each hidden neuron's hinge and their weighted sum. The softmax section is
  out of place (it belongs with classification). The overflow trick (subtract the max) is implementation detail.
- Misconception: "the activation function is what makes the network smart"; "more layers alone make a model more powerful".

**06 Modules (`modules`)**
- Move to the "Using TorC" appendix. Keep a two-sentence version of "a model is a list of layers with parameters" in lesson 6.
- The parameter-count viewer is useful; reuse it in lessons 6 and 14.

**07 Losses (`losses`)**
- Keep: "cross-entropy gives a strong gradient when very wrong, squared error does not" with the comparison widget; the uniform
  logits check (ln 3).
- Fix: "TorC does not check that `y` and `t` have the same shape ... gives the loss 3.333 instead of 0.1667" is a library pitfall
  on a lesson page (violates decision 4 in spirit); move it to the appendix. Introduce cross-entropy as "surprise": −log of the
  probability you gave the right answer (100 % → 0 surprise, 1 % → large). Introduce ∂ before using it or keep d/d notation.
- Misconception: "the loss is the accuracy".

**08 The training loop (`training-loop`)**
- Keep: the batch viewer; the batch-size curves; "the loss does not go to zero because the data has noise" (very good).
- Fix: the code shows `0.10000000149011612f` (a float artefact from the trace, §10). The `reset()` check is TorC API trivia;
  replace with "why shuffle?" or "what is the cost of a tiny batch?". Explain *why* noisy small-batch gradients are often fine
  (the average direction is still downhill).

**09 Better optimizers (`optimizers`)**
- Keep: the narrow valley race; Adam's first step ≈ lr·sign(g) check (excellent).
- Fix: the race readout is a small monospace table of w1, w2 and loss for each optimizer; for this learner a loss-vs-step plot
  beside the path says more. Add the ball-rolling analogy for momentum and
  "per-parameter step size" analogy for Adam. Do not repeat momentum from lesson 2 (remove it there).

**10 Generalization (`generalization`)**
- Keep: the degree-15 polynomial on 10 points; the "noise floor" check; underfitting with large weight decay.
- Fix: say early what the model is in words ("a curve with 16 adjustable coefficients"); "the powers of x up to 15" is hard for
  this learner. The validation set is only text; give it a widget moment (pick the stopping step by validation loss, then look at
  the test loss once). Analogy: memorizing last year's exam answers vs understanding the subject.

**11 Digits with an MLP (`mnist-mlp`)**
- Keep: the digit pad, the misread test images, the first-layer weight gallery.
- Fix: report accuracy in percent; explain the weight images (what amber/blue mean is in a tiny caption only); add a confusion
  matrix and "which digits get mixed up and why". The "which digit is worst" check is a look-up; replace with a reasoning question
  ("why might 4 and 9 get confused?").

**12 Convolution (`convolution`)**
- Keep: the sliding window, the edge-kernel feature maps, the parameter comparison (6 280 vs 80) as a predict.
- Fix: motivate convolution with a failure of the MLP (shifted digit). "TorC does not flip the kernel" is a footnote for the
  appendix. The output-size formula with ⌊ ⌋ needs a picture first (count the window positions on a strip).

**13 Digits with a CNN (`mnist-cnn`)**
- Keep: feature maps of the trained first layer; the CNN vs MLP comparison.
- Fix: the comparison is confounded (5 epochs vs 20, different learning rates); the page says "with the settings of the two
  examples", which this learner will not parse. Say plainly "the comparison is not perfectly fair, but the CNN wins with a third
  of the parameters". End with the bridge to lesson 17 instead of "This is the last lesson of the course."

**Backprop lab**
- The scenes and math windows are strong. Move scenes 1-2 into lesson 4 and scene 3 into lesson 9; keep the lab as a playground.

---

## 7. Voice and writing (R3, R7)

**Diagnosis.** `docs/authoring.md` makes ASD-STE100 the writing standard. STE was designed for technical manuals read by
non-native speakers under time pressure. Its rules that help here: short sentences, one idea per sentence, defined terms, one name
per thing. Its effects that hurt here: no rhetorical questions, no analogies, no "you" in a story, no motivation, no humour, the
same flat cadence everywhere ("This lesson shows ...", "The next lesson, X, shows ..."). The 2026-10 focus group found the pages
"read like LLM output"; STE fixed the slogans but not the absence of a teacher.

**New voice rules** (replace the "Writing rules" section of `authoring.md`; keep the linter only for the rules that remain):
- Write like a good teacher talking to one curious 18-year-old. Use "you" and "we". Ask the question the reader is about to ask.
- Open every lesson with a question or a problem, not with "This lesson shows". Example: "You can see that this line fits the
  points badly. But how would a computer know which way to move it?"
- One idea per sentence; ≤ 25 words stays as a default, but readability wins over the count.
- Define every term before use, in plain words, with a picture where possible. Keep "one name for one thing".
- Use an analogy for every core idea, and say where it breaks.
- Dashes, colons and semicolons are allowed when they help reading. Ban the actual problems instead: slogans, "X, not Y"
  punchlines, empty transitions, sentences that only narrate numbers.
- End every lesson with "the answer" to the opening question in two or three sentences, and with what the next lesson will
  answer.

**Analogy bank** (starting set; each needs its limit stated):
- Gradient: standing on a hillside in fog, feeling which way the ground slopes under your feet. (Limit: tells direction and steepness here, not where the valley is.)
- Learning rate: the length of your stride. (Too long and you jump over the valley.)
- Momentum: a ball rolling downhill that keeps some of its speed.
- Adam: each parameter gets its own stride length, based on how bumpy its recent path was.
- Chain rule: gears; if gear B turns 2× as fast as A, and C 3× as fast as B, C turns 6× as fast as A.
- Neuron: a weighted vote, with the bias as how easy it is to convince.
- ReLU: a hinge; a sum of hinges can bend a line into any shape.
- Softmax: turning raw scores into shares of 100 %.
- Cross-entropy: surprise when the right answer comes out.
- Overfitting: memorizing last year's exam instead of learning the subject.
- Convolution kernel: a small stencil or flashlight that you slide over the image, asking "is my pattern here?" at every spot.
- Weight decay: a slow leak that pulls every weight toward zero unless the data holds it up.

---

## 8. Interaction design, checks and supporting material (R8, R9)

**Checks.**
- Replace "Show the answer" + "I got it" with a committed answer: multiple choice (3 to 4 options, each wrong option written to
  match a real misconception, with feedback that names it), a numeric input with tolerance, or "drag the dot to where you think
  `w` lands". The reveal opens after the attempt. Store correctness, not self-report, in the progress store.
- At least one check per lesson is a transfer question (new situation, same idea). Today most are look-ups or recall.
- One check per lesson revisits an earlier lesson (spacing, interleaving). After lessons 4, 8, 12, 16 add a short mixed review.
- Add one "explain it in one sentence" prompt per lesson (free text, not graded, shown next to a model answer). Self-explanation
  is one of the strongest generative activities.

**Widgets.**
- Each widget gets 1 to 3 guided tasks in the text with the expected observation. The text must still work for someone who
  skips the widget.
- Relax the "≤ 2 widgets, ≤ 2 controls" cap (decision 7) to "every widget must earn its place with a guided task"; some lessons
  need three small ones.
- Load-bearing new widgets: the hand-fitting game (lesson 0/1), the gradient arrow on a contour map (lesson 3), the hinge
  decomposition in the curve fit (lesson 7), a 2D classification playground with a bending boundary (lesson 8), a shifted-digit
  demo (lesson 15), a confusion matrix (lesson 14). A recurring small "where are we in the loop" diagram highlights the part the
  lesson is about.
- Numeric readouts: 2 to 3 significant figures; units or meaning beside each number ("loss 1.2 (lower is better)").

**Math support.**
- Just-in-time refreshers as collapsible boxes, each 100 to 200 words with a picture: derivative as slope; Σ notation; vectors and
  the dot product; matrix product; exp and log; probability and "adds up to 1"; the normal distribution (for initialization).
- A glossary page with hover definitions on first use in each lesson (one source of truth, e.g. `site/glossary.json`).
- Notation: pick one gradient notation and introduce it once (∂L/∂w in words first: "how much L changes per unit change of w").

**Home page.**
- Lead with the promise and the demo (the digit pad in the hero), not with a list of C++ snippets. Move the code to the lesson
  pages' "In TorC" panels or behind a toggle on the cards.
- State prerequisites and honest total time.
- Keep the current progress ticks and filter.

**Visual design.** Dark-only theme with small grey text is hard to read for long sessions; add a light theme (already in the
content plan's M6) and raise the minimum caption size in widgets.

---

## 9. Process changes (why the site drifted, and how to stop it) (R10)

- **The quality gates do not measure learning.** The `independent-reviewer` agent's nine dimensions are source fidelity, numbers,
  math, coverage, visuals, build, deployment, writing rules and scope. None asks "would the target learner understand this?".
  Add a `pedagogy-reviewer` agent (read-only) with the checklist in Appendix A, and a `cold-reader` agent that role-plays the
  persona of §2, reads the rendered lesson and reports every point of confusion. Run both on every new or changed lesson.
- **Outside readers.** ROADMAP M3 already lists "outside cold reader per lesson" as not done. Make it a real gate: at least two
  people from the target group per part, watching where they stop.
- **Budgets that encourage the wrong thing.** Replace "≤ 25 bound numbers" with "≤ 10" plus the number-narration heuristic (§4).
  Replace the 800-word cap with a per-lesson time target (10 to 20 minutes) and the rule that every paragraph must teach
  (explain a why, set up a widget task, or answer a question).
- **Decisions to revisit** (record new decisions in `docs/decisions.md`): 7 (concise lessons, ≤ 2 widgets), 9 (no visible
  tables: fine, keep), 12 (STE writing). Also the `CLAUDE.md` rule "load the asd-ste100 skill before writing any visible text".
- **Tests to add**: every lesson has an opening question and a closing answer; every term in the glossary used in a lesson is
  linked on first use; no symbol from a "notation list" (∂, Σ, ᵀ, overline) appears before its introduction in the course order;
  no sentence with three or more bound numbers without an allow-list entry.

---

## 10. Defects to fix now (independent of the redesign)

| # | Where | Defect | Evidence |
|---|---|---|---|
| D1 | `site/js/lessons/linear-layer.js:56` | The shape formula renders with literal semicolons (`x ;·; Wᵀ ;+; b ;=; y`): `'\;'` in a single-quoted JS string is just `;`; it needs `\\;` | Visible on the Layers page at desktop and phone width |
| D2 | `site/js/lessons/sgd-step.js:115` | "The next lesson, Layers" but the next lesson in `course.json` is Tensors | `course.json` order |
| D3 | `site/js/lessons/linear-layer.js:105-106` | The page ends with "Go back to the course map." instead of naming the next lesson (Nonlinearity) like every other lesson | rendered text |
| D4 | `site/js/widgets/matmul-grid.js:50-51,73,78,87` and `linear-layer.js:100-102` | The grid labels rows, columns and neurons from 1 (`i=1`, `k=1`, `Σ_{j=1}^{3}`), but `authoring.md` and the Tensors lesson say positions count from 0. The check "increase `b[2]`" names a bias that does not exist under 0-indexing (b has two entries) | widget code and lesson text |
| D5 | `site/js/lessons/training-loop.js:46` (`cf(s.noise_std)`) | The code shows `data::SyntheticRegression dataset(100, 1, 3.0f, 1.0f, 0.10000000149011612f);`: the float32 noise value from the trace is printed as a double. Format it as the shortest float that round-trips (`0.1f`) | rendered code |
| D6 | `site/js/lessons/losses.js` §1 | "TorC does not check that `y` and `t` have the same shape" is a library pitfall on a lesson page (decision 4); move to the appendix | rendered text |
| D7 | `site/js/lessons/modules.js` §3 | `parameters()` order from an unordered map is a library quirk on a lesson page; move to the appendix | rendered text |
| D8 | several lessons | Thousands formatting is inconsistent: home page "26,506"; lessons "26506", "25450", "9380" | rendered text |
| D9 | `mnist-mlp`, `mnist-cnn` | Accuracy reported as counts ("reads 9666 correctly") rather than percent | rendered text |

---

## 11. Work plan for the next session

Order matters: decisions first, then the cheapest high-impact changes, then new lessons.

**Phase A. Decisions and quick fixes (one session)**
1. Fix D1 to D5, D8, D9. Run `node --test site/test/`, `tools/prose/check.sh`, `node tools/check_site.mjs`. Publish.
2. Write new decisions in `docs/decisions.md`: audience (R1), numbers-serve-claims (R2), voice (R3), concept layer + "In TorC"
   layer + appendix (R6), relaxed budgets (§9). **Ask the owner to approve these before Phase B**; they revise decisions 7 and 12.
3. Update `docs/content-plan.md` §1 (persona) and §2 (curriculum v3, §5.2 here), `docs/authoring.md` (voice, number rule, check
   formats, budgets), `CLAUDE.md` (writing rule), and add the `pedagogy-reviewer` and `cold-reader` agents in `.claude/agents/`.

**Phase B. Rewrite existing lessons in the new voice (one lesson per commit)**
- For each lesson: opening question, pretraining words, concreteness fading, analogies, number rule, misconception refutation,
  committed-answer checks, one cumulative check, closing answer, "In TorC" panel for the code.
- Start with the vertical slice: lessons 1 to 3 of v3 (models and losses, the gradient, gradient descent). Run the cold-reader
  and pedagogy-reviewer agents and, if possible, two human readers from the target group. Adjust the rules, then continue.
- Move `modules` and TorC pitfalls to the appendix track (needs a `kind: "appendix"` part in `course.json`, like `"lab"`).

**Phase C. New lessons and widgets**
- Lesson 0 (whole game + hand-fitting game), lesson 5 (vectors and matrices), lesson 8 (classification playground), lesson 17
  (from digits to ChatGPT), math refreshers, glossary. Each new interactive part still gets a trace or a test, per `CLAUDE.md`.

**Phase D. Checks and spacing**
- Committed-answer check component with per-option feedback; progress store records correctness; review sets after each part.

**Acceptance criteria for "best resource" (per lesson)**
- A cold reader from the target group can answer the lesson's opening question in their own words afterwards.
- No term, symbol or notation is used before it is introduced in the course order (test).
- ≤ 10 bound numbers in prose; no sentence with three or more numbers unless allow-listed.
- Every widget has at least one guided task with its expected observation in the text.
- At least one transfer check and one cumulative check.
- The pedagogy-reviewer checklist (Appendix A) passes with no High findings.
- All existing gates still pass (trace drift, Node tests, site check, a11y).

---

## 12. Sources

- R. E. Mayer, *Multimedia Learning* (3rd ed., 2020) and principles summaries:
  [Mayer 2020 handout](https://ugc.futurelearn.com/uploads/files/7d/d6/7dd6188d-c343-4311-b064-ac98d2c95abc/Multimedia_Principles._R._E._Mayer__2020.pdf),
  [Lucidea summary of the 12 principles](https://lucidea.com/blog/mayers-12-principles-of-multimedia-learning).
- D. Perkins, *Making Learning Whole* (2009): [Harvard Project Zero](https://pz.harvard.edu/resources/making-learning-whole-how-seven-principles-of-teaching-can-transform-education),
  [interview](https://www.the74million.org/article/74-interview-author-and-harvard-scholar-david-perkins-on-what-traditional-classroom-teachers-can-learn-from-science-fairs-backyard-sports-whole-game-learning/).
- Fyfe, McNeil, Son, Goldstone, "Concreteness fading in mathematics and science instruction: a systematic review" (2014):
  [summary](https://pc.cogs.indiana.edu/?p=885), [Learning Scientists blog](https://learningscientists.org/blog/2018/2/1-1).
- M. Kapur, productive failure: [Bold Science interview](https://boldscience.org/we-can-take-this-powerful-mechanism-and-deliberately-design-it-to-learn),
  [Springer study on observing failure](https://link.springer.com/doi/10.1007/s11251-020-09528-z).
- Worked examples and retrieval practice under cognitive load theory:
  [Chartered College of Teaching](https://my.chartered.college/?p=3470), [Andy Matuschak's notes on the worked example effect](https://notes.andymatuschak.org/Worked_example_effect),
  [2025 study on retrieval practice and transfer](https://link.springer.com/article/10.1007/s40593-025-00511-8).
- Explorable explanations: [Nicky Case, workshop notes](https://blog.ncase.me/explorable-explanations),
  [Red Blob Games notes on Case and Victor](https://redblobgames.com/x/2306-educ432); Bret Victor, "Explorable Explanations" (2011, worrydream.com).
- C. Olah and S. Carter, ["Research Debt"](https://distill.pub/2017/research-debt/), Distill (2017).
- ML education research: [9th graders' conceptual understanding of ML (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/41511)
  (inner workings conceptions persist without direct engagement with misconceptions);
  [Machine Learning Students Overfit to Overfitting](https://arxiv.org/pdf/2209.03032);
  [Lessons learned teaching ML and NLP to high school students](https://mlanthology.org/aaai/2020/norouzi2020aaai-lessons).
- Benchmarks for "best in class" (to study, not copy): 3Blue1Brown's *Neural networks* series (visual intuition), Andrej
  Karpathy's *Neural Networks: Zero to Hero* (build from scratch), Michael Nielsen's *Neural Networks and Deep Learning*
  (narrative prose with small interactive pieces), TensorFlow Playground (decision boundaries), CNN Explainer, MLU-Explain,
  d2l.ai, fast.ai (top-down).

---

## Appendix A. Pedagogy review checklist (for the `pedagogy-reviewer` agent)

For each lesson, answer yes/no with evidence (quote):
1. Does the lesson open with a question or problem the persona would care about?
2. Are 2 to 4 key words introduced before the main explanation?
3. Is every term and symbol introduced before its first use, in the course order?
4. Does each concept go picture → numbers → symbol → (optional) code?
5. Is there at least one analogy per core idea, with its limit stated?
6. Does every number in prose meet the "numbers serve claims" rule?
7. Does every widget have a guided task with the expected observation in the text?
8. Is there a prediction *before* the explanation, with a committed answer?
9. Is there a worked example, and then a step the learner does?
10. Is at least one misconception named and refuted?
11. Is there a transfer check and a cumulative check?
12. Is TorC-specific engineering kept out of the concept layer?
13. Does the lesson close by answering its opening question, and point to the next question?
14. Would a cold reader from the target group be able to explain the main idea in one sentence afterwards?
