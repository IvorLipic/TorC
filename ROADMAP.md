# Roadmap — TorC Learn

Status legend: `[x]` done · `[~]` in progress · `[ ]` planned. The detailed curriculum lives in
[docs/content-plan.md](docs/content-plan.md); this file tracks delivery.

## M0 — Backprop lab (done)
- [x] Real-library trace → JSON (graph edges from the tape, values, gradients, finite-difference check)
- [x] Interactive graph, step timeline, per-node math window (rule + same rule with numbers), editable inputs
- [x] JS port of the op rules, tested against the C++ output; KaTeX vendored; deployed to GitHub Pages
- [x] Independent review (math, UX, accessibility) and fixes

## M1 — Project foundation (this branch as its own project)
- [x] Library removed from this branch; TorC consumed at a pinned ref (`tools/trace/TORC_REF`)
- [x] TorC packaged on `main` (CMake install/export, options, version; TorC PRs 2–3), `tools/trace` pinned to its merge commit
- [x] TorC `v0.1.0` released (pre-release); `TORC_REF` pinned the tag until 2026-10-29
- [x] Project docs: README, architecture, authoring guide, decisions, content plan
- [x] Compact lab (fits one laptop screen), auto-play removed
- [x] CI: trace drift check + Node tests + site check on every push/PR (`verify.yml`); TorC's own CI (build matrix, consumer smoke test, ASan/UBSan) lives on `main`

## M2 — Course scaffolding
- [x] Home / course-map page (`site/index.html` from `course.json`: parts, lesson cards, filter box, done ticks), lesson template (committed shells + `lesson-boot.js`), shared UI modules (`js/core/`, `js/widgets/`), lesson data format (`js/lessons/<id>.js` with `text`/`widget`/`gotcha`/`predict`/`check` sections and `T('scene.path')` bound numbers)
- [x] Per-lesson routing under `/TorC/` (relative URLs only, old `#mlp/9` links redirected), progress persistence (localStorage `torc:v1`, works with storage blocked)
- [x] Move Backprop lab into the template as lesson 1 (`backprop-lab`)
- [x] Op registry: one module per op in `site/js/ops/` (incl. `transpose`, `mse`); engine and explain are generic consumers; every registered op must appear in a trace scene
- [x] `tools/check_site.mjs` (course.json, shells, modules, trace files, no root-absolute URLs, asset references) runs in `pages.yml` and `verify.yml`
- [ ] Op-card reference page (`site/ref/ops/`, generated from the registry) and API index (`site/ref/api/`, built from `symbols`)

## M3 — First lessons (see content plan for specs)
- [x] `why-gradients`: why following gradients works (slope, step size, chain rule as gears) — published after independent review
- [x] `sgd-step`: the SGD update step, learning rate, zero_grad, momentum — published after independent review
- [x] `linear-layer`: `nn::Linear` shapes, matmul + bias, backward, init — published after independent review
- [x] Independent review of M3.5–M3.6 (no blocker/high; 3 medium and 12 low findings, all fixed: CI now compares quoted library lines, snippet checker refuses other refs and checks more values, Kaiming wording uses the 10-draw range, budget limits per section/TL;DR/question type tested, a11y and formatting fixes)
- [x] Automated accessibility audit (axe-core 4.10, WCAG 2.1 A/AA + best practice) clean on every page at 1366 and 390 px; keyboard walk of every page
- [ ] Outside cold reader per lesson (definition-of-done item 6), a pass with a real screen reader, Firefox/Safari check (only Chromium available here)
- [x] `TORC_REF` bumped to `38785db` (TorC `main` after PR #5 and PR #6: `MSELoss` throws `ShapeError` for unequal shapes); only the MSE-mismatch trace entries changed
- [ ] Pin a release tag again when TorC tags one that includes PR #6 (`docs/decisions.md`, Open)

## M3.5 — Concise lessons (owner review: less prose, fewer numbers and features; decisions 7-9)
- [x] Style budget in `docs/authoring.md`; `tldr` section kind; standard one-line footer; plot tables are screen-reader only
- [x] `why-gradients` rewritten as the pilot (one merged bowl widget + graph lab; ~400 words) — approved by the owner
- [x] `sgd-step`: one landscape widget (Step, lr and momentum presets) + a `zero_grad` checkbox; graph lab, sandbox, update window removed (~400 words)
- [x] `linear-layer`: the matrix grid + the six-layer signal plot; shape explorer, graph lab, backward widget and histogram removed (~360 words)
- [x] `backprop-lab` hero/footer trim (concept cards became the TL;DR); course outcomes ≤ 12 words; time estimates 6-7 min
- [x] Budget test for every published lesson in `lessons.test.mjs`

## M3.6 — Curriculum home page (decisions 10-11)
- [x] Curriculum in the usual university order (content-plan section 2): 4 parts, 13 lessons, planned ones shown, labs apart
- [x] Home page: numbered lessons with the TorC code beside each and its widgets; Backprop lab moved to Labs
- [x] `tools/snippets/check.mjs`: every snippet compiled and run against TorC v0.1.0 (13 run, 1 compile-only), and again against `38785db`
- [x] The snippet check runs in CI (`verify.yml`, after the trace drift check, on the TorC source it fetched)

## M4+ — The planned lessons, in curriculum order
- [x] 03 Tensors (2026-10-16): trace `tensors.json`, broadcast grid and axis explorer, tests, prose check clean
- [x] 05 Nonlinearity (2026-10-16): trace `nonlinearity.json` (activations, softmax, two models fitted in TorC), activation plot and curve fit
- [x] 06 Modules (2026-10-16): trace `modules.json` (parameter counts, the mismatch error, call order), model viewer and call order
- [x] 07 Losses (2026-10-16): trace `losses.json` (MSE, cross-entropy, batch mean, CE vs squared-error gradient, label errors), loss explorer and gradient comparison
- [x] 08 The training loop (2026-10-20): trace `training.json` (batch orders, three batch sizes, the run without `reset()`, the loss of the true line), batch viewer and training curves
- [x] 09 Better optimizers (2026-10-20): trace `optimizers.json` (SGD, momentum and Adam in a narrow valley, Adam's first step, AdamW's decay), optimizer race
- [x] 10 Generalization (2026-10-20): trace `generalization.json` (a degree-15 polynomial on 10 samples, three weight decays), model and loss curves
- [x] 11 Digits with an MLP (2026-10-20): MNIST fetched and checked by `fetch_mnist.sh`, trace `mnist-mlp.json` (the example's model trained in TorC, its weights, 20 test images with TorC's logits), weights gallery and digit pad (a tested browser port of the forward pass)
- [x] 12 Convolution (2026-10-20): trace `conv.json` (a small example in three settings, edge kernels on a digit, the example CNN's shapes and parameters, the errors), sliding window and feature maps
- [x] 13 Digits with a CNN (2026-10-20): trace `mnist-cnn.json` (the example's CNN trained in TorC for 5 epochs, its weights, test images with TorC's logits, the first feature maps), kernels and feature maps, digit pad with a tested browser port of the CNN

All planned lessons are published. Specs in content-plan section 2 (v1 course map) and section 3.
Lessons 09 to 13 had an independent review on 2026-10-20; its findings are fixed. Still open: a pass with a real screen reader, Firefox and Safari, and a real touch
device for the digit pads. TorC findings for `main` are in `docs/library-audit/v0.1.0-findings.md`.

## M5 — Curriculum v3 (the pedagogy rework; brief `docs/pedagogy-review.md`, contract `docs/v3-implementation.md`)
Status (2026-11-08): phases A to C are done; phase D, the human tests and the merge into `web` are open.
Goal: the best resource for the high-school graduate of decision 13. All approval items of brief section 12 were granted on
2026-10-29 (decisions 13-23). Work happens on the work branch `claude/trusting-archimedes-kjzrto` (decision 20). Lessons are
referred to by id. Defect and error ids (D1-D20, C1-C10) are in the brief, sections 5 and 11.

### Phase A — Quick fixes (safe on `web`) — done on `web`
- [x] Copy the brief to the repository as `docs/pedagogy-review.md` (`3c7fa0c` on `web`)
- [x] Fix D1-D6, D10-D15, D17, D19, D20 (D6 by removing its text), and the wording parts of C4, C7, C9 and C10 (`280036c`, merged by `9bcb2e4`)
- [x] The LaTeX-escape grep test and the next-lesson test (in `280036c`)
- [x] `generalization` `SPEC` in `tools/snippets/check.mjs` updated where C10 changed the `code`
- [x] `node --test site/test/`, `tools/prose/check.sh`, `node tools/check_site.mjs` pass; independent-review findings fixed (`2fba6b0`)
- [x] Pushed to `web`. The TorC pin bump to `38785db` (TorC PR #6) is `fa1ea01` on `web`

### Phase B — Decisions, docs, machinery and the vertical slice — done on the work branch
- [x] Record the approved decisions (13-23 in `docs/decisions.md`; 24 records the naming rules from the reviews)
- [x] Update `docs/content-plan.md` sections 1 and 2, `docs/authoring.md` policy sections, `CLAUDE.md` and `.claude/agents/`
      (`pedagogy-reviewer`, `cold-reader`, `independent-reviewer` dimension 8; `da2094b`)
- [x] Machinery: template 3, `quiz`, `explain` and `answer` kinds, progress store v2 (`torc:v2`, v1 migration), role-tagged `T()`,
      `T.human`, `T.anchor`, the prose formatter, `site/anchors.json`, blocks (`worked`, `inTorc`, `realLine`, `analogy`, `chatgpt`,
      `refresher`, `term`, `sym`, `loopDiagram`), part kinds (`lab`, `appendix`), v3 tests (`6cd01f3`, merged `c8746f7`)
- [x] Notation guide, glossary of 61 terms, ten maths refreshers, the `notation` and `glossary` pages (`74a4ca6`, `622312f`, merged `03f60f4`)
- [x] Vertical slice: `what-is-learning`, `why-gradients`, `sgd-step`, each with module, widgets, tests and `docs/lessons/<id>.md`
- [x] `pedagogy-reviewer` and `cold-reader` ran on the slice
- [ ] At least 3 human readers from the target group (owner-only, not done)
- [ ] Adjust the rules in the brief and `docs/authoring.md` from what the readers show (waits for the readers)

### Phase C — The rest of the main path and the appendix — done on the work branch
Each lesson passed three agent gates: the `independent-reviewer`, the `pedagogy-reviewer` and the `cold-reader`. The appendix pages
had two gates and no cold reader, because they are for TorC users. The cold reader is a smoke test, not evidence of learning.
- [x] Part `networks`: `linear-layer` (`3ac3830`), `nonlinearity` (`ccb645b`), `classification` (`10d6082`), `losses` (`832ea66`), `backprop` (`b3ba26c`), `deep-nets` (`417a1f9`)
- [x] Part `training`: `training-loop` (`66bb181`), `optimizers` (`9ecdf19`), `generalization` (`b246885`)
- [x] Part `images`: `mnist-mlp` (`463e71c`), `one-step` (`e813db5`), `convolution` (`d2cbbd3`), `mnist-cnn` (`a7f02d9`)
- [x] Part `next`: `from-digits-to-chatgpt` (`edf4a88`; conceptual, with cited sources)
- [x] Appendix `using-torc`: `tensors` (`d88051d`), `modules` (`67cb7b1`), `torc-pitfalls` (`793b530`), `notation`, `glossary`
- [x] Defects D7 (with `modules`), D16 (graph labs at phone width, with `backprop`), D18 (with `optimizers`)
- [~] Legacy kinds `tldr`, `predict`, `check`: the main path and the appendix are all `template: 3`, but the kinds stay in
      `site/js/core/sections.js` because the Backprop lab (exempt) still uses them. Not removed.

### Phase D — Spacing and polish — not done
- [ ] End-of-part reviews and cumulative checks
- [ ] Light theme (the shells are still `color-scheme: dark`; rewrite the shells with `--write-shells`; raise the minimum caption size in widgets)
- [ ] Firefox and Safari (only Chromium has been available)
- [ ] A real screen reader
- [ ] The digit pad on a touch device

### Not done, owner-only
- [ ] The human think-aloud tests, 3 to 5 readers from the target group per part (`CLAUDE.md`, "Verification")
- [ ] Merge of the work branch `claude/trusting-archimedes-kjzrto` into `web`. The owner decides, after the human tests

### Known open items
- A TorC library bug, recorded in `docs/library-audit/v0.1.0-findings.md`: `conv2d` accepts an input smaller than the kernel when the
  size numerator is negative. It awaits a fix branch on `main`, then a pin bump here.
- The cited sources of the conceptual pages (the epilogue `from-digits-to-chatgpt`, the refreshers, the backprop cost principle)
  could not be fetched in this environment. They are marked "verify before quoting" in their lesson notes (`docs/lessons/`).
- Pin a release tag when TorC tags one that includes PR #6 (see "Open decisions").

### Acceptance criteria (per lesson; brief section 12)
- [ ] A human reader from the target group can answer the lesson's opening question in their own words afterwards
- [ ] No term, symbol or notation appears before its introduction in course order (test)
- [ ] Every prose number has a role; at most 10 numbers outside `worked`, `setup` and `reveal`; at most 2 non-`worked` numbers per sentence (test)
- [ ] Every widget has a guided task, and the text states the observation
- [ ] The core template of `docs/authoring.md` is met (exceptions: `from-digits-to-chatgpt` and the appendix pages)
- [ ] No unfair comparison, no test-set peeking, and claims from a single seed are labelled
- [ ] The `pedagogy-reviewer` reports no High findings
- [ ] All existing gates pass: trace drift, Node tests, site check, prose linter, accessibility audit

## Open decisions
Tracked in [docs/decisions.md](docs/decisions.md) ("Open").
