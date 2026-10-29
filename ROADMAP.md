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

## Open decisions
Tracked in [docs/decisions.md](docs/decisions.md) ("Open").
