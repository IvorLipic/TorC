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
- [x] TorC `v0.1.0` released (pre-release); `TORC_REF` now pins the tag
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
- [x] Automated accessibility audit (axe-core 4.10, WCAG 2.1 A/AA + best practice) clean on every page at 1366 and 390 px; keyboard walk of every page
- [ ] Outside cold reader per lesson (definition-of-done item 6), a pass with a real screen reader, Firefox/Safari check (only Chromium available here)
- [ ] Bump `TORC_REF` when TorC tags a release that includes the `MSELoss` broadcast-gradient fix (merged on `main`, not in v0.1.0)

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
- [x] `tools/snippets/check.mjs`: every snippet compiled and run against TorC v0.1.0 (13 run, 1 compile-only)
- [x] The snippet check runs in CI (`verify.yml`, after the trace drift check, on the TorC source it fetched)

## M4+ — The planned lessons, in curriculum order
Tensors → Nonlinearity → Module → Losses → The training loop → Better optimizers → Generalization → MNIST MLP → Convolution →
MNIST CNN. Specs in content-plan section 2 (v1 course map) and section 3.

## Open decisions
Tracked in [docs/decisions.md](docs/decisions.md) ("Open").
