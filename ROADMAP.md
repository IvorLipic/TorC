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
- [ ] CI: trace drift check + Node tests on every push/PR (`verify.yml`)

## M2 — Course scaffolding
- [ ] Home / course-map page, lesson template, shared UI modules, lesson data format
- [ ] Per-lesson routing under `/TorC/`, progress persistence (localStorage)
- [ ] Move Backprop lab into the template as lesson 1

## M3 — First lessons (see content plan for specs)
- [ ] Intro: why following gradients works (slope, chain rule intuition)
- [ ] SGD update step (and momentum)
- [ ] `nn::Linear`: shapes, matmul + bias, init

## M4+ — Remaining library coverage
Tensors/shapes/broadcasting, nn activations & losses, optimizers (Adam/AdamW), data loading, conv2d, end-to-end examples
(linear regression, MNIST MLP/CNN). Order and specs in the content plan.

## Open decisions
Tracked in [docs/decisions.md](docs/decisions.md) ("Open").
