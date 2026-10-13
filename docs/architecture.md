# Architecture

## Goal
A static website (GitHub Pages, served under `/TorC/`) of interactive lessons on TorC. Every number a learner sees must be
traceable to a real run of the library; learners can edit inputs and see results recompute.

## Pipeline
```
TorC @ pinned ref ──► tools/trace/*.cpp (real run) ──► site/data/*.json ──┬─► site/engine.js   (JS port of op rules, tested == JSON)
                                                                         ├─► site/explain.js  (rule + numbers as KaTeX "paper work")
                                                                         └─► site/app.js      (UI: graph, stepper, math window, inputs)
                                                  site/scenes.js: layout + teaching text only (never numbers)
```
- **Trace programs** (`tools/trace/`) build against TorC fetched by CMake `FetchContent` at the ref in `tools/trace/TORC_REF`
  (a release tag, or the commit SHA the tag points to). `run.sh --check` regenerates and fails on drift.
- **JSON schema** (`autograd_trace.json`): `scenes[] = {id,title,expression,note,root,root_has_grad,nodes[],finite_difference}`;
  `nodes[] = {name,op,role("leaf"|"op"),requires_grad,inputs[],value{shape,values},grad{shape,values}|null}`. `inputs` are read
  from the real `TapeEntry`; `finite_difference` holds numerical vs. analytic gradients (h=1e-2, atol=1e-2).
- **Engine** (`engine.js`): a small JS re-implementation of the op rules actually used in lessons (add, sub, mul, relu, sum,
  mean, matmul). It exists only so learners can edit inputs; `site/test/engine.test.mjs` asserts it reproduces every value and
  gradient of the real run, agrees with finite differences, and that every generated LaTeX string parses in KaTeX.
- **Explain** (`explain.js`): turns engine results into LaTeX: the general rule, then the rule with numbers substituted, plus
  accumulation ("c1 + c2 + c3") and broadcast-reduction bookkeeping.
- **UI** (`app.js`, `style.css`, `index.html`): persistent SVG graph, step timeline (checkpoints), math window, input popover,
  `#scene/step` deep links, keyboard control. No framework, no build step; KaTeX is vendored in `site/vendor/katex`.

## Relationship to the library
`main` (TorC) and `web` are separate projects. `web` holds no library code. It depends on TorC through a *pin*:
bump `tools/trace/TORC_REF`, run `tools/trace/run.sh`, review the JSON diff and tests, commit. Do **not** merge `main` into `web`.
Library bugs found while writing lessons are reported with a minimal reproduction and fixed on `main` (see `CLAUDE.md`).

## Where it is going
Single lesson → multi-lesson site (course map, lesson template, shared modules, lesson data format). See
[content-plan.md](content-plan.md) and [../ROADMAP.md](../ROADMAP.md).

## Testing and CI
- `node --test site/test/` — engine vs. real output, finite differences, LaTeX validity, explanation content.
- `tools/trace/run.sh --check` — committed JSON equals a fresh run against the pinned TorC.
- `.github/workflows/verify.yml` runs both on every push/PR touching `site/` or `tools/`; `pages.yml` deploys `site/` on push to
  `web` (and runs the Node tests first).
