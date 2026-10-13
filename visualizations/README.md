# Autograd visualizations

Ownership: `visualizations/` (examples, Manim source, scripts), `site/` (static website), `docs/autograd-audit/`
(contract + plan), `.github/workflows/pages.yml`. The C++ library (`include/`, `src/`, `tests/`) is untouched.

## Pipeline
```
examples/autograd_trace.cpp ──(real torc run)──► generated/autograd_trace.json ──► site/data/autograd_trace.json
                                                                                    └─► site/engine.js (JS port, tested against it)
                                                                                    └─► site/explain.js (LaTeX of every step) ─► site/app.js (UI)
```
`site/` is a dependency-free static site (KaTeX is vendored in `site/vendor/katex`). Scene layouts and teaching text live in `site/scenes.js`;
numbers never do. `site/test/engine.test.mjs` asserts that the JS engine reproduces every value and gradient of the real C++ run,
that finite differences agree, and that every generated LaTeX string parses in KaTeX.

JSON schema: `scenes[] = {id,title,expression,note,root,root_has_grad,nodes[],finite_difference}`;
`nodes[] = {name,op,role("leaf"|"op"),requires_grad,inputs[] (read from the real TapeEntry),value{shape,values},grad{shape,values}|null}`.

## Commands
```bash
# toolchain for the C++ trace: clang++ >= 18 (or GCC >= 14)
visualizations/examples/build_and_run.sh      # real library run -> visualizations/generated/autograd_trace.json
visualizations/build_site.sh                  # copy into site/ and run the tests (needs Node >= 20)
python3 -m http.server -d site 8000           # preview at http://localhost:8000/ (fetch() needs HTTP)
```
All site URLs are relative, so the same files work at `/` and at `/TorC/`.

`visualizations/manim/` holds the earlier Manim video scenes. They are optional and not used by the site.

## GitHub Pages
Pages source: **GitHub Actions**; deployment branch: `web`. `.github/workflows/pages.yml` deploys `site/` on every push to `web`
that changes `site/**` (or the workflow), and can be run manually; it fails on any other branch. Site URL: https://ivorlipic.github.io/TorC/

Branch flow: library fixes → their own branch → review → `main`; then `main` is merged into `web`. Site and visualization work is committed on `web`.
