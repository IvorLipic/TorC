# Autograd visualizations

Ownership: `visualizations/` (examples, Manim source, scripts), `site/` (static website), `docs/autograd-audit/`
(contract + plan), `.github/workflows/pages.yml`. The C++ library (`include/`, `src/`, `tests/`) is untouched.

## Pipeline
```
examples/autograd_trace.cpp ──(real torc run)──► generated/autograd_trace.json ──► manim/torc_scenes.py ──► media/*.mp4 ──► site/
                                                                       └──────────────────────────────► site/data/ (tables on the page)
```
JSON schema: `scenes[] = {id,title,expression,note,root,root_has_grad,nodes[],finite_difference}`;
`nodes[] = {name,op,role("leaf"|"op"),requires_grad,inputs[] (read from the real TapeEntry),value{shape,values},grad{shape,values}|null}`;
`finite_difference = {h,atol,leaves[{name,numerical[],analytic[],max_abs_err,ok}]}`.
Manim files contain layout only; scene values, edges and notes come from the JSON. The site prose quotes a few values by hand (checked, not generated).

## Commands
```bash
# toolchain: clang++ >= 18 (or GCC >= 14); manim + LaTeX + libpango1.0-dev/libcairo2-dev for Manim
visualizations/examples/build_and_run.sh                          # C++ trace -> JSON
python3 -m venv .venv-viz && .venv-viz/bin/pip install manim
cd visualizations/manim && ../../.venv-viz/bin/manim -qm --media_dir ../media torc_scenes.py ReuseScene ChainScene MlpScene
visualizations/build_site.sh                                      # JSON + mp4 (+ webm via ffmpeg) -> site/
python3 -m http.server -d site 8000                               # preview (fetch() needs HTTP)
```
Fast preview while iterating: `manim -ql ...` (480p15).

## Base path
All site URLs are relative, so the same files work at `/` and `/TorC/`.

## GitHub Pages
Pages source: **GitHub Actions**; deployment branch: `web`. `.github/workflows/pages.yml` deploys `site/` on every push to `web`
that changes `site/**` (or the workflow), and can be run manually; it fails on any other branch. Site URL: https://ivorlipic.github.io/TorC/

Branch flow: library fixes → their own branch → review → `main`; then `main` is merged into `web`. Site and visualization work is committed on `web`.
