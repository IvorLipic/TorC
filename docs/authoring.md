# Authoring guide

How to add a lesson, a scene or an operation. Architecture and file map: [architecture.md](architecture.md); what to teach and in
which order: [content-plan.md](content-plan.md).

## Add a lesson
1. **Trace data**: write the C++ trace program(s) (see "Add a scene"); the lesson lists the files it reads, e.g. `["why-gradients.json"]`.
2. **`site/course.json`**: add the lesson to its part with `{id, title, outcome, prereqs, symbols, trace, est_minutes, status}`.
   Start with `"status": "draft"`: a draft has a page and is reachable at `lessons/<id>/`, but is not listed on the home page or in
   anyone's prev/next. Switch to `"published"` when the lesson has passed review. Optional `legacy_hashes: ["reuse", …]` forwards old
   `/#reuse/3` links from the home page to this lesson.
3. **Module `site/js/lessons/<id>.js`**: `export default {id, title, trace:[files], sections:[…], footer?: (D, T) => html}`.
   `id`, `title` and `trace` must equal the course.json entry (checked). Section kinds:
   - `{kind:'text', html:(D, T) => '<section>…</section>'}` — markup is used as given. `D` is the loaded trace data
     (`D.scenes[id]`, `D.files[file stem]`); `T` puts numbers from it into prose (below). Must not touch the DOM.
   - `{kind:'widget', id, className?, label?, mount(host, D, ctx)}` — an interactive part; `host` is an empty `<section>`.
     `mount` runs in the browser only; return `{destroy()}` if it registers global listeners.
   - `{kind:'gotcha', title, html}` — "Why did my X fail?" box with the recorded real message.
   - `{kind:'predict', id, prompt, reveal}` and `{kind:'check', id, prompt, reveal}` — think first, then open the answer
     (`check` also stores "I got it" in the progress store). `prompt`/`reveal` are `(D, T) => html`.
   The first section should be a hero containing the lesson's `<h1>`.
4. **Shell**: `node tools/check_site.mjs --write-shells` writes `site/lessons/<id>/index.html` (a ~15-line committed file; the
   checker fails if a shell differs from the template).
5. `node --test site/test/` and `node tools/check_site.mjs`; then look at every step on desktop and phone width under a
   `/TorC/` prefix server.

### Numbers in prose
Never type a number the trace could change. Use `T('scene.path')`, e.g. `` `x.grad is ${T('reuse.x.grad')}` ``:
it returns `<span class="num" data-trace="reuse.x.grad">7</span>` and **throws if the path is missing**, so prose can never
silently drift from the data. Paths: `<sceneId>.<node>.<value|grad>[.<index>]` (a one-element tensor resolves to its number),
or `files.<file stem>.<key>…` for the extra tables of a trace file; in arrays a segment is an index or an item's `name`.
`T.num(path)` / `T.text(path)` give the bare number / formatted text. `site/test/lessons.test.mjs` renders every lesson in Node
against the real JSON, rejects any `class="num"` without `data-trace`, and re-resolves every bound one. If a number is
unavoidable in a string that is not rendered through `T` (for example a scene takeaway), add a test that asserts it.

### Using the graph lab
`widgets/graph-lab.js` (`mountGraphLab(host, {scenes, layouts, defaultScene})`) gives the scene tabs, step timeline, SVG graph,
math window and inputs popover for any scenes in the trace format. A layout per scene says where each named node is drawn
(`pos`, `view`, `minWidth`, optional `above`) plus the teaching text (`tab, goal, intro, forwardHint, backwardHint, takeaway`);
see `js/lessons/backprop-lab.scenes.js`. The four parts (`graph`, `stepper`, `math-window`, `inputs-popover`) can be used
separately. There is one lab per page (it uses fixed element ids).

## Add a scene
1. **Trace**: in `tools/trace/` add a `Scene` (leaf values, a builder that registers every node with
   `g.op(name, opName, <real torc op>)`, constants listed in `s.constants`). Keep every tracked `Variable` alive (the program
   uses a `std::deque`), and make sure no ReLU kink is within the finite-difference step of any input.
2. `tools/trace/run.sh` → writes `site/data/*.json`. Inspect the numbers; hand-check at least the gradients.
3. **Layout and text**: a layout entry for the scene in the lesson (see above). Never put numbers in these strings that the trace
   could change; if a number is unavoidable, add a test that asserts it.
4. `node --test site/test/` and look at every step on desktop and phone width.

## Add an operation
1. Confirm in TorC (`src/autograd.cpp`, `src/nn/*.cpp`) what the forward and backward rules really are — including broadcasting
   and edge cases.
2. Add `site/js/ops/<name>.js` exporting `{name, title, arity, forward, backward, texForward, whyForward, forwardNumbers,
   texBackward, whyBackward, backwardNumbers}` (copy a small op such as `sum.js`; `elementwise.js` is shared by add/sub/mul).
   `backward(g, inputs, out)` returns *raw* gradients; reduction to each input's shape is done once in `runBackward`.
   `forwardNumbers`/`backwardNumbers` receive a context and return the rule with the real numbers as LaTeX. Every string must
   parse in KaTeX (the tests render them with `throwOnError`). Register it in `ops/index.js`.
3. Add a trace scene that uses the op, so the C++ output exercises it: `ops.test.mjs` fails for any registered op that appears in
   no trace scene (`OPS_AWAITING_TRACE` in that file is the temporary exception list; keep it empty). The engine test then
   compares the JS port with the real run, and add a finite-difference test on random tensors for non-trivial rules (see
   the `transpose` and `mse` tests).

## Conventions
- Multi-letter node names render as `\texttt{…}` so `ab` is never read as a product; single letters are italic math.
- Gradient notation: overline (`ā` = ∂out/∂a), defined where it first appears.
- Explanations are about what **TorC does**, with the math next to it. Quirks of the library belong in `docs/library-audit/`,
  not on lesson pages.
- Everything must work with browser storage blocked, without a pointer (keyboard), and with reduced motion; never autoplay.
- URLs in source are relative (`../../js/…`, `new URL('data/x.json', SITE)`); never root-absolute and no `<base>`.
