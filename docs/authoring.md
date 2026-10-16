# Authoring guide

How to add a lesson, a scene or an operation. Architecture and file map: [architecture.md](architecture.md); what to teach and in
which order: [content-plan.md](content-plan.md).

## Writing rules (every visible sentence)
Owner focus group, 2026-10: the pages read as LLM output, and every sentence had friction. Length is not the problem. Follow the
structural rules of ASD-STE100 (Simplified Technical English), STE-flavored mode, and write like a teacher, not like a slogan.
- **Complete sentences.** A subject and a verb. No fragments ("One weight, one score"), no slogans ("Step size decides
  everything"), no "X, not Y" punchlines, no colon in the middle of a sentence. A colon may only end a line before a formula,
  code or a list. No semicolons, no dashes.
- **One idea per sentence, at most 25 words.** Split anything longer.
- **Define every term before you use it**, in plain words: learning rate, local derivative, neuron, batch, momentum.
- **One name for one thing** on a page: loss (not score), weight or parameter (not knob), gradient (not slope).
- **Active voice and plain verbs.** "TorC adds them", not "they are added". No phrasal verbs.
- **Headings say the topic plainly** ("How to choose the learning rate").
- `site/test/lessons.test.mjs` checks what a program can (`proseProblems` in `site/test/helpers.mjs`): semicolons, dashes, colons
  inside sentences, sentences over 25 words, on every lesson, course description and graph-lab text.

## Style budget (every lesson)
Lessons are short and dense. The interactive parts carry the lesson; the text only sets them up.
- **Shape**: hero (`<h1>` + one-line lead) → `tldr` (shown as "Summary", 3-4 sentences, ≤ 60 words) → the TorC code → 3-4
  sections → checks → one "Next" line. Each section title names its topic; its body is ≤ 100 words with at most one equation. No follow-up
  "what you should have seen" sections, no recap that repeats the TL;DR, no mix-up cards.
- **Limits**: ≤ 800 words of prose, ≤ 25 numbers bound in the text, **≤ 2 widgets** (one main, one small), ≤ 1 predict + 2 checks,
  a gotcha only where TorC really throws.
- **Provenance is said once**, on the home page, plus the standard one-line footer (`Numbers: TorC <ref> · source`, added by
  `lesson-boot.js` when the lesson has no `footer`). No badges, "recorded/recomputed/printed by TorC" labels, dotted underlines or
  recorded-run overlays on lesson pages. The guarantee lives in the tests (`T()` binding, engine == trace), not in the prose.
- **Widgets**: ≤ 2 controls plus Step/Reset; preset buttons instead of a slider where a slider adds nothing; a compact readout
  instead of explanations inside the widget; no visible tables (`srTable` gives screen readers the values).
- `site/test/lessons.test.mjs` asserts the budget for every published lesson (`lessonStats` in `site/test/helpers.mjs`).

## Add a lesson
1. **Trace data**: write the C++ trace program(s) (see "Add a scene"); the lesson lists the files it reads, e.g. `["why-gradients.json"]`.
2. **`site/course.json`**: add the lesson to its part with `{id, title, outcome, prereqs, symbols, trace, est_minutes, status}`.
   Add `widgets` (names shown on the home page) and `code` (the TorC lines the lesson is about, 2-6 lines; add its prelude and
   checks to `SPEC` in `tools/snippets/check.mjs` and run it). A lesson on the curriculum without a page yet is `"planned"`.
   Start the page with `"status": "draft"`: a draft has a page and is reachable at `lessons/<id>/`, but is not listed on the home page or in
   anyone's prev/next. Switch to `"published"` when the lesson has passed review. Optional `legacy_hashes: ["reuse", …]` forwards old
   `/#reuse/3` links from the home page to this lesson.
3. **Module `site/js/lessons/<id>.js`**: `export default {id, title, trace:[files], sections:[…], footer?: (D, T) => html}`.
   `id`, `title` and `trace` must equal the course.json entry (checked). Section kinds:
   - `{kind:'tldr', items:(D, T) => ['…', …]}` — the TL;DR box; second section, right after the hero.
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

- **Layout text is HTML.** `goal`, `intro`, `forwardHint`, `backwardHint` and `takeaway` are HTML fragments everywhere they appear
  (the goal line, the takeaway box and the math window's intro heading all insert them as markup). Plain text works as it is; use
  `<code>`, `<strong>`, `<em>` freely, with no wrapper element (the lab puts the goal/takeaway into one inline span itself, so
  phones do not split it into separate flex items). Write `&amp;` and `&lt;` for a literal `&` or `<` that is not followed by a
  space. Numbers still come from `T()` (or are covered by a test).
- **Keyboard.** `←`/`→` step the lab only when focus is inside the lab, or when nothing is focused and the lab is the only one on the
  page (`labOwnsArrows` in `graph-lab.js`, unit-tested). Controls of other widgets (sliders, buttons, tables) keep their arrow keys
  without any `stopPropagation` in the widget; do not add one.
- **Numbers on the graph.** `fmt` (shared by `T()`, the graph and the math window) prints integer-valued numbers below 1e6 exactly
  (`25088`) and rounds everything else to 4 significant digits. A contribution pill on an edge is signed (`+3`, `-6`, `+0`), never
  `+-6`; tensor contributions are labelled `c1`, `c2`, … and listed in the math window.

### Plots (`widgets/plot.js`)
`createPlot(host, {width, height, x, y, xLabel, yLabel, ariaLabel, margin?, responsive?, minWidth?, onResize?})` returns the plot API
(`axes`, `curve`, `dot`, `line`, `tangent`, `text`, `heat`, `layer`, `clear`, `sx`, `sy`, `plotArea`, `destroy`); the header of the file
lists every call. Rules for lessons:
- **Create the plot inside the element it lives in** (not in a detached `div` that is inserted later): the width is measured from `host`
  at creation. `width`/`height` are the design size; a host narrower than `width` (a phone) gets a plot built at the host's width, so the
  11-12px text is never scaled down. A host wider than the design size scales the SVG up, capped at 78% of the viewport height (the
  SVG's `max-width` is its aspect times that height), so a plot never letterboxes and never exceeds the screen. `responsive: false`
  keeps the design size whatever the host is.
- **Redraw on resize with `onResize(plot, {width, height})`.** When given, a `ResizeObserver` rebuilds the geometry when the host
  width changes (rotating a phone, resizing a window): the plot clears its axes and all layers, updates `sx`, `sy`, `width`, `height`
  and `plotArea` in place, and calls `onResize` - which must draw the content again (put the drawing in one function, call it
  once after `createPlot` and pass it as `onResize`). Keep widget state (the slider value, the selection) outside the drawing function.
  Without `onResize` the size is decided once at creation, which is fine for a plot that is rebuilt on every input anyway.
- **Ids are per plot** (clip path, and the arrow marker, which only exists when `line(…, {arrow: true})` is used), so any number of
  plots can share a page; do not strip or reference `plot-arrow`.
- **`heat(grid)`** draws one rectangle per grid point; the cells tile the plot exactly (edge cells reach as far outward as inward).
- **`srTable(host, {columns, rows, caption})`** adds a screen-reader-only table of the plotted values (nothing on screen);
  `update(rows)` refreshes it. **`fmtTick`** gives 4 significant digits and exponent form (`1.5e6`, `2e-4`) beyond 1e6 / below 1e-3.

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
