# Authoring guide

How to add a lesson, a scene or an operation. Architecture and file map: [architecture.md](architecture.md); what to teach and in
which order: [content-plan.md](content-plan.md).

## Writing rules (every visible sentence)
Owner focus group, 2026-10 (decision 12): the pages read as LLM output, and every sentence had friction. Decision 14 (2026-10-29,
brief R8) keeps the structural rules of ASD-STE100 (Simplified Technical English, STE-flavored mode), adds specified teacher moves
and bans a list of LLM tics. The reader is the high-school graduate of decision 13: someone who knows algebra and graphs of
functions and nothing else. Write like a teacher, not like a slogan. The teacher moves, the banned tics, the notation rules and the
number rules below are checked on lessons with `template: 3`. A lesson that is not migrated yet keeps the older checks until it is
rewritten (see "Legacy lessons" below).

**Structure rules (kept from decision 12).**
- **Complete sentences.** A subject and a verb. No fragments ("One weight, one score"), no slogans ("Step size decides
  everything"), no "X, not Y" punchlines, no colon in the middle of a sentence. A colon may end a line before a formula, code or a
  list, and may follow a label (see "Allowed by the linter"). No semicolons, no dashes used as punctuation.
- **One idea per sentence, at most 25 words.** Split anything longer.
- **Define every term before you use it**, in plain words and in course order: learning rate, gradient, neuron, batch, momentum. No
  term or symbol appears before the lesson that introduces it. The first use of a glossary term in a lesson is linked with `term()`.
- **One name for one thing**, on a page and across the course: loss (not score), gradient (not slope), `B` for the batch size.
  **Parameters** are all the numbers training changes; a **weight** multiplies an input and a **bias** is added (so a parameter count
  such as the MLP's 26,506 counts both: say "parameters", never "weights", for it). The loss is `L` and the call is `L.backward()`. Two exceptions. "Slope" may be used once, to define the gradient ("the gradient is the slope of the loss curve at the
  current point"); after that, say "gradient". "Knob" may be used only in the `what-is-learning` analogy, with the mapping stated
  ("the knobs are the parameters"). Use "graph" only for the computation graph and "plot" or "curve" for a plot.
- **Active voice and plain verbs.** "TorC adds them", not "they are added". No phrasal verbs.
- **Headings say the topic plainly** ("How to choose the learning rate").
- Load the vendored skill `.claude/skills/asd-ste100` before you write; use its STE-flavored mode.

**Teacher moves (allowed).**
- **At most one real question per section**, answered by the paragraph that follows. The "Questions" box and quiz prompts do not
  count toward this.
- **"You"** for instructions, tasks and predictions. **"We"** only for the course itself ("we practise this on a simple problem"),
  at most once per section.
- **A concrete situation before the abstraction** ("Two notebooks cost 6 euros", then the weight).
- **Analogies only from the bank in `docs/pedagogy-review.md` section 8**: gradient (fog), learning rate (stride), momentum (ball),
  Adam (a step size per weight), chain rule (gears), neuron (weighted vote), ReLU units (hinges), softmax (shares of 100 %),
  cross-entropy (surprise), overfitting (memorized exam), weight decay (leak), kernel (stencil). At most one per section. State its
  limit at first use, in a line that starts with "Where it breaks:". Use the `analogy(html, limit)` block.
- Bold may mark a key sentence or a term at its definition. Do not bold for rhythm.

**Banned tics (the linter fails on these, `proseProblems` in `site/test/helpers.mjs` and `tools/prose/check.sh`).** "let's", "here's
the thing", "it turns out", "simply", "magic", "crucially", "notice that", "the key insight", "under the hood", "in other words",
"think of it as", "imagine", "essentially", "at its core", "the heart of", "powerful", "elegant", "beautiful", "not just X but Y",
"That's it", "This is where X comes in", and "This lesson shows" as an opener.

**Flagged by the reviewer, not by the linter** (Appendix A item 13 of the brief). "Just" used as a hedge ("just multiply"). Lists
of three items used for rhythm. A sentence that only restates the paragraph as a punchline. Decorative metaphors.

**Allowed by the linter.** The minus sign (U+2212) and dashes inside maths are not punctuation dashes. A label colon at the start of
a block is allowed: "Task:", "What you will see:", "Your turn:", "Reveal:", "Where it breaks:". The bans on semicolons and on
dashes as punctuation stay.

**Checks in CI.** `tools/prose/check.sh` runs the skill's linter on the extracted prose of every page (0 hard findings allowed).
`site/test/lessons.test.mjs` checks the site rules (`proseProblems` in `site/test/helpers.mjs`): semicolons, dashes, colons inside
sentences, sentences over 25 words, and for `template: 3` lessons the banned list, on every lesson, course description and
graph-lab text. Visible strings in widgets and in `lesson-boot.js` follow the same rules (brief D15). The real test of the voice
is the owner's focus group and the human think-alouds (`CLAUDE.md` "Verification"), not a principle.

## Style budget (every lesson)
Decisions 17 and 23 (2026-10-29) replace the budget of decision 7 for lessons with `template: 3` in their module. The contract for
section kinds, blocks and tests is `docs/v3-implementation.md`. The goal is a lesson of 12 to 18 minutes in which every paragraph
explains a why, sets up a task, or answers a question.

### Lesson template (`template: 3`): a core plus a palette
**Core, in every lesson on the main path.**
1. **"Questions this lesson answers"** (section kind `questions`, right after the hero): 1 to 3 questions the reader would ask, each
   set in a concrete situation. The loop diagram (data, model, loss, adjust) sits in this box with this lesson's part lit
   (`loop` in the module).
2. **A committed prediction** (`quiz`, `purpose: 'predict'`), placed before any heading or text that contains its answer. The
   reader commits first. At least one wrong option looks plausible, and each wrong option's feedback names the misconception.
   No prompt may be answerable by re-reading the sentence just before it. Every number a prompt needs is on the page.
3. **One "aha" interaction** with a guided task. The text also states the observation, so the page works without the widget.
4. **One worked example** in which the reader does the last step (a `worked()` block, then a numeric `quiz`).
5. **One misconception refuted** with the widget's own evidence, not by assertion.
6. **"The answer"** (section kind `answer`, last): two or three sentences that answer the opening questions, then the next question
   that the next lesson answers.
7. **One transfer quiz** (a new situation, the same idea) and **one `explain` prompt** ("Explain it in one sentence", with a model
   answer to compare).
8. **One cumulative quiz** from an earlier lesson (`from: '<lesson id>'`), from the third lesson on.

**Palette, 0 to 2 per lesson.** An analogy with its stated limit; a maths refresher (`refresher(key)`); an "In TorC" panel
(`inTorc(title, html)`); a "same thing in ChatGPT" aside (`chatgpt(html)`).

**Per part.** The last lesson of each part ends with a short mixed review. It recommends going back and never blocks.

**Order of ideas.** Each idea goes: a thing you drag, then the same with numbers, then the symbol, then (optionally) the TorC code
(concreteness fading). Name the parts of a system on a labelled diagram before the main explanation.

**Exceptions.** `from-digits-to-chatgpt` and the appendix pages need no worked example, no widget and no widget-based refutation.
The appendix pages use `gotcha` where TorC really throws.

The core is a guide for judgement. Do not tick it like a checklist on every lesson, because a formula reads like LLM output.

### Budgets (tested for `template: 3` lessons)
| Item | Limit |
|---|---|
| "Questions" box | at most 3 items |
| "The answer" | at most 80 words |
| A text section | at most 150 words |
| Numbered sections | at most 6 |
| Quizzes | at most 6 |
| Prose in a lesson | at most 1,400 words (a soft cap that the test still enforces) |
| Predictions | at least 1 `predict` quiz before any text that answers it |
| Other quizzes | 1 `transfer`, 1 `explain`, and from the third lesson on 1 `cumulative` |
| Widgets | no cap on how many, but each has 1 to 3 guided tasks |
| Worked blocks | at most 3 |
| Reading time | 12 to 18 minutes (`est_minutes` in `course.json`); `deep-nets` and `from-digits-to-chatgpt` are shorter |
| Numbers | see "Number roles" |

The first three lessons (`what-is-learning`, `why-gradients`, `sgd-step`) aim at the low end, about 12 to 15 minutes each, so that
the first two parts give a complete payoff in under 45 minutes.

### Number roles
Numbers serve claims. A number the reader cannot check, did not predict, and cannot compare with something known is noise
(decision 23, brief R2). **Every number in a `template: 3` lesson has a role**, given in its binding
`T(path, role, opts?)`. An untagged `T(path)` fails the tests. How the binding resolves a path is in "Numbers in prose".

| Role | Where it is allowed | Shown as |
|---|---|---|
| `setup` | A constant or starting value of the scenario (x = 2, the start `w` = 0, the lr presets), or a value a prompt gives the reader | exact |
| `worked` | Only inside `worked(html)`, a calculation whose inputs are all visible. At most 3 worked blocks per lesson | exact |
| `reveal` | Inside a quiz reveal or quiz feedback | exact |
| `anchor` | `T.anchor(name)`, from `site/data/anchors.json` (784 pixels, 10 classes, 26,506 parameters) | integers exact, else 2 significant figures |
| `contrast` | Half of an A-versus-B pair, in a sentence that states the comparison in words first | 2 significant figures |
| `human` | `T.human(path, kind, opts)`: `'rmse'` (typical miss), `'prob'` (the probability given to the right class), `'pct'`, `'per100'` (mistakes per 100) | whole numbers or 2 significant figures |

- **Formatting** is done by the binding, not by hand. Integers are exact, with a comma every three digits from 1,000 up
  (`26,506`, decision 22). The exact roles use `fmt` (4 significant figures); all other numbers use 2. `opts.sig` (1 to 4) overrides.
  Accuracy is a percentage, or mistakes per 100.
- **Limits, tested.** At most 2 numbers per sentence outside `worked` blocks. At most 10 numbers per lesson outside the `worked`,
  `setup` and `reveal` roles. Numbers in code blocks and widgets do not count.
- **No bare loss value** outside `worked` or `reveal`. Say it in a human unit with `T.human` ("off by about 0.24", "66 % on the
  right class"). The reviewer checks this one.
- **Derived values** (a step size, a share in percent) come from new trace fields, never from arithmetic in the lesson.
- **Anchors** are trace paths listed in `site/anchors.json` (at most about 15, each used in at least two lessons). `node
  tools/anchors.mjs` writes their values to `site/data/anchors.json`, and a test checks that they still equal the traces.
- **Measured results in words are bound too.** "Nineteen more passes bring it to 97 %" is a measured result and needs a binding.
  Counting words in a setup or a comparison ("two notebooks", "ten times smaller") are fine.
- **Reviewer heuristic:** delete the number. If the sentence loses no meaning, leave it deleted.

### Widgets
- **Every widget earns its place with 1 to 3 guided tasks** (`tasks: (D, T) => [{ do, see }]`): what to do, and what you will see.
  The text states the expected observation, so the page works without the widget. A bare sandbox does not teach. This replaces the
  caps of 2 widgets and 2 controls.
- **Readouts** show 2 to 3 significant figures and carry a meaning label ("loss 1.2, lower is better"). Show only what the task
  needs. No visible multi-column tables (decision 9; `srTable` gives screen readers the values). The graph-lab maths windows keep
  `fmt`.
- **Labels on the plot.** Label curves directly on the plot, not in a legend under it. Give a symbol the same colour in the formula,
  the plot and the code.
- **Explain each plot type at its first use** with one sentence: a contour map ("each colour is the loss for one pair (w, b); dark
  is low"), a log axis, a heat map.
- Preset buttons instead of a slider where a slider adds nothing. Widget text is visible text and follows the writing rules.

### "This is the real line" moments
The concept text is completable without reading C++ (decision 15). Four or five short real lines stay on the main path, each with
a plain-English comment, through the `realLine(code, caption)` block:
- `loss.backward()` in `why-gradients`;
- the `SGD::step` update line in `sgd-step`;
- a `Sequential` model in `classification`;
- the five-line training loop in `training-loop`;
- the digit model in `mnist-mlp`.

Never show a novice setup code such as `Variable w(Tensor({0.0f}, {1}), true);`. All other TorC code, library facts and error
strings go in a collapsed `inTorc` panel or in the appendix part `using-torc`.

### Content beyond TorC
A page that teaches what TorC cannot show is `conceptual: true` in its module and lists `sources: [{ title, url }]`, which
`lesson-boot.js` prints in the footer (decision 18). It says in the text that the content is conceptual and not a TorC run. It
keeps numbers out, or states the number with its source. It does not claim anything about TorC that the library does not do.

### Provenance
Provenance is said once, on the home page, plus the standard one-line footer (`Numbers: TorC <ref> · source`, added by
`lesson-boot.js` when the lesson has no `footer`). No badges, "recorded/recomputed/printed by TorC" labels, dotted underlines or
recorded-run overlays on lesson pages. The guarantee lives in the tests (`T()` binding, engine == trace), not in the prose.
`site/test/lessons.test.mjs` asserts the budget that applies to each published lesson, legacy or `template: 3` (`lessonStats` in `site/test/helpers.mjs`).

### Legacy lessons (not yet `template: 3`)
Unmigrated lessons still follow the budget of decision 7, until each is rewritten. Do not add new lessons to it.
- **Shape**: hero (`<h1>` + one-line lead) → `tldr` (shown as "Summary", 3-4 sentences, ≤ 60 words) → the TorC code → 3-4
  sections → checks → one "Next" line. Each section title names its topic; its body is ≤ 100 words with at most one equation. No follow-up
  "what you should have seen" sections, no recap that repeats the TL;DR, no mix-up cards.
- **Limits**: ≤ 800 words of prose, ≤ 25 numbers bound in the text, **≤ 2 widgets** (one main, one small), ≤ 1 predict + 2 checks,
  a gotcha only where TorC really throws.
- **Widgets**: ≤ 2 controls plus Step/Reset; preset buttons instead of a slider where a slider adds nothing; a compact readout
  instead of explanations inside the widget; no visible tables (`srTable` gives screen readers the values).
- `site/test/lessons.test.mjs` asserts this budget for every lesson that is not `template: 3` yet (`lessonStats` in
  `site/test/helpers.mjs`). A `template: 3` lesson has the v3 budgets instead (`site/test/v3.mjs`, see "Add a lesson").
- The `predict` and `check` kinds and the `tldr` kind are removed once every lesson on the main path is `template: 3`.

## Add a lesson
1. **Trace data**: write the C++ trace program(s) (see "Add a scene"); the lesson lists the files it reads, e.g. `["why-gradients.json"]`.
2. **`site/course.json`**: add the lesson to its part with `{id, title, outcome, prereqs, symbols, trace, est_minutes, status}`.
   Add `widgets` (names shown on the home page) and `code` (the TorC lines the lesson is about, 2-6 lines; add its prelude and
   checks to `SPEC` in `tools/snippets/check.mjs` and run it). A lesson on the curriculum without a page yet is `"planned"`.
   Start the page with `"status": "draft"`: a draft has a page and is reachable at `lessons/<id>/`, but is not listed on the home page or in
   anyone's prev/next. Switch to `"published"` when the lesson has passed review. Optional `legacy_hashes: ["reuse", …]` forwards old
   `/#reuse/3` links from the home page to this lesson.
3. **Module `site/js/lessons/<id>.js`**: `export default {id, title, trace:[files], template: 3, loop, conceptual?, sources?,
   sections:[…], footer?: (D, T) => html}`. `id`, `title` and `trace` must equal the course.json entry (checked). `template: 3`
   turns on the v3 rules (below); a module without it keeps the old template until it is migrated. `loop` (`'data'`, `'model'`,
   `'loss'`, `'adjust'` or `'all'`) is the part of the training loop that the lesson lights in its loop diagram (main path only).
   `conceptual: true` marks content beyond TorC; it needs `sources: [{title, url}]`, which the default footer lists.
   Section kinds (all `(D, T) => …` fields may also be plain strings; option html sits in a `<label>`, so it is inline markup):
   - `{kind:'text', html:(D, T) => '<section>…</section>', setup?}` — markup is used as given. `D` is the loaded trace data
     (`D.scenes[id]`, `D.files[file stem]`, `D.anchors`); `T` puts numbers from it into prose (below). Must not touch the DOM.
     `setup: true` marks a text that only sets the scene (the notebook story) and answers nothing, so it may come before the
     first predict quiz.
   - `{kind:'questions', items:(D, T) => [html]}` — "Questions this lesson answers", 1-3 items, with the loop diagram. Section 1.
   - `{kind:'quiz', id, purpose, type, prompt, reveal, title?, from?, …}` — the learner answers first, then gets feedback, and the
     reveal opens after the first attempt. `purpose` is `'predict'` (before the text that answers it), `'check'`, `'transfer'`
     (a new situation, the same idea) or `'cumulative'` (with `from: '<earlier lesson id>'`). `type: 'choice'` has
     `options: [{html, correct, feedback}]`: 3-4 options, exactly one correct, and each wrong option's feedback names the
     misconception. `type: 'numeric'` has `answer: '<trace path>'`, an absolute `tolerance` above 0, and an optional `unit`.
   - `{kind:'explain', id, prompt, model}` — "Explain it in one sentence": a text box, then "Compare with a good answer".
   - `{kind:'widget', id, className?, label?, mount(host, D, ctx), tasks?:(D, T) => [{do, see}]}` — an interactive part; `host`
     is an empty `<section>`, and the guided tasks ("Task", "What you will see") follow it. `mount` runs in the browser only;
     return `{destroy()}` if it registers global listeners.
   - `{kind:'answer', html:(D, T) => html, next:(D, T) => html}` — "The answer", then "Next question" and a link to the next
     lesson, which `lesson-boot.js` takes from `neighbors()` in `core/course.js`. Last section.
   - `{kind:'gotcha', title, html}` — "Why did my X fail?" box with the recorded real message (appendix pages only in v3).
   - Legacy, until every lesson is migrated: `{kind:'tldr', items}` (the summary box), `{kind:'predict', id, prompt, reveal}` and
     `{kind:'check', id, prompt, reveal}` (open the answer; `check` also stores "I got it" as `seen`).
   The first section should be a hero containing the lesson's `<h1>`.

   **Blocks** for text sections (`site/js/core/blocks.js`, pure): `worked(html)` (the only place for `worked` numbers),
   `inTorc(title, html)` (a collapsed panel), `realLine(code, caption)` (code is plain text), `analogy(html, limit)` (from the
   brief's analogy bank, with "Where it breaks"), `chatgpt(html)`, `refresher(key)` (from `site/js/content/refreshers.js`),
   `term(key, text?)` (a glossary link with a hover, focus and tap definition from `site/glossary.json`), `sym(key, html?)` (a
   symbol linked to the notation guide, from `site/js/content/notation.js`) and `loopDiagram(active)`.

   **The v3 rules** (`checkV3` in `site/test/v3.mjs`, run by `lessons.test.mjs` on every `template: 3` lesson and proven on
   `site/test/fixtures/v3-fixture.js` in `v3.test.mjs`):
   - every bound number has a role; `worked` numbers only inside `worked()`, at most 3 worked blocks; `reveal` numbers only in a
     quiz reveal or feedback (or an explain model); at most 2 numbers per sentence outside worked blocks; at most 10 `anchor`,
     `contrast` and `human` numbers (code blocks and widgets do not count); no hand-typed digit in a quiz prompt, option,
     feedback or reveal;
   - Questions at section 1 with at most 3 items; The answer last, at most 80 words; a text section at most 150 words; at most 6
     numbered sections, 6 quizzes and 1,400 words; at most 2 of analogy, refresher, In TorC and ChatGPT;
   - a predict quiz before every text section that is not `setup: true`; a transfer quiz; an explain prompt; from the third
     main-path lesson on, a cumulative quiz whose `from` is earlier in course order; every widget has 1-3 tasks; no `tldr`,
     `predict` or `check`; a `gotcha` only on appendix pages;
   - every glossary term that the lesson uses is linked with `term()` at its first use (headings, code and the loop diagram are
     skipped); no symbol from `NOTATION` before the lesson that introduces it, in course order (a symbol whose lesson is not in
     course.json yet counts as not introduced); no LaTeX escape bug in the module source (`'\;'` is a bare `;`, write `'\\;'`,
     the same for `,` `!` `:` and a space; D1);
   - exceptions (brief §9): an appendix lesson (part `kind: "appendix"`) needs no Questions/Answer frame and no predict, transfer,
     explain or cumulative quiz, and may use every symbol. A `conceptual` lesson needs no predict before the text and no transfer.
4. **Shell**: `node tools/check_site.mjs --write-shells` writes `site/lessons/<id>/index.html` (a ~15-line committed file; the
   checker fails if a shell differs from the template).
5. `node --test site/test/` and `node tools/check_site.mjs`; then look at every step on desktop and phone width under a
   `/TorC/` prefix server.

### Numbers in prose
Never type a number the trace could change. In a `template: 3` lesson every number has a role, given in its binding (R2):

| call | role | where | shown as |
|---|---|---|---|
| `T(path, 'setup')` | setup | a constant or starting value of the scenario, or a value that a prompt gives the learner | exact |
| `T(path, 'worked')` | worked | only inside `worked(html)`, a calculation whose inputs are all visible | exact |
| `T(path, 'reveal')` | reveal | inside a quiz reveal or quiz feedback | exact |
| `T(path, 'contrast')` | contrast | half of an A-vs-B pair in a sentence that states the comparison in words | 2 significant figures |
| `T.anchor(name)` | anchor | a course-wide number from `site/anchors.json` | integers exact, else 2 significant figures |
| `T.human(path, kind, {of})` | human | a loss or a count in a unit a person can feel | see below |

- **Formatting** (`site/js/core/numbers.js`): an integer is always exact, with a comma every three digits from 1,000 up
  (`26,506`). "Exact" for other numbers means `fmt`, 4 significant figures. A negative number gets the minus sign U+2212.
  `opts.sig` (1-4), as in `T(path, 'contrast', {sig: 3})` or `T(path, {role, sig})`, overrides the role and then also rounds
  integers ("about 27,000"). `fmt` itself (`site/js/ops/texutil.js`) is unchanged: KaTeX and the graph lab use it.
- **`T.human` kinds**: `'rmse'` is √ of an MSE at 2 significant figures ("misses by about 0.24"); `'prob'` is 100·e^(−L) as a whole
  percent with a `%` sign, the probability given to the right class; `'pct'` is 100·v/of (or 100·v without `of`) as a whole
  percent; `'per100'` is mistakes per 100, 100 minus the whole-number `pct`, so "97%" and "3 per hundred" always add up. `of` is
  a trace path. No bare loss value outside worked or reveal numbers: say it with `T.human` (the reviewer checks this one).
- **Anchors**: add a name and its trace path to `site/anchors.json` (at most about 15, each used in at least two lessons), then
  run `node tools/anchors.mjs`, which writes `site/data/anchors.json`. `tools/trace/run.sh` runs it after the traces, and
  `anchors.test.mjs` fails when the file is stale. `lesson-boot.js` always loads it, so a lesson binds `T.anchor('mlp_parameters')`
  without loading `mnist-mlp.json`. The MNIST traces will get `setup.pixels` and `setup.classes`; add anchors for them then.
- **Derived values** (a step size, a share in percent) come from new trace fields or `T.human`, never from arithmetic in the lesson.
- **The span** is `<span class="num" data-trace="path" data-role="…" data-sig="exact|1-4">` (plus `data-human`, `data-of` or
  `data-anchor`; an anchor's `data-trace` is `anchors.<name>`). `lessons.test.mjs` re-formats every bound number from these
  attributes (`T.reformat`) and compares it with the page.
- **Legacy**: in a lesson without `template: 3`, the untagged `T('scene.path')` still gives `<span class="num" data-trace="…">`
  with `fmt` (no comma), so old lessons render exactly as before. A `template: 3` lesson must not use it (tested).
- **Paths**: `<sceneId>.<node>.<value|grad>[.<index>]` (a one-element tensor resolves to its number), `files.<file stem>.<key>…`
  for the extra tables of a trace file, or `anchors.<name>`; in arrays a segment is an index or an item's `name`. Every form of
  `T` **throws if the path is missing**, so prose can never silently drift from the data. `T.num(path)` / `T.text(path)` give the
  bare number / `fmt` text, for formulas, code and attributes. `site/test/lessons.test.mjs` renders every lesson in Node against
  the real JSON and rejects any `class="num"` without `data-trace`. If a number is unavoidable in a string that is not rendered
  through `T` (for example a scene takeaway), add a test that asserts it.

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
- **Concept text and TorC text.** The main text explains the idea with TorC's real numbers and is completable without C++. What
  **TorC does** in code, with the maths next to it, goes in "In TorC" panels and the appendix. Quirks of the library belong in
  `docs/library-audit/`, not on lesson pages (decision 4). The one exception is the appendix lesson `torc-pitfalls`, which shows
  recorded errors, including the `ShapeError` of a mismatched `MSELoss` (decision 22).
- Everything must work with browser storage blocked, without a pointer (keyboard), and with reduced motion; never autoplay.
- URLs in source are relative (`../../js/…`, `new URL('data/x.json', SITE)`); never root-absolute and no `<base>`.
- Formula and code boxes in the prose get `tabindex="0" role="group" aria-label="…"`, so a keyboard user can scroll a box that is
  wider than the screen. `lesson-boot.js` removes the tab stop from a box that does not scroll, and adds it back on resize.
- A widget readout that follows the mouse is not an `aria-live` region; announce the change on focus and click only.

### Notation (decision 14, brief section 8; one guide, `notation` in the appendix)
- **Gradient.** From its first use until `backprop`, write dL/dw and read it aloud as "how much the loss changes per unit change of
  w". Introduce ∂ once, in `backprop`: "the same as d, used when the loss depends on many weights". The overline (`ā` for ∂out/∂a)
  appears only inside the graph-lab maths windows (`explain.js`, `ops/*`) and is defined there. Prose never mixes the three
  notations. The matrix rule for a layer's backward pass is written with ∂ in the `backprop` "In TorC" panel.
- **Indices.** Tensor positions, rows, columns and axes count from 0 everywhere: prose, widgets and the TeX in `ops/*` and
  `linear-math.js`. Say so where a lesson first uses them. Counts of epochs, steps and batches in readouts and prose count from 1,
  and so do ordinal words ("the first neuron"). TorC code that loops from 0 is shown as it is.
- **Symbols.** One guide page lists ←, Σ, subscripts, ᵀ, ⌊ ⌋, hats, the overline and the Greek letters with their names (μ mu, β
  beta, λ lambda, ε epsilon, σ sigma). `site/js/content/notation.js` names the lesson that introduces each symbol. A symbol links
  to the guide at its first use (`sym(key, html)`), and a test rejects a symbol before its introducing lesson, in course order.
  Explain ← as "replace `w` by".
- **Logs.** "log" always means ln, and the page says so at first use. Write ln in formulas where the base matters.
- **One letter per thing.** The batch size is `B`, never `N`. The prediction is `y` (or ŷ once defined). A parameter is `w` or `W`.
  Do not use `p` for two things, and do not use `g` for the gradient. Write `lr` upright so it does not read as l × r.
- **Refreshers and glossary.** A maths refresher is collapsible, 100 to 200 words, with a picture (derivative as slope found by
  nudging, Σ notation, vectors and the dot product, the matrix product, exp and ln, probability, mean and standard deviation, how
  to read a contour map, how to read a log axis). `site/glossary.json` is the one source for term definitions.
