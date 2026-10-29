# TorC Learn v3: implementation spec

This file turns `docs/pedagogy-review.md` (the brief) into concrete work. Read the brief first. Where the two disagree, this
file wins for implementation details (API names, file ownership), and the brief wins for teaching goals.

## Owner decisions (2026-10-29)

The owner approved every approval item of brief §12 as the brief recommends:

1. Audience: the high-school graduate of brief §2 (R1).
2. Voice: STE structure stays, plus the teacher moves and banned tics of brief §8 (R8).
3. Concept layer: C++ moves to "In TorC" panels, 4–5 "real line" moments stay, `tensors` and `modules` move to an appendix (R6).
4. Home page: promise and live demo first, code behind a toggle (revises decisions 10 and 11).
5. Template: "Questions" and "The answer" replace the TL;DR, new budgets (revises decision 7).
6. Content beyond TorC: sources are cited, numbers are left out or cited.
7. New agents: `pedagogy-reviewer` and `cold-reader`.
8. Work branch: the rework lives on `claude/trusting-archimedes-kjzrto` (the brief's `web-v3`). Pages deploy only from `web`.
9. Playgrounds: precomputed presets from trace programs first. Live JS training later, under decision 2.
10. Small choices: thousands separator is a **comma** (`26,506`), because the reader knows it from school and the home page
    already uses it. The MSE shape pitfall lives in the appendix lesson `torc-pitfalls` only. Since TorC PR #6, `MSELoss`
    throws `ShapeError` on a shape mismatch, so the pitfall shows that recorded error message.

## Branches

- `web` is live. It gets the Phase A quick fixes and the TorC pin bump (PR #6), because both are safe and tested.
- `claude/trusting-archimedes-kjzrto` is the work branch. It merges `web` whenever `web` changes. It merges into `web` only when
  the owner says so, after the human-reader test of brief §10.
- Never merge `main` into either branch. Never force-push `web`.
- Agents work in their own worktrees on local branches. The lead integrates.

## Commit rules (all agents)

- Never add a `Claude-Session:` line or any session URL. End each commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Cite the brief's ids in the message (R1–R12, C1–C10, D1–D20).

## Machinery API (the contract every lesson uses)

A lesson opts into the v3 rules with `template: 3` in its module. Lessons without it keep the old rules until they are
migrated. At the end every lesson on the main path is `template: 3`, and the old kinds (`tldr`, `predict`, `check`) are removed.

### Lesson module

```js
export default {
  id, title, trace: ['x.json'], template: 3,
  loop: 'data' | 'model' | 'loss' | 'adjust' | 'all',   // the part of the loop diagram this lesson lights (main path only)
  conceptual: false,                                     // true for content beyond TorC (needs `sources`)
  sources: [{ title, url }],                             // cited in the footer when conceptual
  sections: [ hero, questions, …, answer ],
}
```

### Section kinds

| kind | fields | renders |
|---|---|---|
| `text` | `html: (D, T) => html` | as given |
| `questions` | `items: (D, T) => [html]` (1–3) | "Questions this lesson answers" box, with the loop diagram when `loop` is set. Section 1, right after the hero |
| `quiz` | `id, purpose, prompt, type, options \| answer, tolerance?, unit?, reveal, from?` | a committed question: answer first, then feedback and the reveal |
| `explain` | `id, prompt, model` | "Explain it in one sentence": a text box, then "Compare with a model answer" |
| `widget` | `id, mount, label?, className?, tasks?: (D, T) => [{ do, see }]` | the host, then the guided tasks ("Task", "What you will see") |
| `answer` | `html: (D, T) => html, next: (D, T) => html` | "The answer", then "Next question: …" and a link to the next lesson filled from `neighbors()`. Last section |
| `gotcha` | as today | appendix pages only |

`quiz` details:
- `purpose`: `'predict'` (before the text that answers it), `'check'`, `'transfer'` (a new situation, the same idea),
  `'cumulative'` (needs `from: '<lesson id>'`, an earlier lesson in course order).
- `type: 'choice'`: `options: [{ html: (D, T) => html, correct: bool, feedback: (D, T) => html }]`, 3–4 options, exactly one
  correct. Each wrong option's feedback names the misconception.
- `type: 'numeric'`: `answer: '<trace path>'` (resolved with `T.num`), `tolerance` (absolute), optional `unit`.
- The reveal opens only after an attempt. The feedback region is `aria-live="polite"`. Works with keyboard only, and with
  storage blocked. The store records attempts and correctness, never a self-report.

### Numbers: `T(path, role, opts?)`

Every number in a v3 lesson has a role. An untagged `T(path)` fails the v3 tests.

| role | where | shown as |
|---|---|---|
| `setup` | a constant or starting value of the scenario, or a value a prompt gives the learner | exact (`fmt`) |
| `worked` | only inside `worked(html)`, a calculation whose inputs are all visible. At most 3 worked blocks per lesson | exact |
| `reveal` | inside a quiz reveal or quiz feedback | exact |
| `anchor` | `T.anchor(name)`, from `site/data/anchors.json` | integers exact, else 2 significant figures |
| `contrast` | half of an A-vs-B pair in a sentence that states the comparison in words | 2 significant figures |
| `human` | `T.human(path, kind, opts)` | see below |

- Formatting: integers are exact with a comma every three digits from 1,000 up. Other numbers: exact roles use `fmt`
  (4 significant figures), the rest 2 significant figures. `opts.sig` (1–4) overrides. The span carries `data-role` and
  `data-sig` so the tests can re-format and compare.
- `T.human(path, kind, { of })`: `'rmse'` (√ of an MSE, "off by about 0.24"), `'prob'` (100·e^−L as a whole percent, the
  probability given to the right class), `'pct'` (100·v/of, or 100·v when `of` is absent, whole percent), `'per100'`
  (mistakes per 100: 100 − pct). `of` is a trace path. The span carries `data-human` and `data-of`.
- Limits per v3 lesson (tested): at most 2 numbers per sentence outside worked blocks; at most 10 numbers outside worked,
  setup and reveal roles; numbers in code blocks and widgets do not count. No bare loss value outside worked or reveal: say it
  with `T.human` (the reviewer checks this one).
- Derived values (a step size, a share in percent) come from new trace fields, never from arithmetic in the lesson.
- Anchors: `site/anchors.json` lists `{ name: '<trace path>' }` (at most about 15, each used in at least two lessons).
  `node tools/anchors.mjs` writes `site/data/anchors.json` with each value. A test checks that the values still equal the traces.
  `lesson-boot.js` always loads it.

### Blocks for text sections (`site/js/core/blocks.js`, pure)

`worked(html)`, `inTorc(title, html)` (collapsed `<details>`), `realLine(code, caption)` (the 4–5 "this is the real line"
moments), `analogy(html, limit)` (from the brief's analogy bank only, "Where it breaks:" stated), `chatgpt(html)` (the
"same thing in ChatGPT" aside), `refresher(key)` (a collapsible maths refresher from `site/js/content/refreshers.js`),
`term(key, text?)` (a glossary link with a hover and tap definition from `site/glossary.json`), `sym(key, html)` (a symbol
linked to the notation guide), `loopDiagram(active)`.

- Palette: 0–2 of analogy, refresher, In TorC, ChatGPT per lesson.
- Glossary: every glossary term a lesson uses is linked with `term()` at its first use in that lesson (tested).
- Notation: `site/js/content/notation.js` lists each symbol with the lesson that introduces it. No symbol appears in a lesson
  before its introducing lesson, in course order (tested).

### Course

- `part.kind` is `'lab'`, `'appendix'` or absent. The curriculum excludes labs and the appendix. Prev/next stay within a kind.
- Part ids: `whole-game`, `gradient`, `networks`, `training`, `images`, `next`, and the appendix `using-torc`. Lessons are
  numbered from 1.
- Appendix pages: `tensors`, `modules`, `torc-pitfalls`, `notation`, `glossary`.

### Budgets for v3 lessons (tested)

"Questions" ≤ 3 items. "The answer" ≤ 80 words. A text section ≤ 150 words. ≤ 6 numbered sections. ≤ 6 quizzes. ≤ 1,400
words of prose. At least one `predict` quiz before any text that answers it. One `transfer` quiz and one `explain`. One
`cumulative` quiz from the third lesson on. Every widget has 1–3 tasks. Exceptions (brief §9): `from-digits-to-chatgpt` and
the appendix pages need no worked example, widget or refutation.

### Store v2

`localStorage["torc:v2"] = { "<lessonId>": { visited, done, quiz: { "<id>": { tries, correct } }, explained: { "<id>": true },
seen: { "<id>": true } } }`. On first load, v1 data migrates: v1 `checks` become `seen`, never `correct`.

### Machinery as built (wave 1): choices and deviations

The machinery implements this API. Where the contract left a choice open, or could not work as written, this is what was built.
The details are in `docs/authoring.md` ("Add a lesson", "Numbers in prose") and in the file heads of `site/js/core/`.

1. **Scenario text before the first predict.** "A predict quiz before any text section" would forbid the gold-standard opening of
   brief Appendix B ("The price of a notebook" comes before the quiz). A `text` section may carry `setup: true`: it sets the
   scene and answers nothing. The test requires a predict quiz before every text section that is not `setup: true`. The
   pedagogy reviewer checks that a setup text really answers nothing.
2. **The next-lesson line stays pure.** `answer` renders "The next lesson is …" from `ctx.next`, which `lesson-boot.js` passes
   from `neighbors()` (`renderLesson(lesson, D, {course, next})`). Node renders the same line, so the D2 test compares it with
   `neighbors()`. Without a next lesson, the line links the course map.
3. **Labels end with a full stop, not a colon** ("Analogy.", "Where it breaks.", "In TorC.", "Refresher."). Today's
   `proseProblems` flags a colon inside a block, and the full stop works under both the old and the new linter.
4. **`quiz.correct` is the result of the latest attempt**; `tries` counts every attempt. A reload shows the reveal of a quiz that
   was tried before. The `explain` model answer opens only after the learner typed a sentence; the sentence is not stored.
5. **`T(path, role, opts)` also accepts `T(path, {role, sig})`** (the brief's form). `T(path, 'anchor')` and `T(path, 'human')`
   throw and name `T.anchor` / `T.human`. `T.anchor(name)` resolves `anchors.<name>` in `D.anchors`.
6. **`T.human`**: `prob` and `pct` show a `%` sign ("97%"); `per100` is 100 minus the whole-number `pct`, so the two always add
   to 100. `sig` applies to `rmse` only.
7. **Exceptions** (brief §9): an appendix lesson needs no Questions/Answer frame and no predict, transfer, explain or cumulative
   quiz, and may use every symbol. A `conceptual` lesson needs no predict before the text and no transfer quiz, but it needs an
   explain prompt, a cumulative quiz from the third lesson on, and `sources` (which the default footer lists).
8. **Also tested**: the palette limit (at most 2 of analogy, refresher, In TorC, ChatGPT), no legacy kind in a v3 lesson, a
   `gotcha` only on appendix pages, no hand-typed digit in a quiz prompt, option, feedback or reveal, and a numeric quiz's
   answer path. A notation symbol whose introducing lesson is not in `course.json` yet counts as not introduced.
9. **Not tested yet**: "each anchor is used in at least two lessons" (no lesson is migrated), "no bare loss value" (the reviewer
   checks it), and the contrast sentence wording. The MNIST anchors `setup.pixels` and `setup.classes` come with the MNIST trace
   change; `site/anchors.json` lists only paths that exist today.
10. **`site/data/anchors.json` is not a trace file.** `tools/trace/run.sh` writes it with `node tools/anchors.mjs` after the
    traces, and `--check` runs `node tools/anchors.mjs --check` and knows the file in its "no generator" check.

## Curriculum v3 (ids in order)

- Part 1 `whole-game`: `what-is-learning`.
- Part 2 `gradient`: `why-gradients`, `sgd-step`.
- Part 3 `networks`: `linear-layer`, `nonlinearity`, `classification`, `losses`, `backprop`, `deep-nets`.
- Part 4 `training`: `training-loop`, `optimizers`, `generalization`.
- Part 5 `images`: `mnist-mlp`, `one-step`, `convolution`, `mnist-cnn`.
- Part 6 `next`: `from-digits-to-chatgpt`.
- Appendix `using-torc`: `tensors`, `modules`, `torc-pitfalls`, `notation`, `glossary`.
- Labs: `backprop-lab`.

## Work waves and file ownership

| Wave | Work | Owns |
|---|---|---|
| 0 | PR #6 review, pin prep, impact audit | `tools/trace/trace_losses.cpp`, `trace_linear.cpp`, `docs/library-audit/`, D6 in `losses.js` |
| 0 | Phase A quick fixes (to `web`) | the lesson and widget lines named in brief §11 Phase A |
| 1 | Machinery | `site/js/core/*`, `site/js/lesson-boot.js`, `site/test/{core,lessons,site}.test.mjs`, `site/test/helpers.mjs` (except `proseProblems`), `site/css/base.css`, `site/js/content/`, `site/glossary.json`, `site/anchors.json`, `tools/anchors.mjs` |
| 1 | Course kinds and home page | `site/js/core/course.js` (kinds only), `site/js/home.js`, `site/index.html`, `site/css/home.css`, `site/course.json` |
| 1 | Writing linter | `proseProblems` in `site/test/helpers.mjs`, `tools/prose/`, `.github/workflows/verify.yml` prose step |
| 1 | Docs and agents | `docs/decisions.md`, `docs/content-plan.md`, `docs/authoring.md` (policy sections), `CLAUDE.md`, `ROADMAP.md`, `.claude/agents/` |
| 1 | Traces, non-MNIST | new and changed trace programs except MNIST ones and the wave-0 files |
| 1 | Traces, MNIST | `trace_mnist_mlp.cpp`, `trace_mnist_cnn.cpp`, new `trace_one_step.cpp` |
| 2 | Vertical slice | `what-is-learning`, `why-gradients`, `sgd-step` (module, widgets, tests, `docs/lessons/<id>.md`) |
| 3 | Parts 3–6 and the appendix | one agent per part, disjoint lesson files |
| 4 | Review, integration, polish | pedagogy reviewer and cold reader on every lesson; fixes; light theme |

## What cannot be done by agents

The real gate of brief §10 is 3–5 human readers from the target group per part. The cold reader is a smoke test only. The
owner decides when the work branch merges into `web`.
