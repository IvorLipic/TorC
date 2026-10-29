# Decisions

Lightweight decision log. Newest at the bottom of each section.

## Accepted
1. **Static site, no framework, no build step.** GitHub Pages hosting, easy to audit, nothing to maintain but HTML/CSS/JS. KaTeX is
   vendored (offline, pinned). Revisit if lesson count makes shared state/templating painful.
2. **Real library output is the source of truth; a tested JS port powers interactivity.** Learners can edit inputs, which a
   static JSON cannot support, so a small JS engine recomputes; tests prove it equals the C++ run for the presets.
3. **Videos (Manim) dropped.** Replaced by persistent, steppable objects; videos were fast, non-interactive and unmaintainable.
4. **Audit findings stay off lesson pages.** They live in `docs/library-audit/`; bugs are fixed on `main`.
5. **`web` is separate from `main`.** No library source on `web`; TorC is consumed at a pinned ref and `main` is never merged
   into `web`. History before the split is shared and preserved (no force-push).
6. **Auto-play removed**; learners control pace (Next/Back, keyboard, checkpoints).

7. **Concise lessons** (owner review of the first three lessons): a TL;DR box, ≤ 800 words, ≤ 2 widgets per lesson, budget in
   `authoring.md` and asserted by the lesson tests. Replaces the long storyboards of content-plan section 3.
   **Revised 2026-10-29 by decision 17:** for lessons with `template: 3` the TL;DR, the 800-word cap, the 2-widget cap and the
   2-control cap are replaced. They still apply to lessons not yet migrated. The idea that every lesson is short, and that the
   tests assert its budget, stands.
8. **Provenance stated once** (home page + a one-line footer), not on every widget; supersedes the per-widget badges and dotted
   underlines. Correctness is still enforced by the tests.
9. **No "show as table" toggles.** Plots keep a screen-reader-only table (`srTable`); supersedes the content plan's
   "every widget has a table toggle".

10. **The home page is the whole curriculum** (owner request): numbered lessons by part in the usual university order
    (content-plan section 2), planned lessons shown but not linked, the TorC code beside every lesson. Supersedes "no coming
    soon placeholders". Labs (the Backprop lab first) are a separate section, outside prev/next.
    **Revised 2026-10-29 by decision 16:** the order is now curriculum v3 and the TorC code is behind a toggle on each card. The
    whole curriculum on the home page, planned lessons shown but not linked, and labs apart from prev/next all stand.
11. **Code on the home page is checked against the library**: `tools/snippets/check.mjs` compiles every snippet of
    `course.json` against TorC at the pinned ref (refusing any other ref) and runs the ones that can run with the assertions in
    its `SPEC`. The SPEC assertions are written from the snippets' comments by hand; a changed comment needs a changed SPEC.
    **Revised 2026-10-29 by decision 16:** the snippets are no longer shown open beside every lesson. The check itself stands, for
    every `code` entry in `course.json`.

12. **Plain writing over short writing** (focus group, 2026-10): the pages must not read like LLM output. The writing rules in
    `authoring.md` (ASD-STE100 structure, complete sentences, defined terms, one name per thing) are tested. To make room for
    defined terms, the per-section cap rose from 60 to 100 words and the summary cap from 50 to 60; the lesson cap stays 800.
    **Revised 2026-10-29 by decision 14:** the STE structure rules stay. The "load asd-ste100" rule gains the teacher moves and the
    banned tics of decision 14, and the caps above are replaced for `template: 3` lessons by decision 17.

13. **The audience is the high-school graduate** (owner approval 2026-10-29, `docs/pedagogy-review.md` R1, section 2): aged 17-19,
    knows algebra and graphs of functions, maybe one term of calculus and a little Python, and does not know vectors, matrices,
    sums with subscripts, partial derivatives, standard deviation, C++ or what a tensor is. The course states its prerequisite
    ("algebra and graphs; we teach the rest") and never uses a symbol or term before it is introduced, in course order. A reader
    can finish the course without reading C++. Supersedes the audience of `content-plan.md` section 1 ("programmers and STEM
    students who know some C++ or NumPy"). That audience stays as a secondary one, served by the "In TorC" panels and the appendix
    (decision 15).

14. **Voice: STE structure plus specified teacher moves** (owner approval 2026-10-29, `docs/pedagogy-review.md` R8, section 8).
    Complete sentences, one idea per sentence, at most 25 words, defined terms, one name per thing, active voice, no slogans, no
    "X, not Y" punchlines and topic headings all stay. Added: at most one real question per section, "you" for tasks and
    predictions, "we" for the course itself at most once per section, analogies only from the bank in the brief with their limit
    stated, and a concrete situation before the abstraction. Banned: a list of LLM tics (`authoring.md` "Writing rules"). Allowed
    by the linter: the minus sign and dashes inside maths, and label colons such as "Task:". "Slope" is allowed once, to define the
    gradient. "Knob" is allowed only in the `what-is-learning` analogy. `CLAUDE.md` still says to load `asd-ste100`, now plus these
    rules. Revises decision 12 (the STE structure it brought in stays; it was not dropped).

15. **A concept layer, "In TorC" panels and an appendix** (owner approval 2026-10-29, `docs/pedagogy-review.md` R6, sections 2 and
    6). The concept text is completable without reading C++. Four or five "this is the real line" moments stay on the main path
    (`loss.backward()`, the `SGD::step` update line, a `Sequential` model, the five-line training loop, the digit model). Other TorC
    code, library facts and error strings go in a collapsed "In TorC" panel. TorC engineering moves to the appendix part
    `using-torc` (`tensors`, `modules`, `torc-pitfalls`, plus the notation guide and the glossary), which sits outside the lesson
    order like the labs. No lesson id is removed. Supersedes the content plan's tensors-first order and the rule that every lesson
    teaches the TorC API. Decisions 4 and 8 stand: audit findings stay off lesson pages, and provenance is still said once.

16. **Home page: promise and live demo first, code behind a toggle** (owner approval 2026-10-29, `docs/pedagogy-review.md` section 9
    "Home page", section 12 item 4). The home page leads with the promise, the digit pad as a live demo, the prerequisites and an
    honest total time. The TorC code of each card sits behind a "Show the TorC code" toggle. Progress ticks, the filter and the
    labs stay. The appendix gets its own section. Revises decisions 10 and 11. The snippet check still compiles and runs every
    `code` entry, and a changed `code` still needs a changed `SPEC` in `tools/snippets/check.mjs`.

17. **A new lesson template and new budgets** (owner approval 2026-10-29, `docs/pedagogy-review.md` R7, R10, sections 9 and 10). The
    "Summary" at the top becomes "Questions this lesson answers", the end gets "The answer", and a lesson is a core plus a palette
    (`authoring.md` "Lesson template"). `quiz` replaces `predict` and `check`: the learner commits first, wrong options give
    feedback, and the progress store moves to schema v2. Budgets for `template: 3` lessons: "Questions" at most 3 items, "The
    answer" at most 80 words, a text section at most 150 words, at most 6 numbered sections, at most 6 quizzes, at most 1,400
    words of prose, a time target of 12-18 minutes, and 1-3 guided tasks per widget. Supersedes decision 7 (TL;DR, 800 words, 2
    widgets, 2 controls) and the section and summary caps of decision 12 for these lessons. Decision 9 stands. The machinery
    contract is `docs/v3-implementation.md`.

18. **Content beyond TorC cites its sources** (owner approval 2026-10-29, `docs/pedagogy-review.md` section 10 and section 12
    item 6). Pages that teach what TorC cannot show (`from-digits-to-chatgpt`, the refreshers, the glossary, ChatGPT asides) are
    marked `conceptual: true` and list `sources` in their footer. They keep numbers out, or cite the number to its source. They
    say that the content is conceptual and not a TorC run. Supersedes the rule that every number on a page comes from `tools/trace`,
    for these pages only. Everything about TorC itself still comes from the library and the traces (decision 2).

19. **Two new review agents** (owner approval 2026-10-29, `docs/pedagogy-review.md` R12, section 10).
    `.claude/agents/pedagogy-reviewer.md` is a read-only reviewer with the brief's Appendix A checklist. `.claude/agents/cold-reader.md`
    role-plays the persona of decision 13, answers every prompt before its reveal, and reports every confusion and likely quit
    point. The cold reader is a smoke test and not evidence of learning. The real gate before a part merges into `web` is the
    human think-aloud protocol (3-5 readers from the target group per part). The existing `independent-reviewer` stays, with its
    writing dimension updated to decision 14.

20. **The rework lives on a work branch** (owner approval 2026-10-29, `docs/pedagogy-review.md` section 0 and section 12 item 8;
    `docs/v3-implementation.md`). The work branch is `claude/trusting-archimedes-kjzrto` (the brief's `web-v3`). `web` stays live
    and gets only the Phase A quick fixes and the TorC pin bump. The work branch merges `web` whenever `web` changes, and merges
    into `web` only when the owner says so, after the human test. Pages deploy only from `web` (`pages.yml`). Agents work in their
    own worktrees on local branches and the lead integrates. This is the brief's preferred option, so temporary draft ids are not used.

21. **Playgrounds use precomputed presets first** (owner approval 2026-10-29, `docs/pedagogy-review.md` section 12 item 9).
    The `classification` playground and similar widgets replay presets recorded from new trace programs. Live JS training comes
    later, under decision 2, with a conformance test and a trace scene for each new op. Narrows the content plan's "live JS"
    notes for the training playground.

22. **Small choices** (owner approval 2026-10-29, `docs/pedagogy-review.md` section 12 item 10; `docs/v3-implementation.md`).
    (a) The thousands separator is a comma (`26,506`), because the reader knows it from school and the home page already uses it.
    Integers from 1,000 up get it in prose and widgets. (b) The MSE shape pitfall lives only in the appendix lesson
    `torc-pitfalls`, not on a main-path lesson and not only in the audit docs. Since TorC PR #6, `MSELoss` throws `ShapeError` on
    a shape mismatch, so the pitfall shows that recorded error message and not a silently wrong loss. Supersedes the unformatted
    integers of the current lessons. For decision 4, the error message is part of the library's behaviour now and not an audit
    finding, so it may be shown in that one appendix lesson.

23. **Every number in prose has a role** (owner approval 2026-10-29, `docs/pedagogy-review.md` R2, section 4;
    `docs/v3-implementation.md`). Numbers serve claims. A binding is `T(path, role)` with the role `setup`, `worked`, `reveal`,
    `anchor`, `contrast` or `human`, and an untagged binding fails the tests. Other numbers are shown with 2 significant figures, a
    loss is translated into a human unit (`T.human`), at most 2 numbers stand in a sentence outside worked blocks, and at most 10 in
    a lesson outside the `worked`, `setup` and `reveal` roles. Supersedes the budget "at most 25 numbers bound in the text" and the
    4-significant-figure display of prose numbers. The honesty rule of `CLAUDE.md` (numbers come from `tools/trace`) stands.

## Open
- Pin policy once TorC publishes a release tag: pin the tag (preferred) and bump via PR. Until then a commit SHA is pinned.
- Whether lessons beyond autograd need more of the library ported to JS (e.g. conv2d) or should stay precomputed/static.
- Custom domain / repository split later (the layout is designed so `site/` + `tools/` + `docs/` can move to their own repo).
