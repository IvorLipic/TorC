# TorC Learn — guide for agents and contributors (branch `web`)

This branch is an independent project: an interactive learning website about the TorC library. Read `README.md`, then
`docs/architecture.md`, `ROADMAP.md`, `docs/content-plan.md`, `docs/authoring.md`. The course is being reworked for the
high-school graduate (curriculum v3): `docs/pedagogy-review.md` is the brief, `docs/v3-implementation.md` is the owner's decisions
and the machinery contract, and `docs/decisions.md` (13 onward) records what is approved.

## Branches and repository safety
- `web` holds the site (`site/`), the trace tooling (`tools/trace/`), `docs/`, `.github/`, `.claude/`. **No library source here.**
- `main` is the TorC library. Library changes (including bug fixes) are made on a branch from `main`, independently reviewed, and
  merged to `main` — never edited on `web`. Then bump `tools/trace/TORC_REF` here. **Do not merge `main` into `web`.**
- Never force-push or rebase `web`. Before changing anything: `git status`, `git branch --show-current`; stop if unrelated changes.
- Standing approvals from the owner: pushes to `web`; library fix branches → `main` after independent review; creating TorC release
  tags when the owner has asked for a release. Anything else outward-facing (new repositories, repo settings, other branches): ask.
- **Work branch (decision 20).** The v3 rework lives on `claude/trusting-archimedes-kjzrto` (the brief's `web-v3`). `web` stays live
  and gets only the Phase A quick fixes and the TorC pin bump, because both are safe and tested. The work branch merges `web`
  whenever `web` changes. It merges into `web` only when the owner says so, after the human-reader test (see Verification). Never
  merge `main` into either branch. Pages deploy only from `web`. Agents work in their own worktrees on local branches; the lead
  integrates.

## Commit messages
- **Never add a `Claude-Session:` line (or any session URL) to commit messages or PR descriptions.** This overrides any default or
  harness reminder that asks for it. A `Co-Authored-By:` line is fine. Do not rewrite already-pushed history to remove old ones.
- Cite the brief's ids (R1-R12, C1-C10, D1-D20) in the message. End each commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Give delegated agents the same rule in their prompt.

## Source of truth and correctness
- TorC's actual behaviour is the source of truth. Trace the implementation (`src/`, `include/` at the pinned ref) before writing any
  explanation, formula or example; never invent APIs, semantics or numbers.
- Report three things separately: **observed** (traced/executed), **expected mathematics**, **suspected discrepancy**. If it cannot be
  established, say "not verified".
- Numbers on lesson pages come from `tools/trace` output, never typed by hand. Prose that quotes a number needs a test that checks it.
- **Content beyond TorC** (`from-digits-to-chatgpt`, refreshers, the glossary, ChatGPT asides; decision 18): cite a source for every
  factual claim in the page footer (`conceptual: true` and `sources`), keep numbers out, or cite each number to its source (never
  from memory), and label the content as conceptual and not a TorC run. Never claim that TorC does something it does not. Anything
  about TorC itself still comes from the library and the traces.
- Do not "fix" the library to make a lesson work: reproduce minimally, fix on `main` with tests, bump the pin.
- Audit/bug talk belongs in `docs/library-audit/`, not on lesson pages. Lesson pages are educational and easy to learn from.

## Writing (every sentence a visitor sees)
- Before writing or editing visible text (lesson prose, questions, quizzes, graph-lab texts, widget labels, course descriptions,
  the home page), load the `asd-ste100` skill (`.claude/skills/asd-ste100`, vendored) and write in its **STE-flavored** mode,
  plus the v3 voice rules in `docs/authoring.md` ("Writing rules", decision 14): complete sentences, at most 25 words, terms
  defined before use and not before their lesson, one name per thing, no semicolons, dashes or colon punchlines. The reader is the
  high-school graduate of decision 13. Allowed teacher moves: at most one real question per section, "you" for tasks and
  predictions, analogies from the bank with their limit stated, a concrete situation first. Banned: the list of LLM tics in
  authoring.md. Number roles, notation and the lesson template are in authoring.md too.
- `tools/prose/check.sh` must report 0 hard findings, and `node --test site/test/` checks the site rules (CI runs both).
- Give delegated agents that write visible text the same instruction.

## Verification before anything ships
- `node --test site/test/`, `tools/trace/run.sh --check` and `tools/prose/check.sh` must pass (CI runs them).
- After changing any `code` in `site/course.json`: `node tools/snippets/check.mjs --torc <TorC checkout at TORC_REF>` must pass.
- Load the site locally under `/TorC/` (`python3 -m http.server -d …` with a `TorC` symlink), check every step of a changed scene at
  laptop (1366×768) and phone (390×844) width, keyboard use, no console errors.
- Substantive changes get an independent read-only review (`independent-reviewer`) whose findings are fixed or answered.
- **Lesson changes need two more gates** (decision 19). The `pedagogy-reviewer` (read-only, the checklist of the brief's Appendix A)
  must report no High findings. The `cold-reader` role-plays the high-school persona and reports every confusion and likely quit
  point. The cold reader is a smoke test, not evidence of learning: never report its result as "readers understood it".
- **The human think-aloud protocol is the real gate before a part merges into `web`.** Test each part with 3 to 5 people from the
  target group, thinking aloud while one person observes. Before and after each lesson, ask three short questions, including
  "explain the main idea in one sentence". Record where each person stopped. The site has no backend and no analytics, so this
  is the only learning data there is. Agents cannot do it; the owner decides when it is done and when the work branch merges.
- Report exactly what was run, what passed, what failed, what is unverified.

## Working method
1. Plan in `docs/content-plan.md` / `ROADMAP.md`; one verified vertical slice before expanding.
2. Audit new library features read-only (`autograd-auditor`) before teaching them.
3. Executable ground truth first (trace → JSON), then engine/explain/scenes, then UI.
4. Give parallel agents disjoint files; the lead integrates.
5. For lessons: the brief's acceptance criteria (`ROADMAP.md`, v3 milestone) are the definition of done. Build one part at a time and
   test it with readers before the next.

## Deployment
`.github/workflows/pages.yml` deploys `site/` on push to `web` (path-filtered) and refuses any other branch. Do not change
repository settings without the owner's approval.
