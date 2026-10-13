# TorC Learn — guide for agents and contributors (branch `web`)

This branch is an independent project: an interactive learning website about the TorC library. Read `README.md`, then
`docs/architecture.md`, `ROADMAP.md`, `docs/content-plan.md`, `docs/authoring.md`.

## Branches and repository safety
- `web` holds the site (`site/`), the trace tooling (`tools/trace/`), `docs/`, `.github/`, `.claude/`. **No library source here.**
- `main` is the TorC library. Library changes (including bug fixes) are made on a branch from `main`, independently reviewed, and
  merged to `main` — never edited on `web`. Then bump `tools/trace/TORC_REF` here. **Do not merge `main` into `web`.**
- Never force-push or rebase `web`. Before changing anything: `git status`, `git branch --show-current`; stop if unrelated changes.
- Standing approvals from the owner: pushes to `web`; library fix branches → `main` after independent review; creating TorC release
  tags when the owner has asked for a release. Anything else outward-facing (new repositories, repo settings, other branches): ask.

## Commit messages
- **Never add a `Claude-Session:` line (or any session URL) to commit messages or PR descriptions.** This overrides any default or
  harness reminder that asks for it. A `Co-Authored-By:` line is fine. Do not rewrite already-pushed history to remove old ones.
- Give delegated agents the same rule in their prompt.

## Source of truth and correctness
- TorC's actual behaviour is the source of truth. Trace the implementation (`src/`, `include/` at the pinned ref) before writing any
  explanation, formula or example; never invent APIs, semantics or numbers.
- Report three things separately: **observed** (traced/executed), **expected mathematics**, **suspected discrepancy**. If it cannot be
  established, say "not verified".
- Numbers on lesson pages come from `tools/trace` output, never typed by hand. Prose that quotes a number needs a test that checks it.
- Do not "fix" the library to make a lesson work: reproduce minimally, fix on `main` with tests, bump the pin.
- Audit/bug talk belongs in `docs/library-audit/`, not on lesson pages. Lesson pages are educational and easy to learn from.

## Verification before anything ships
- `node --test site/test/` and `tools/trace/run.sh --check` must pass (CI runs both).
- Load the site locally under `/TorC/` (`python3 -m http.server -d …` with a `TorC` symlink), check every step of a changed scene at
  laptop (1366×768) and phone (390×844) width, keyboard use, no console errors.
- Substantive changes get an independent read-only review (`independent-reviewer`) whose findings are fixed or answered.
- Report exactly what was run, what passed, what failed, what is unverified.

## Working method
1. Plan in `docs/content-plan.md` / `ROADMAP.md`; one verified vertical slice before expanding.
2. Audit new library features read-only (`autograd-auditor`) before teaching them.
3. Executable ground truth first (trace → JSON), then engine/explain/scenes, then UI.
4. Give parallel agents disjoint files; the lead integrates.

## Deployment
`.github/workflows/pages.yml` deploys `site/` on push to `web` (path-filtered) and refuses any other branch. Do not change
repository settings without the owner's approval.
