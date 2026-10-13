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

## Open
- Pin policy once TorC publishes a release tag: pin the tag (preferred) and bump via PR. Until then a commit SHA is pinned.
- Whether lessons beyond autograd need more of the library ported to JS (e.g. conv2d) or should stay precomputed/static.
- Custom domain / repository split later (the layout is designed so `site/` + `tools/` + `docs/` can move to their own repo).
