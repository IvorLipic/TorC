# TorC Autograd Visualization Project Rules

## Repository safety
- Work only on the designated session branch (`claude/vigilant-fermat-qbzq3t`; the user chose it for this cloud session in place of `autograd-visualizations`) unless the user explicitly chooses another branch.
- Never checkout, modify, commit to, or push `main`.
- Before making changes, inspect `git status` and `git branch --show-current`. Stop if the worktree has unrelated changes or the current branch is not the dedicated work branch.
- Do not merge, open a pull request, or push to a remote without explicit user approval.
- Preserve the existing library build, tests, and public APIs unless the user explicitly authorizes a separate core fix.

## Source of truth and correctness
- The existing TorC code is the source of truth for claims about current behavior.
- Trace the real implementation before writing explanations, equations, examples, or animations. Cite relevant repository file paths and symbols in the internal behavior notes.
- Never invent APIs, supported operations, broadcasting behavior, graph lifetime semantics, gradient-reset semantics, or other implementation details.
- Distinguish three things in all reports:
  1. **Observed behavior:** directly traced to the implementation and/or an executable test.
  2. **Expected mathematical behavior:** independently derived from mathematics.
  3. **Suspected discrepancy/bug:** implementation appears inconsistent with expected mathematics or documented intent.
- Do not silently “fix” the core library to make an animation work. Report discrepancies with a minimal reproduction and evidence. Ask the user before changing core behavior.
- Prefer executable C++ examples that produce actual forward values and gradients. Where practical, make the animation pipeline consume their generated outputs instead of hardcoding numerical results independently.
- For gradient correctness, use analytical derivatives and finite differences where appropriate. State numerical tolerances and limitations. Finite differences are evidence, not a replacement for code tracing.
- If behavior cannot be established, label it **unknown / not verified** and do not present it as fact.

## Architecture and ownership
- Keep website/Manim assets isolated from the existing C++ library and build system.
- Before parallel implementation, the orchestrator must define directory ownership and a shared interface for examples, generated values, and media paths.
- Avoid two agents editing the same files concurrently.
- Prefer a static website; GitHub Pages cannot execute the C++ library or render Manim scenes on the server at request time.
- Ensure project-site paths work under `/TorC/`, not just at a domain root.

## Verification
- Inspect existing build/test instructions before running commands.
- Run the existing tests and relevant new tests; do not claim success for commands that were not run.
- Render at least one low-quality Manim preview early and inspect it before producing all final videos.
- Build the website locally and check navigation, media URLs, responsive layout, and the `/TorC/` base path.
- Report exactly what was tested, what passed, what failed, and what remains unverified.

## GitHub Pages
- Use GitHub Actions only after verifying the workflow triggers exclusively from the chosen deployment branch.
- Do not change repository settings through external services without user approval.
- Do not deploy or push without user approval.
