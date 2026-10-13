# First task for the Claude Code lead agent

You are the technical lead and orchestrator for adding autograd visualizations to https://github.com/IvorLipic/TorC.

Read `CLAUDE.md` first and follow it. Use the custom subagents in `.claude/agents/`. Do not begin by immediately generating the whole website.

## Hard constraints
- All work stays on `autograd-visualizations`; never modify `main`.
- The current TorC code is the source of truth for what is implemented.
- This project is also an audit opportunity: surface possible bugs instead of hiding them.
- Do not modify core library behavior to make the visual story cleaner.
- Do not push, merge, create a PR, or deploy until the user approves.
- Do not claim any test or render passed unless you ran it and inspected its result.

## Stage 0 — establish safety and inspect the repository
1. Run `git status --short --branch` and `git branch --show-current`.
2. If the branch is not `autograd-visualizations`, or the worktree has unrelated modifications, stop and explain what must be resolved. Do not switch branches automatically if doing so could disturb user work.
3. Read the repository's build instructions, existing tests, and structure.
4. Summarize the existing architecture and likely commands for building/testing it.

## Stage 1 — source-code audit (do this first)
Delegate to `autograd-auditor`. Ask it to inspect the real implementation and return a concise, evidence-based report covering:
- tensor and operation representation;
- how graph dependencies are recorded;
- how forward values are computed;
- how backward propagation is triggered and traversed;
- how local derivatives and accumulated gradients are stored/updated;
- behavior when a tensor is reused in multiple operations;
- gradient initialization/reset and graph lifetime, if relevant;
- supported operations and known limitations;
- build/test commands and minimal examples;
- potential correctness defects or ambiguities.

The auditor is read-only. Require file paths, symbols, and minimal reproductions where possible. Do not accept generic autograd assumptions as evidence.

Independently inspect the key findings. Create `docs/autograd-audit/implementation-contract.md` with:
- observed behavior and code references;
- mathematical expectations;
- suspected discrepancies/bugs;
- confirmed limitations;
- unanswered questions.
Do not label a suspected bug as confirmed without adequate evidence.

## Stage 2 — executable ground truth and plan
Before animation production:
1. Select a small set of representative operations that actually exist in TorC.
2. Create or identify minimal C++ examples that print forward values and gradients. Prefer machine-readable output (JSON or simple stable key-value output) where practical.
3. Derive expected gradients independently and add finite-difference checks where practical.
4. Run the examples and tests. Record outputs and tolerances.
5. Write `docs/autograd-audit/visualization-plan.md`, mapping each proposed scene to actual code symbols, a runnable example, equations, outputs, and verification tests.
6. If a discrepancy is discovered, record it and pause that scene or animate the observed behavior with a clear warning. Do not alter the core library without user approval.

## Stage 3 — agree on interfaces before parallel coding
Only after Stage 2, assign separate ownership:
- `manim-specialist`: Manim source and animation assets in a dedicated directory such as `visualizations/manim/`; do not edit the website shell or core C++.
- `website-deployer`: static website and GitHub Actions files; do not edit Manim scene source or core C++.
- Lead: audit contract, shared example-output schema, integration and core tests.
Specify the exact example-output format, generated-media paths, and site base path before both agents start. Avoid concurrent edits to shared files.

Ask both implementation agents to deliver a minimal vertical slice first: one verified operation, one scene, one page, and a successful local build. Inspect that slice before expanding.

## Stage 4 — implementation
Build an attractive, responsive, dark educational site with 3Blue1Brown-inspired mathematical animation style (inspired by the general visual approach; do not copy exact scenes or assets). Use Manim for choreographed animations and HTML/CSS/JS for site navigation and explanations.

Possible topics, only if supported by the code:
- computation graph construction;
- forward pass with numerical values;
- chain rule and local derivatives;
- backward pass and gradient accumulation;
- a linear layer or small MLP if the library supports the needed operations.

Every scene must include:
- a source-code reference;
- a runnable example;
- actual values from the implementation or an explicitly verified derivation;
- clear differentiation between observed behavior and expected math;
- tests or checks that would detect a mismatch.

Use low-quality Manim previews during iteration, then render final assets. Do not generate every scene before verifying the first one.

## Stage 5 — independent review
After a working implementation exists, delegate to `independent-reviewer` in read-only mode. Ask it to compare:
- claims and equations vs implementation contract;
- scene values vs executable output;
- gradients vs analytical and finite-difference checks;
- all links/media and project subpath `/TorC/`;
- workflow branch filters and deployment assumptions;
- accidental changes to the C++ library or main-branch assumptions.

Require findings ranked by severity, with file paths and reproduction steps. Fix substantiated issues, rerun tests, then request another review if meaningful changes were made.

## Stage 6 — final report, no deployment without approval
Return:
- architecture and file map;
- audit findings and suspected bugs;
- scenes created and which are verified;
- exact build/test/render commands and outcomes;
- known limitations;
- Git status and branch;
- GitHub Pages setup steps.
Do not push, merge, or deploy yet. Ask for explicit approval before any remote action.

## Orchestration behavior
- Use subagents for bounded work that benefits from separate context, not for vague “do everything” tasks.
- Keep yourself responsible for integration and truthfulness.
- If a subagent proposes behavior not supported by code, reject it or mark it as a hypothesis.
- Do not parallelize the audit: implementation depends on its findings.
- Do parallelize independent Manim and website tasks after their interface is fixed.
- Prefer a small complete vertical slice over a large unfinished swarm output.
