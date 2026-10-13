# TorC Autograd Visualization — Claude Code Starter Pack

This pack configures a Claude Code workflow for building a Manim-powered GitHub Pages site for TorC while keeping the existing `main` branch untouched.

## Recommended workflow

1. Open a terminal and clone the repository if you have not already:
   ```bash
   git clone https://github.com/IvorLipic/TorC.git
   cd TorC
   ```
2. Ensure your worktree is clean and fetch branches:
   ```bash
   git status
   git fetch origin
   ```
   If you have uncommitted work, commit or stash it before continuing.
3. Create and switch to a dedicated branch:
   ```bash
   git switch -c autograd-visualizations
   ```
   If that branch already exists, use `git switch autograd-visualizations` instead.
4. Copy the `.claude/` directory and `CLAUDE.md` from this pack into the repository root. Do not replace an existing `CLAUDE.md` blindly; merge the instructions if you already have one.
5. Start Claude Code from the repository root:
   ```bash
   claude
   ```
6. Open `ORCHESTRATION_PROMPT.md` and paste its contents into Claude Code as your first task.

## How the roles work

- **Main Claude session:** orchestrator, decision-maker and integrator.
- **autograd-auditor:** read-only investigation of actual implementation and mathematical behavior.
- **manim-specialist:** creates animations only after the behavior contract is established.
- **website-deployer:** owns static site, GitHub Pages workflow and deployment documentation.
- **independent-reviewer:** read-only review of code/math/visualization consistency and deployment readiness.

The four specialists are not meant to all code the same thing concurrently. The auditor goes first. The two implementation specialists can then work in parallel only after the lead agrees on the interface and directory ownership. The reviewer works after a meaningful implementation exists.

## Important safeguards

- The existing TorC implementation is the source of truth for *what the library currently does*.
- Mathematical derivations and finite-difference checks are independent evidence about whether the implementation is correct.
- If implementation and mathematics disagree, document a suspected bug; do not change the core library or animate the expected behavior as though it were implemented.
- Do not edit or commit to `main`.
- Do not merge or push changes without user approval.
- Never enable automatic deployment until the workflow's branch filters and repository Pages settings have been checked.
- Do not expose secrets or add tokens to the repository.
- Do not render every animation at maximum quality during early iterations; first use low-quality previews, then render final assets.

## Why not begin with a large swarm?

This workflow is deliberately small. More agents create coordination overhead and can duplicate work. Add more only if a concrete bottleneck appears.
