# Authoring guide (current machinery)

How to add a scene or a new operation to the existing lab. (Multi-lesson scaffolding is planned; see the roadmap.)

## Add a scene
1. **Trace**: in `tools/trace/autograd_trace.cpp` add a `Scene` (leaf values, a builder that registers every node with
   `g.op(name, opName, <real torc op>)`, constants listed in `s.constants`). Keep every tracked `Variable` alive (the program
   uses a `std::deque`), and make sure no ReLU kink is within the finite-difference step of any input.
2. `tools/trace/run.sh` → writes `site/data/autograd_trace.json`. Inspect the numbers; hand-check at least the gradients.
3. **Layout and text**: add an entry to `site/scenes.js` (`pos` per node name, `view`, `goal`, `intro`, hints, `takeaway`).
   Never put numbers in these strings that the trace could change; if a number is unavoidable, add a test that asserts it.
4. `node --test site/test/` and look at every step on desktop and phone width.

## Add an operation
1. Confirm in TorC (`src/autograd.cpp`) what the forward and backward rules really are — including broadcasting and edge cases.
2. `site/engine.js`: add forward and backward rules (raw gradients; reduction to input shape is centralised in `runBackward`).
3. `site/explain.js`: add the rule text (`FORWARD_RULE/WHY`, `BACKWARD_RULE/WHY`), the numeric substitution, and any new
   bookkeeping lines. Every string must parse in KaTeX (the test renders them with `throwOnError`).
4. Add a trace scene using the op so the C++ output exercises it; the test then compares engine vs. real run.

## Conventions
- Multi-letter node names render as `\texttt{…}` so `ab` is never read as a product; single letters are italic math.
- Gradient notation: overline (`ā` = ∂out/∂a), defined where it first appears.
- Explanations are about what **TorC does**, with the math next to it. Quirks of the library belong in `docs/library-audit/`,
  not on lesson pages.
