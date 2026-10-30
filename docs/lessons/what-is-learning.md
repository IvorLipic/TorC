# Lesson notes: `what-is-learning`

Part 1 `whole-game`, the first lesson of the main path (brief §6 Part 0, R4; `loop: 'all'`). Module
`site/js/lessons/what-is-learning.js`, widget `site/js/widgets/fit-game.js` and the digit pad of `site/js/widgets/digit-widgets.js`
(with `simple: true`: plain captions, no "label", "TorC" or "probability" on this first page), tests
`site/test/what-is-learning.test.mjs`. Data: `site/data/whole-game.json` (`tools/trace/trace_whole_game.cpp`) and
`site/data/mnist-mlp.json` (the pad), plus the anchors `train_images` and `mlp_parameters`.

## Shape (after the three-gate review)
Hero with the prerequisites (and a pointer to the loop picture) → Questions → setup "A computer that reads digits" → digit pad →
setup "Hours of practice and test scores" (model, training, hidden points, prediction, target, the line `w × hours + b`; the
noise is not mentioned yet, so it does not cue the predict) → predict quiz (four options of similar length; the reveal points at
the game and states no number) → **the game right after the predict** (productive failure: the squares introduce the miss and
the loss) → §1 how to measure the misses (why square, why the root; worked example on w = 1, b = 3) → "Your turn" → §2 the loop
in words (data, model, loss, adjust) with the diagram, the rule behind the data and the computer's line to 2 significant figures
→ §3 parameters (w is a weight, b is a bias; the knobs are the parameters, lead's naming decision) → §4 new data → transfer quiz
(a wiggly curve with zero training loss) → §5 the course promise, ChatGPT aside as a promise only → explain → The answer (Q1 via
the digit model's 60,000 images, Q3 as "keep examples hidden and measure the miss").
Palette: analogy (knobs) and ChatGPT. No In TorC panel, no C++. No "gradient descent" (plain words; the term is introduced in
`why-gradients`), no `zero_grad` wording.

## The game (H1)
"Show the hidden points" is disabled until the computer's run reaches its last step. Showing them fixes the hand line for good
(the handles are `aria-disabled` and ignore keys and drags), so nobody can tune a line on the hidden points. Only a drag that
starts on a handle (40 px touch target) moves the line; other touches scroll the page. The handle elements persist, so focus
survives redraws during Play.

## Observed (TorC at the pin, traced)
- `runs[0]` (`from-zero`): `nn::Linear(1, 1)` from w = b = 0, `nn::MSELoss`, `optim::SGD` lr 0.1 on all ten training points.
  It settles after 200 steps at w = 1.595, b = 1.897, training loss 0.1597 (typical miss 0.40), hidden loss 0.4256 on the thirty
  hidden points (typical miss 0.65).
- The first step moves the line from (0, 0) to (2.48, 1.02); the last 50 steps move it by less than 0.01. Near the end the training
  loss wobbles at the 1e-7 level (float32), so the page never says "falls at every step".
- Worked example, probe 3 (w = 1, b = 3): student 1 (3.9 h, 8) has prediction 6.9, miss −1.1, square 1.21; student 6 (3.2 h, 6.7)
  has square 0.25 (the "your turn" answer).
- Digit pad: of the 20 shown test images, the first 16 are read correctly and the last four are misread (tested).

## Expected mathematics
- MSE of a line = mean of squared residuals; the JS `mse()` equals TorC's loss for every probe and every step. One JS gradient step
  reproduces TorC's next (w, b).
- The run reaches the least-squares loss on the training points (`derived.least_squares`), so no line beats the computer there
  (tested on 2,000 random lines and every probe).
- `derived.best_hidden_line` (least squares on the hidden points): hidden loss 0.4096, typical miss 0.64. The computer's line
  (0.65) is within 0.05 of it; 21 of 4,941 grid lines beat the computer on the hidden points (tested: under 1 %).

## Suspected discrepancy
None. History: with only 5 hidden points (the first version), a hand line could tie the computer on them (probe `w1-b3`: 0.686 vs
0.6856), and the "harder on new data" claim rested on one point. The hidden set is now 30 points from the same rule, noise and
seed; the first 15 points are unchanged. The page says that on a few points a hand line can tie or win by luck (tested on the
first five hidden points). Single data set, single run: labelled ("the students are made up, so these numbers are only an
example").

## Not verified
- The ChatGPT aside is only a promise of the last lesson; no claim beyond TorC is made on this page.
- Reading time 14 minutes is an estimate. The human think-aloud test (CLAUDE.md) has not been done.
