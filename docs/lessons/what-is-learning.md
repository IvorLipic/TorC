# Lesson notes: `what-is-learning`

Part 1 `whole-game`, the first lesson of the main path (brief §6 Part 0, R4; `loop: 'all'`). Module
`site/js/lessons/what-is-learning.js`, widget `site/js/widgets/fit-game.js` (new) and the digit pad of
`site/js/widgets/digit-widgets.js`, tests `site/test/what-is-learning.test.mjs`. Data: `site/data/whole-game.json`
(`tools/trace/trace_whole_game.cpp`) and `site/data/mnist-mlp.json` (the pad), plus the anchors `train_images` and `mlp_parameters`.

## Shape
Hero with the prerequisites → Questions (loop diagram, all parts lit) → setup "A computer that reads digits" → digit pad (1 task) →
setup "Hours of practice and test scores" (model, prediction, target, training, hidden points, the line `w × hours + b`) →
predict quiz "what will the computer's line do?" → §1 how to score a line (miss, square, mean squared error, worked example on
the line w = 1, b = 3, typical miss = square root of the loss) → "Your turn" numeric quiz → the fitting game (3 tasks) → §2 the
loop in words (data, model, loss, adjust; the five-line loop of `training-loop` as a numbered list: forward, loss, backward, step,
zero_grad as "forgets this direction") → §3 weights, the one knob analogy of the course → §4 new data (the refuted misconception)
→ transfer quiz (weather) → §5 what the course explains, ChatGPT aside (number-free) → explain → The answer.
Palette: analogy (knobs) and ChatGPT. No In TorC panel (the palette allows two), no C++ on the page.

## Observed (TorC at the pin, traced)
- `runs[0]` (`from-zero`): `nn::Linear(1, 1)` from w = b = 0, `nn::MSELoss`, `optim::SGD` lr 0.1 on all ten training points.
  It settles after 200 steps at w = 1.595, b = 1.897, training loss 0.1597 (typical miss 0.40), hidden loss 0.6856 (0.83).
- The first step moves the line from (0, 0) to (2.48, 1.02) and the loss from 29.6 to 1.98. The last 50 steps move it by less
  than 0.01. Near the end the training loss wobbles at the 1e-7 level (float32), so the page never says "falls at every step".
- `probes[]`: MSELoss of fixed lines, with predictions, residuals and (new, derived) `train_square` / `hidden_square`, the
  float32 squares of the residuals. The worked example uses probe 3 (w = 1, b = 3): student 1 (3.9 h, 8) has prediction 6.9,
  miss −1.1, square 1.21; student 6 (3.2 h, 6.7) has square 0.25 (the "your turn" answer); the mean of the ten squares is 0.624.
- Digit pad: of the 20 shown test images, the first 16 are read correctly and the last four are misread (tested).

## Expected mathematics
- MSE of a line = mean of the squared residuals; the JS port `mse()` equals TorC's loss for every probe and every recorded step
  (relative 1e-5). One JS gradient step (dL/dw = 2/n Σ r·x, dL/db = 2/n Σ r) reproduces TorC's next (w, b) (tested).
- The least-squares line (`derived.least_squares`, double) has training loss 0.159748; TorC's run reaches it within 1e-4, so no
  line, including the learner's, can beat the computer on the training points (tested on 2,000 random lines and every probe).
  This backs the correct predict option "at least as well as yours".
- The fitted line is close to the rule y = 1.5x + 2 that made the data, not equal to it (noise, ten points).

## Suspected discrepancy
None found. Note for readers of the data: hidden point 13 (0.8 h, 4.8) lies far above the rule, so the five hidden points are a
rough check. A hand line can score as well as the computer's line on them (probe `w1-b3`: hidden loss 0.686 vs 0.6856). The page
says "five made-up students are few" and claims only that the computer's line misses the hidden points by more than the
training points. Single data set, single run: labelled on the page.

## Not verified
- The ChatGPT aside (same loop, text data, next-word prediction, far more weights) is content beyond TorC, kept number-free;
  the last lesson (`from-digits-to-chatgpt`) is meant to carry the sources.
- Reading time 14 minutes is an estimate. The human think-aloud test (CLAUDE.md) has not been done.
