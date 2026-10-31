# Lesson `losses` ("Softmax and cross-entropy") — internal note

Not shipped. Library symbols, and what is observed, expected and suspected. Rewritten 2026-10-30 as a `template: 3` lesson (brief §6
and §7): softmax moved here from `nonlinearity`, MSE is taught in `what-is-learning`, and the MSE shape pitfall and the label errors
went to the appendix lesson `torc-pitfalls` (decision 22). The trace keeps `mse`, `mse_mismatch` and `errors` for that page.

## Library symbols (TorC `38785db`, `main` = v0.1.0 + PR #5 + PR #6)
- `torc::softmax(a, axis)` (`src/autograd.cpp:882`): `Tensor::softmax(axis)` subtracts the row maximum before `exp` (stability), then
  divides by the row sum. The page explains it as "e to the power of each score, divided by the sum", which is the same function.
- `torc::exp`, `torc::sum` (`include/torc/autograd.hpp:212`, `:198`): used only by the trace to record e^score and the sum
  (`exp_rows`), so the page never computes them.
- `nn::CrossEntropyLoss` (`src/nn/losses.cpp:63`): rank-2 logits `(B, C)`, targets `(B)` or `(B, 1)` with integral classes in
  `[0, C)`; loss = mean over rows of `log-sum-exp(row) − row[target]` = `−ln softmax(row)[target]`; backward `(softmax − onehot) / B`.
- `nn::MSELoss`: used by the trace for the squared miss of the shares (`curves.sq_*`); not named on the page.

## Observed (tools/trace/trace_losses.cpp → site/data/losses.json; all checked by site/test/losses.test.mjs)
- `exp_rows` (v3): scores `2, 1, 0.1` → e^ = 7.389, 2.718, 1.105, sum 11.21, shares 0.659, 0.2424, 0.09857 (identical to the
  `softmax` of `ce`). Scores `4, 1, −2` → shares 0.9503, 0.04731, 0.002356 (the predict: the cherry gets a small share above zero).
- `ce`: uniform scores give `ln 3` = 1.0986 for every target; `2, 1, 0.1` with target 0 gives 0.417 (`T.human prob` = 66 %),
  targets 1 and 2 give 1.417 and 2.317; `4, 1, −2` with target 2 gives 6.051, the largest surprise on the page.
- `spread` (v3): `[2, 0, 0]` and `[2, ln 1.9, ln 0.1]`, target 0: the same apple share 0.787 and the same loss 0.2395 (bit-equal in
  float32), while the other shares are 0.1065/0.1065 and 0.2024/0.01065.
- `curves`: scores `[z, 0]`, target 0. `curves.p` (v3) is the share of the right class; the cross-entropy gradient equals `p − 1`
  ("the share minus one"); at z = −6 it is −0.9975, the squared-miss gradient −0.004921.
- `ten` (v3): ten equal scores, target 3: every share 0.1, loss 2.302585 = ln 10. The transfer quiz's feedback names the school-log
  error (log10 of 10 is 1).

## Expected (mathematics)
`softmax(s)_k = e^{s_k} / Σ_j e^{s_j}`; `CE = −ln softmax(s)_t`; `dCE/ds_k = softmax_k − [k = t]` (one sample). For two classes with
scores `[z, 0]` and `p = σ(z)`: CE gradient `p − 1`; squared-miss gradient (MSE over the two shares) `−2(1 − p)² p`, which goes to zero
as `p → 0`. The loss depends only on the share of the right class.

## Suspected discrepancy
None at the pin.

## Page choices
- "Score" throughout; "logit" is named once ("Books call such scores logits"). The glossary entry `logit` names this lesson.
- Palette: the `exp-ln` refresher and the ChatGPT aside (both required). So the bank's analogies (softmax as shares, cross-entropy as
  surprise) are plain prose without the `analogy()` block, with their limits stated as sentences ("They say how sure the model is.
  They do not say that it is right."; "Only the probability of the right class counts").
- The ChatGPT aside gives no vocabulary size, says that the surprise is the loss of pretraining and that later training uses other
  methods, and says it is a concept, not a TorC run. Its sources are in the page footer (decision 18): Radford et al. 2018, section
  3.1 (softmax over the vocabulary, the log-likelihood loss of pretraining) and Ouyang et al. 2022 (later stages). The lead asked
  for a chapter of Jurafsky and Martin's SLP3; web.stanford.edu was not reachable from this session, so a chapter number could not
  be checked, and the two papers are cited instead (not verified online here either; cited from their known content).
- Naming (lead): the output of softmax is a "probability"; "shares of 100 %" appears once, as the analogy with its limit.
  "Squared miss" is the mean squared error of the first lesson applied to the probabilities and the target probabilities.
- The your-turn step uses the second photo (4, 1, −2), so the learner computes e^−2 and the sum (the worked example uses the first).
- No ∂ on the page (∂ is introduced in `backprop`, which now comes later); the gradient is said in words.
- The fruit names map to classes 0, 1, 2 (apple, banana, cherry).
