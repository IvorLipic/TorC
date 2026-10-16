// Lesson "losses": mean squared error, cross-entropy, the gradient of cross-entropy, and the targets TorC accepts.
// Every number and message comes from data/losses.json (tools/trace/trace_losses.cpp) through T() or the recorded text.
// Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { mountLossExplorer, mountGradientCompare } from '../widgets/loss-widgets.js';

const F = 'files.losses';
const MILD = `${F}.ce.1`, CONF = `${F}.ce.2`, UNI = `${F}.ce.0`;
const eq = (latex) => `<div class="ls2-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="ls2" aria-labelledby="ls2-${id}"><h2 id="ls2-${id}">${title}</h2>${body}</section>`;
const err = (D, id) => { const e = D.files.losses.errors.find((x) => x.id === id); return `<code>${escapeHtml(e.exception)}: ${escapeHtml(e.what)}</code>`; };
const list = (T, path) => [0, 1, 2].map((i) => T(`${path}.${i}`)).join(', ');

const STYLE = `<style>
.ls2 p{max-width:44rem}
.ls2 h2{margin-top:2rem}
.ls2-eq{overflow-x:auto;margin:.2rem 0}
.ls2-eq .katex-display{margin:.4em 0}
.ls2-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.ls2-code code{background:none;padding:0}
</style>`;

export default {
  id: 'losses',
  title: 'Losses',
  trace: ['losses.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Losses</h1>
    <p class="lead">A loss turns the output of a model into one number that says how wrong the model is. Training makes this number smaller. This lesson shows the two losses in TorC.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'Mean squared error averages the squared differences. Use it when the model predicts numbers.',
        'Cross-entropy is for classes. It is small when the model gives the correct class a high probability.',
        'The gradient of cross-entropy is the softmax minus the correct answer. So a very wrong model still gets a strong gradient.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('mse', '1 · Mean squared error', `
    ${eq('L = \\frac{1}{N}\\sum_{i=1}^{N} (y_i - t_i)^2 \\qquad \\frac{\\partial L}{\\partial y_i} = \\frac{2\\,(y_i - t_i)}{N}')}
    <p>Here <code>y</code> is the prediction, <code>t</code> is the target and <code>N</code> is the number of values. With the predictions ${list(T, `${F}.mse.prediction`)} and the targets ${list(T, `${F}.mse.target`)}, the loss is ${T(`${F}.mse.loss`)}. The third prediction is exactly right, so its gradient is ${T(`${F}.mse.grad.2`)}.</p>
    <pre class="ls2-code" tabindex="0" role="group" aria-label="Mean squared error in TorC"><code>nn::MSELoss mse;
Variable L = mse(y, t);    // y and t have the same shape</code></pre>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('ce', '2 · Cross-entropy', `
    <p>A classifier gives one score for each class. These scores are called logits. Softmax turns the logits into probabilities. Cross-entropy is the negative logarithm of the probability of the correct class, averaged over the batch.</p>
    ${eq('p = \\mathrm{softmax}(z) \\qquad L = -\\log p_{t}')}
    <p>For the logits ${list(T, `${MILD}.logits`)}, the first class has probability ${T(`${MILD}.cases.0.softmax.0`)}. If the first class is correct, the loss is ${T(`${MILD}.cases.0.loss`)}. If the third class is correct, the loss is ${T(`${MILD}.cases.2.loss`)}.</p>
    <pre class="ls2-code" tabindex="0" role="group" aria-label="Cross-entropy in TorC"><code>nn::CrossEntropyLoss ce;
Variable L = ce(logits, labels);   // logits (B, C), labels (B)</code></pre>`),
    },
    {
      kind: 'widget', id: 'explorer', className: 'widget', label: 'Interactive: logits, probabilities, loss and gradient for one sample',
      mount: (host, D) => mountLossExplorer(host, D),
    },
    {
      kind: 'text',
      html: () => sec('grad', '3 · The gradient of cross-entropy', `
    ${eq('\\frac{\\partial L}{\\partial z_j} = \\frac{p_j - [\\,j = t\\,]}{B}')}
    <p>Here <code>B</code> is the batch size, and ${math('[\\,j = t\\,]', false)} is one for the correct class and zero for the others. So the gradient raises the logit of the correct class. It lowers each other logit in proportion to its probability.</p>`),
    },
    {
      kind: 'predict', id: 'p-wrong', title: 'Before you try it',
      prompt: () => '<p>A model gives the correct class a probability near zero. Is the gradient for its logit large or small? Compare cross-entropy with the squared error between the probabilities and the correct answer.</p>',
      reveal: (D, T) => `<p>With cross-entropy the gradient is large. At a logit of ${T(`${F}.curves.z.0`)} it is ${T(`${F}.curves.ce_grad.0`)}. With the squared error it is only ${T(`${F}.curves.sq_grad.0`)}. So the squared error learns very slowly when the model is very wrong.</p>`,
    },
    {
      kind: 'widget', id: 'compare', className: 'widget', label: 'Interactive: the gradient of cross-entropy and of squared error, for one logit',
      mount: (host, D) => mountGradientCompare(host, D),
    },
    {
      kind: 'text',
      html: (D) => sec('targets', '4 · Labels in TorC', `
    <p>The labels are class numbers, stored as float values in a tensor of shape <code>(B)</code> or <code>(B, 1)</code>. TorC checks each label. A label of <code>1.5</code> gives this error.</p>
    <p>${err(D, 'not-integer')}</p>
    <p>A label that is not a class of the model gives this error.</p>
    <p>${err(D, 'out-of-range')}</p>`),
    },
    {
      kind: 'check', id: 'c-uniform',
      prompt: () => '<p>A model gives the logits <code>0, 0, 0</code> for three classes. What is the cross-entropy, whichever class is correct?</p>',
      reveal: (D, T) => `<p>It is ${T(`${UNI}.cases.0.loss`)}. Each class has probability ${T(`${UNI}.cases.0.softmax.0`)}, and the loss is the negative logarithm of that probability.</p>`,
    },
    {
      kind: 'check', id: 'c-batch',
      prompt: (D, T) => `<p>A batch has two samples. The loss of the first sample is ${T(`${MILD}.cases.0.loss`)}, and the loss of the second is ${T(`${CONF}.cases.2.loss`)}. What loss does TorC report for the batch?</p>`,
      reveal: (D, T) => `<p>It reports ${T(`${F}.batch.loss`)}, which is the mean of the two losses.</p>`,
    },
    {
      kind: 'text',
      html: () => '<section class="ls2"><p>Go back to the <a href="../../">course map</a>. The next lesson is about the training loop, which repeats the forward pass, the loss and the update many times.</p></section>',
    },
  ],
};
