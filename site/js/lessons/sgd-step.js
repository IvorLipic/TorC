// Lesson "sgd-step": one optimizer.step(), the learning rate, momentum and zero_grad. Numbers come from data/optim.json (a real
// TorC run) through T(); widgets: descent.js (landscape, lr and momentum presets) and loop-timeline.js (with/without zero_grad),
// both on the optimizer port in ../optim.js. Library lines are quoted from SOURCES. Style budget: docs/authoring.md.
import { mountDescent } from '../widgets/descent.js';
import { mountLoopTimeline } from '../widgets/loop-timeline.js';
import { fmt } from '../widgets/sgd-ui.js';
import { SOURCES } from '../widgets/sgd-code.js';
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';

// one real line of the library, copied from SOURCES (so it can never drift from the quoted source)
const quote = (key, n) => { const s = SOURCES[key]; return `<code class="quote" data-src="${key}:${n}">${escapeHtml(s.lines[n - s.from].trim())}</code>`; };
const lines = (key, from, to) => { const s = SOURCES[key]; return s.lines.slice(from - s.from, to - s.from + 1).map((l) => escapeHtml(l.trim())).join('\n'); };
const eq = (latex) => `<div class="sg-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="sg" aria-labelledby="sg-${id}"><h2 id="sg-${id}">${title}</h2>${body}</section>`;
// recorded runs, by index: 0 lr 0.05 · 1 lr 0.1 · 2 lr 0.15 · 3 lr 0.2 · 4 lr 0.05 μ 0.5 · 5 lr 0.05 μ 0.9 (the test pins this order)
const PLAIN = 0, BIG = 3, MU5 = 4, MU9 = 5;
const step = (run, i, key) => `files.optim.runs.${run}.steps.${i}.${key}`;
const fin = (run, key) => `files.optim.runs.${run}.final.${key}`;
const loop = (i, key) => `files.optim.no_zero_grad.training_loop.iterations.${i}.${key}`;
const acc = (i, key) => `files.optim.no_zero_grad.accumulate_only.${i}.${key}`;

const STYLE = `<style>
.sg p{max-width:44rem}
.sg h2{margin-top:2rem}
.sg-eq{overflow-x:auto;margin:.2rem 0}
.sg-eq .katex-display{margin:.4em 0}
.sg-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.sg-code code{background:none;padding:0}
</style>`;

export default {
  id: 'sgd-step',
  title: 'Using the gradient',
  trace: ['optim.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Using the gradient</h1>
    <p class="lead">The previous lesson changed one weight by hand. This lesson uses TorC's optimizer <code>optim::SGD</code> to train a model with two parameters.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        '<code>step()</code> subtracts the learning rate times the gradient from each parameter. It does not compute gradients.',
        'If the learning rate is too large, the loss grows.',
        'Momentum adds part of the previous step to each new step.',
        '<code>backward()</code> adds to the stored gradient. Call <code>zero_grad()</code> in each iteration to clear it.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('setup', 'The model', `
    <p>The model is a line, <code>y = W·x + b</code>. It must fit three points that lie on the line <code>t</code> = ${T('files.optim.data.true_w')}<code>x</code> + ${T('files.optim.data.true_b')}. The loss is the mean squared error, <code>nn::MSELoss</code>. Both parameters start at zero, where the loss is ${T('sgd.L.value')}. Each iteration of the training loop in TorC's examples ends with these three calls.</p>
    <pre class="sg-code" tabindex="0" role="group" aria-label="The training loop in TorC"><code>${lines('loop', 66, 68)}</code></pre>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '1 · What step() does', `
    ${eq('p \\leftarrow p - \\mathit{lr}\\cdot \\overline{p}')}
    <p>Here <code>p</code> is a parameter and ${math('\\overline{p}', false)} is its gradient, which <code>backward()</code> stored in <code>p.grad</code>. In TorC this is one line of <code>SGD::step()</code>.</p>
    <p>${quote('step', 27)}</p>
    <p>With <code>lr</code> = ${T(`files.optim.runs.${PLAIN}.lr`)}, one step moves <code>W</code> from zero to ${T(step(PLAIN, 1, 'w'))} and <code>b</code> to ${T(step(PLAIN, 1, 'b'))}. The loss falls to ${T(step(PLAIN, 1, 'loss'))}. The data <code>x</code> and <code>t</code> do not change, because the optimizer received only <code>W</code> and <code>b</code>.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('lr', '2 · A learning rate that is too large', `
    <p>Each step is the learning rate times the gradient. With <code>lr</code> = ${T(`files.optim.runs.${PLAIN}.lr`)}, the loss falls at every one of the ten steps. For this loss, a learning rate above about ${T('files.optim.hessian.critical_lr')} is too large. Above that limit, each step moves the parameters past the lowest point, and each time by more than before.</p>`),
    },
    {
      kind: 'predict', id: 'p-lr', title: 'Before you try it',
      prompt: (D, T) => `<p>A learning rate of ${T(`files.optim.runs.${BIG}.lr`)} is four times ${T(`files.optim.runs.${PLAIN}.lr`)}. Does the loss reach its lowest point four times faster? Decide, then try it in the widget below.</p>`,
      reveal: (D, T) => `<p>No. This learning rate is above the limit. Each step moves past the lowest point by more than the last, and the loss grows at every step. After ten steps it is ${T(fin(BIG, 'loss'))}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('mom', '3 · Momentum', `
    ${eq('v \\leftarrow \\mu\\,v + \\overline{p} \\qquad p \\leftarrow p - \\mathit{lr}\\cdot v')}
    <p>With momentum, TorC keeps a velocity <code>v</code> for each parameter. Each step multiplies the old velocity by μ, a number between zero and one, and adds the new gradient.</p>
    <p>${quote('step', 24)}</p>
    <p>The velocity starts at zero, so the first step is the same as without momentum. After that, earlier gradients continue to push the parameters. With μ = ${T(`files.optim.runs.${MU5}.momentum`)}, the loss is ${T(step(MU5, 2, 'loss'))} after two steps, and ${T(step(PLAIN, 2, 'loss'))} without momentum. At step three it rises to ${T(step(MU5, 3, 'loss'))}, because the parameters move past the lowest point. With momentum, the loss often goes up and down before it settles.</p>`),
    },
    {
      kind: 'widget', id: 'descent', className: 'widget', label: 'Interactive: SGD on the loss landscape with learning-rate and momentum presets',
      mount: (host, D) => mountDescent(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => sec('zg', '4 · Clear the gradient in each iteration', `
    <p><code>backward()</code> adds the new gradient to the value that is already in <code>.grad</code>.</p>
    <p>${quote('accumulate', 83)}</p>
    <p>If you call <code>backward()</code> three times at the same <code>W</code>, <code>.grad</code> holds three times the gradient. Without <code>zero_grad()</code>, each <code>step()</code> uses the sum of all earlier gradients, and the loss does not settle. In iteration ten the loss is ${T(loop(9, 'loss'))}. With <code>zero_grad()</code> it is ${T(step(PLAIN, 9, 'loss'))}.</p>
    <p>A second mistake is also silent. <code>step()</code> skips each parameter that has no gradient, so a call to <code>step()</code> before <code>backward()</code> changes nothing.</p>
    <p>${quote('step', 20)}</p>`),
    },
    {
      kind: 'widget', id: 'loop', className: 'widget', label: 'A training loop with and without zero_grad',
      mount: (host, D) => mountLoopTimeline(host, D),
    },
    {
      kind: 'check', id: 'c-zero',
      prompt: (D, T) => `<p>You call <code>backward()</code> three times at <code>W = b = 0</code> without <code>zero_grad()</code>. Then you call <code>step()</code> once with <code>lr</code> = ${T(`files.optim.runs.${PLAIN}.lr`)}. What is <code>W</code> after the step?</p>`,
      reveal: (D, T) => {
        const lr = T.num(`files.optim.runs.${PLAIN}.lr`), g = T.num(acc(2, 'gw'));
        return `<p><code>W.grad</code> is ${T(acc(2, 'gw'))}, which is three times the gradient of one pass. So ${math(`W = 0 - ${lr}\\cdot(${fmt(g)}) = ${fmt(-lr * g)}`, false)}. One step moves <code>W</code> as far as three normal steps.</p>`;
      },
    },
    {
      kind: 'check', id: 'c-mom',
      prompt: (D, T) => `<p>With μ = ${T(`files.optim.runs.${MU9}.momentum`)}, does the second step move <code>W</code> more or less than the second step without momentum? Without momentum, <code>W</code> moves from ${T(step(PLAIN, 1, 'w'))} to ${T(step(PLAIN, 2, 'w'))}.</p>`,
      reveal: (D, T) => `<p>More. <code>W</code> moves to ${T(step(MU9, 2, 'w'))}, because the velocity still contains the first gradient.</p>`,
    },
    {
      kind: 'text',
      html: () => '<section class="sg"><p>The next lesson, <a href="../linear-layer/">Layers</a>, shows how one layer of a neural network uses many weights at once.</p></section>',
    },
  ],
};
