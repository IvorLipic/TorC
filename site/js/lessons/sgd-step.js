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
const eq = (latex) => `<div class="sg-eq">${math(latex, true)}</div>`;
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
      html: () => `${STYLE}<section class="hero"><h1>The SGD update step</h1>
    <p class="lead">Backprop gives a gradient; <code>optimizer.step()</code> spends it.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        '<code>step()</code> does <code>p ← p − lr·p.grad</code> for each parameter. It reads <code>.grad</code>, writes <code>.data</code>, and never computes a gradient.',
        'Too big an <code>lr</code> overshoots, then diverges.',
        'Momentum steps along a running sum of gradients: faster, but it overshoots.',
        '<code>backward()</code> <strong>adds</strong> to <code>.grad</code>, so call <code>zero_grad()</code> every iteration.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('setup', 'The setup', `
    <p>Fit <code>y = w·x + b</code> to three points on the line <code>t</code> = ${T('files.optim.data.true_w')}<code>x</code> + ${T('files.optim.data.true_b')} with <code>nn::MSELoss</code>. Start at <code>w = b = 0</code>, where the loss is ${T('sgd.L.value')}. Each iteration of TorC's training loop ends with:</p>
    <pre class="sg-code" tabindex="0" role="region" aria-label="The training loop in TorC"><code>${lines('loop', 66, 68)}</code></pre>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '1 · step() moves each parameter against its gradient', `
    ${eq('p \\leftarrow p - \\mathit{lr}\\cdot \\overline{p}')}
    <p>Plain SGD is one line of <code>SGD::step()</code>: ${quote('step', 27)} With <code>lr</code> = ${T(`files.optim.runs.${PLAIN}.lr`)}, one step takes <code>W</code> to ${T(step(PLAIN, 1, 'w'))}, <code>b</code> to ${T(step(PLAIN, 1, 'b'))} and the loss to ${T(step(PLAIN, 1, 'loss'))}. <code>x</code> and <code>t</code> have no <code>.grad</code>, so <code>step()</code> never touches them.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('lr', '2 · Too big a learning rate diverges', `
    <p>A step is <code>lr</code> times the gradient, not a fixed distance. At ${T(`files.optim.runs.${PLAIN}.lr`)} the loss falls to ${T(fin(PLAIN, 'loss'))} in ten updates. Past a limit, here about ${T('files.optim.hessian.critical_lr')} (two over the largest curvature), every step overshoots further than the last.</p>`),
    },
    {
      kind: 'predict', id: 'p-lr', title: 'Predict: four times the learning rate',
      prompt: (D, T) => `<p><code>lr</code> = ${T(`files.optim.runs.${BIG}.lr`)} is four times ${T(`files.optim.runs.${PLAIN}.lr`)}. Does it reach the bottom four times faster? Decide, then try it in the widget below.</p>`,
      reveal: (D, T) => `<p>No. It is past the limit: every step overshoots further and the loss grows at every update, to ${T(fin(BIG, 'loss'))} after ten.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('mom', '3 · Momentum remembers past gradients', `
    ${eq('v \\leftarrow \\mu\\,v + \\overline{p} \\qquad p \\leftarrow p - \\mathit{lr}\\cdot v')}
    <p>${quote('step', 24)} The velocity starts at zero, so the first update is the plain one; after that it keeps pushing. With μ = ${T(`files.optim.runs.${MU5}.momentum`)} the loss is ${T(step(MU5, 2, 'loss'))} after two updates (plain: ${T(step(PLAIN, 2, 'loss'))}), then rises to ${T(step(MU5, 3, 'loss'))} as it overshoots. With momentum, a loss that goes up and down is normal.</p>`),
    },
    {
      kind: 'widget', id: 'descent', className: 'widget', label: 'Interactive: SGD on the loss landscape with learning-rate and momentum presets',
      mount: (host, D) => mountDescent(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => sec('zg', '4 · Clear .grad every iteration', `
    <p><code>backward()</code> adds to <code>.grad</code> (${quote('accumulate', 83)}), so three passes at the same <code>W</code> leave three times the gradient. Without <code>zero_grad()</code>, <code>step()</code> reads the sum of every gradient so far and the loop never settles: the loss is ${T(loop(9, 'loss'))} at iteration ten, against ${T(step(PLAIN, 9, 'loss'))} with it. The opposite mistake is silent too: ${quote('step', 20)} skips a parameter that has no gradient.</p>`),
    },
    {
      kind: 'widget', id: 'loop', className: 'widget', label: 'A training loop with and without zero_grad',
      mount: (host, D) => mountLoopTimeline(host, D),
    },
    {
      kind: 'check', id: 'c-zero',
      prompt: (D, T) => `<p>Three forward-and-backward passes at <code>W = b = 0</code> with no <code>zero_grad()</code>, then one <code>step()</code> with <code>lr</code> = ${T(`files.optim.runs.${PLAIN}.lr`)}. Where does <code>W</code> end up?</p>`,
      reveal: (D, T) => {
        const lr = T.num(`files.optim.runs.${PLAIN}.lr`), g = T.num(acc(2, 'gw'));
        return `<p><code>W.grad</code> is ${T(acc(2, 'gw'))}, three times one pass, so ${math(`W = 0 - ${lr}\\cdot(${fmt(g)}) = ${fmt(-lr * g)}`, false)}: three steps' worth in one.</p>`;
      },
    },
    {
      kind: 'check', id: 'c-mom',
      prompt: (D, T) => `<p>With μ = ${T(`files.optim.runs.${MU9}.momentum`)}, does the second update move <code>W</code> more or less than plain SGD's second update (${T(step(PLAIN, 1, 'w'))} → ${T(step(PLAIN, 2, 'w'))})?</p>`,
      reveal: (D, T) => `<p>More: to ${T(step(MU9, 2, 'w'))}. The velocity still carries the first gradient on top of the new one.</p>`,
    },
    {
      kind: 'text',
      html: () => '<section class="sg"><p>Next: <a href="../linear-layer/">Layers</a>, the parameters <code>step()</code> usually updates.</p></section>',
    },
  ],
};
