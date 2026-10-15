// Lesson "why-gradients": one weight, one score. The gradient as a slope, a step against it, step size, and the chain rule
// (the graph lab on scene `why`). Every number comes from data/why-gradients.json through T() (or, inside formulas, T.num and
// the scene's own constants). Notes: docs/lessons/why-gradients.md. Style budget: docs/authoring.md.
import { mountGraphLab } from '../widgets/graph-lab.js';
import { mountBowl } from '../widgets/bowl.js';
import { whyLayout } from './why-gradients.scenes.js';
import { math } from '../core/katex.js';
import { makeModel } from '../widgets/landscape-model.js';

const F = 'files.why-gradients';
// recorded runs, by index: lr 0.05, 0.1, 0.125 (one step), 0.25 (bounces), 0.3 (diverges); the test pins this order
const traj = (i, rest) => `${F}.trajectories.${i}.${rest}`;
const SLOW = 0, EXACT = 2, BOUNCE = 3, DIVERGE = 4;
const eq = (latex) => `<div class="wg-eq">${math(latex)}</div>`;
const sec = (id, title, body) => `<section class="wg" aria-labelledby="wg-${id}"><h2 id="wg-${id}">${title}</h2>${body}</section>`;

const STYLE = `<style>
.wg p{max-width:44rem}
.wg h2{margin-top:2rem}
.wg-eq{overflow-x:auto;margin:.2rem 0}
.wg-eq .katex-display{margin:.4em 0}
.wg-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.wg-code code{background:none;padding:0}
#chain-lab #scene-tabs{display:none}
#chain-lab .lab-top{justify-content:flex-end}
</style>`;

export default {
  id: 'why-gradients',
  title: 'The gradient',
  trace: ['why-gradients.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>The gradient</h1>
    <p class="lead">One weight, one score: what a gradient is, how a step uses it, and where <code>backward()</code> gets it.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'The gradient is a <strong>slope</strong>: how fast the loss changes per unit of weight.',
        'Training steps <strong>against</strong> it: <code>w ← w − lr·g</code>.',
        '<code>lr</code> sets the step: too small is slow, too big overshoots or diverges.',
        '<code>backward()</code> gets the slope by <strong>multiplying local slopes</strong> (the chain rule).',
      ],
    },
    {
      kind: 'text',
      html: (D, T, m = makeModel(D.scenes.why)) => sec('setup', 'The setup', `
    <p>One weight <code>w</code>, one input <code>x</code>, one target <code>t</code>, squared error:</p>
    <pre class="wg-code" tabindex="0" role="group" aria-label="The TorC code"><code>Variable w(Tensor({${T('why.w.value')}.0f}, {1}), true);    // the weight (tracked)
Variable x(Tensor({${T('why.x.value')}.0f}, {1}), false);   // data
Variable t(Tensor({${T('why.t.value')}.0f}, {1}), false);   // data
Variable p = mul(w, x);     // prediction
Variable e = sub(p, t);     // error
Variable L = mul(e, e);     // loss
L.backward();               // w.grad() now holds the slope</code></pre>
    <p>So ${math(`L(w) = (${m.x}w-${m.t})^2 = ${m.x * m.x}(w-${m.wMin})^2`, false)}: a bowl with its bottom at <code>w</code> = ${T(`${F}.setup.w_min`)}.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('slope', '1 · The gradient is a slope', `
    <p>At <code>w</code> = ${T('why.w.value')}, <code>backward()</code> leaves <code>w.grad()</code> = ${T('why.w.grad')}: nudge <code>w</code> up a little and <code>L</code> changes by about ${T('why.w.grad')} times the nudge. Negative means go right, positive means go left, zero means you are at the bottom. It is a rate, not a distance: the bottom is at <code>w</code> = ${T(`${F}.setup.w_min`)}.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '2 · Step against it', `
    ${eq('w \\leftarrow w - \\mathit{lr}\\cdot g')}
    <p>With <code>lr</code> = ${T(traj(SLOW, 'lr'))}, <code>w</code> goes from ${T('why.w.value')} to ${T(traj(SLOW, 'states.1.w'))} and <code>L</code> from ${T('why.L.value')} to ${T(traj(SLOW, 'states.1.L'))}. Steps get shorter near the bottom because the slope does.</p>`),
    },
    {
      kind: 'predict', id: 'big-step', title: 'Predict: a bigger step',
      prompt: (D, T) => `<p>Start at <code>w</code> = ${T('why.w.value')} with <code>lr</code> = ${T(traj(BOUNCE, 'lr'))}. Where is <code>w</code> after one step, and after two?</p>`,
      reveal: (D, T) => `<p>${T(traj(BOUNCE, 'states.1.w'))}, then ${T(traj(BOUNCE, 'states.2.w'))}. Each step jumps to the other side at the same height, so <code>L</code> stays ${T(traj(BOUNCE, 'states.8.L'))} for ever. Try it below.</p>`,
    },
    {
      kind: 'widget', id: 'bowl', className: 'widget', label: 'Interactive: step down the bowl',
      mount: (host, D) => mountBowl(host, D),
    },
    {
      kind: 'text',
      html: (D, T, m = makeModel(D.scenes.why)) => sec('size', '3 · Step size decides everything', `
    <p>In this bowl every step multiplies the distance to the bottom by the same factor:</p>
    ${eq(`w_{k+1}-${m.wMin} = f\\,(w_k-${m.wMin}), \\qquad f = 1-${m.curvature}\\,\\mathit{lr}`)}
    <p>${math('|f|<1', false)} converges, ${math('f=-1', false)} bounces (<code>lr</code> = ${T(traj(BOUNCE, 'lr'))}), ${math('|f|>1', false)} diverges (<code>lr</code> = ${T(traj(DIVERGE, 'lr'))}). <code>lr</code> = ${T(traj(EXACT, 'lr'))} gives ${math('f=0', false)}: the bottom in one step, but only because this bowl is a perfect parabola. Real losses are not, so <code>lr</code> is tuned by watching the loss.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => {
        const s = (i) => T.num(`${F}.nudge.stages.${i}.local_slope`);
        const par = (v) => (v < 0 ? `(${v})` : `${v}`);
        return sec('chain', '4 · backward() multiplies local slopes', `
    <p><code>w</code> reaches <code>L</code> through three operations, each with a local slope. Their product is the gradient:</p>
    ${eq(`\\frac{dL}{dw} = \\frac{dp}{dw}\\cdot\\frac{de}{dp}\\cdot\\frac{dL}{de} = ${par(s(0))}\\cdot ${par(s(1))}\\cdot ${par(s(2))} = ${T.num('why.w.grad')}`)}
    <p><code>e</code> is used twice in <code>e·e</code>, so it gets two contributions of ${T('why.e.value')} and TorC adds them: ${T('why.e.grad')}. Step through it:</p>`);
      },
    },
    {
      kind: 'widget', id: 'chain-lab', className: 'lab', label: 'Interactive lab: forward and backward through the chain',
      mount: (host, D, ctx) => mountGraphLab(host, { scenes: { why: D.scenes.why }, layouts: { why: whyLayout(ctx.makeT()) }, defaultScene: 'why' }),
    },
    {
      kind: 'check', id: 'sign',
      prompt: (D, T) => `<p>At <code>w</code> = ${T(`${F}.probes.5.w`)}, is the slope positive or negative, and which way does a step move <code>w</code>?</p>`,
      reveal: (D, T) => `<p>Positive (${T(`${F}.probes.5.grad`)}), so <code>w − lr·g</code> moves <code>w</code> down, toward ${T(`${F}.setup.w_min`)}.</p>`,
    },
    {
      kind: 'check', id: 'one-step',
      prompt: (D, T) => `<p><code>lr</code> = ${T(traj(EXACT, 'lr'))} reaches the bottom in one step here. Why does that not carry over to a real network?</p>`,
      reveal: () => '<p>It works because this loss is an exact parabola with a known curvature. A real loss bends differently everywhere, so no formula gives the right <code>lr</code>.</p>',
    },
    {
      kind: 'text',
      html: () => `<section class="wg"><p>Next: <a href="../sgd-step/">Using the gradient</a> does this update with TorC's optimizer. The <a href="../backprop-lab/">Backprop lab</a> runs the same chain on bigger graphs.</p></section>`,
    },
  ],
};
