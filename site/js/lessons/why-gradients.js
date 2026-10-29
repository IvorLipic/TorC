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
const eq = (latex) => `<div class="wg-eq" tabindex="0" role="group" aria-label="Formula">${math(latex)}</div>`;
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
    <p class="lead">This lesson uses a model with one weight. It shows what a gradient is, how training uses it, and how <code>backward()</code> computes it.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'The gradient tells you how fast the loss changes when you change a weight.',
        'A training step moves the weight a small distance against the gradient.',
        'The learning rate sets the size of the step. If the step is too large, the loss can grow.',
        '<code>backward()</code> computes the gradient with the chain rule.',
      ],
    },
    {
      kind: 'text',
      html: (D, T, m = makeModel(D.scenes.why)) => sec('setup', 'The model', `
    <p>The model has one weight <code>w</code>. It multiplies an input <code>x</code> by <code>w</code> to make a prediction <code>p</code>. The error <code>e</code> is the prediction minus the target <code>t</code>. The loss <code>L</code> is the error squared. This is the model in TorC.</p>
    <pre class="wg-code" tabindex="0" role="group" aria-label="The TorC code"><code>Variable w(Tensor({${T('why.w.value')}.0f}, {1}), true);    // the weight (tracked)
Variable x(Tensor({${T('why.x.value')}.0f}, {1}), false);   // data
Variable t(Tensor({${T('why.t.value')}.0f}, {1}), false);   // data
Variable p = mul(w, x);     // prediction
Variable e = sub(p, t);     // error
Variable L = mul(e, e);     // loss
L.backward();               // w.grad() now holds the gradient</code></pre>
    <p>With these numbers the loss is ${math(`L(w) = (${m.x}w-${m.t})^2 = ${m.x * m.x}(w-${m.wMin})^2`, false)}. The graph of <code>L</code> against <code>w</code> is a bowl. Its lowest point is at <code>w</code> = ${T(`${F}.setup.w_min`)}.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('slope', '1 · What the gradient tells you', `
    <p>After <code>L.backward()</code>, <code>w.grad()</code> holds the gradient of <code>L</code> with respect to <code>w</code>. At <code>w</code> = ${T('why.w.value')} the gradient is ${T('why.w.grad')}. So if you increase <code>w</code> by a small amount, <code>L</code> changes by about ${T('why.w.grad')} times that amount.</p>
    <p>A negative gradient tells you to increase <code>w</code>. A positive gradient tells you to decrease it. At the lowest point the gradient is zero. The gradient does not tell you how far away the lowest point is.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '2 · A step against the gradient', `
    ${eq('w \\leftarrow w - \\mathit{lr}\\cdot g')}
    <p>Here <code>g</code> is the gradient and <code>lr</code> is the learning rate, a small number that you choose. With <code>lr</code> = ${T(traj(SLOW, 'lr'))}, one step moves <code>w</code> from ${T('why.w.value')} to ${T(traj(SLOW, 'states.1.w'))}. The loss falls from ${T('why.L.value')} to ${T(traj(SLOW, 'states.1.L'))}. Each step is shorter than the one before, because the gradient gets smaller near the lowest point.</p>`),
    },
    {
      kind: 'predict', id: 'big-step', title: 'Before you try it',
      prompt: (D, T) => `<p>Start at <code>w</code> = ${T('why.w.value')} and set <code>lr</code> to ${T(traj(BOUNCE, 'lr'))}. Where is <code>w</code> after one step? Where is it after two steps?</p>`,
      reveal: (D, T) => `<p>After one step <code>w</code> is ${T(traj(BOUNCE, 'states.1.w'))}. After two steps it is ${T(traj(BOUNCE, 'states.2.w'))} again. Each step jumps to the other side of the bowl at the same height, so the loss stays at ${T(traj(BOUNCE, 'states.8.L'))}. Use the widget below to check this.</p>`,
    },
    {
      kind: 'widget', id: 'bowl', className: 'widget', label: 'Interactive: step down the bowl',
      mount: (host, D) => mountBowl(host, D),
    },
    {
      kind: 'text',
      html: (D, T, m = makeModel(D.scenes.why)) => sec('size', '3 · How to choose the learning rate', `
    <p>For this loss, each step multiplies the distance between <code>w</code> and the lowest point by the same number <code>f</code>.</p>
    ${eq(`w_{k+1}-${m.wMin} = f\\,(w_k-${m.wMin}), \\qquad f = 1-${m.curvature}\\,\\mathit{lr}`)}
    <p>If ${math('|f|<1', false)}, the distance gets smaller at each step. If ${math('f=-1', false)}, <code>w</code> jumps between two points forever, as with <code>lr</code> = ${T(traj(BOUNCE, 'lr'))}. If ${math('|f|>1', false)}, each jump is larger than the last and the loss grows, as with <code>lr</code> = ${T(traj(DIVERGE, 'lr'))}. With <code>lr</code> = ${T(traj(EXACT, 'lr'))}, <code>f</code> is zero and one step reaches the lowest point. This works only because this loss is an exact parabola. For a real network, you try several learning rates and watch the loss.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => {
        const s = (i) => T.num(`${F}.nudge.stages.${i}.local_slope`);
        const par = (v) => (v < 0 ? `(${v})` : `${v}`);
        return sec('chain', '4 · How backward() computes the gradient', `
    <p><code>w</code> changes <code>L</code> through three operations. Each operation has a local derivative, which tells how much its output changes when its input changes. The chain rule says that the gradient is the product of the three local derivatives.</p>
    ${eq(`\\frac{dL}{dw} = \\frac{dp}{dw}\\cdot\\frac{de}{dp}\\cdot\\frac{dL}{de} = ${par(s(0))}\\cdot ${par(s(1))}\\cdot ${par(s(2))} = ${T.num('why.w.grad')}`)}
    <p>The loss <code>L = e·e</code> is the square of <code>e</code>, so its gradient with respect to <code>e</code> is <code>2e</code>, which is ${T('why.e.grad')}. TorC gets the same answer in a different way. It treats each use of <code>e</code> separately, which gives ${T('why.e.value')} for each use, and it adds the two results. The <a href="../backprop-lab/">Backprop lab</a> shows why adding is right.</p>`);
      },
    },
    {
      kind: 'widget', id: 'chain-lab', className: 'lab', label: 'Interactive lab: forward and backward through the chain',
      mount: (host, D, ctx) => mountGraphLab(host, { scenes: { why: D.scenes.why }, layouts: { why: whyLayout(ctx.makeT()) }, defaultScene: 'why' }),
    },
    {
      kind: 'check', id: 'sign',
      prompt: (D, T) => `<p>At <code>w</code> = ${T(`${F}.probes.5.w`)}, is the gradient positive or negative? In which direction does a step move <code>w</code>?</p>`,
      reveal: (D, T) => `<p>The gradient is positive (${T(`${F}.probes.5.grad`)}). The step subtracts the learning rate times the gradient, so <code>w</code> decreases toward ${T(`${F}.setup.w_min`)}.</p>`,
    },
    {
      kind: 'check', id: 'one-step',
      prompt: (D, T) => `<p>With <code>lr</code> = ${T(traj(EXACT, 'lr'))}, one step reaches the lowest point of this loss. Why does this not work for a real network?</p>`,
      reveal: () => '<p>It works because this loss is an exact parabola, and we know its curvature. The loss of a real network has a different curvature at every point. So no formula gives the correct learning rate.</p>',
    },
    {
      kind: 'text',
      html: () => `<section class="wg"><p>The next lesson, <a href="../sgd-step/">Gradient descent with many weights</a>, does the same step with TorC's optimizer. The <a href="../backprop-lab/">Backprop lab</a> shows the chain rule on larger graphs.</p></section>`,
    },
  ],
};
