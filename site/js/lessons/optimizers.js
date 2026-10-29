// Lesson "optimizers": why plain gradient descent is slow in a narrow valley, momentum, Adam (its first step) and AdamW (weight
// decay, and the parameters step() skips). Every number comes from data/optimizers.json (tools/trace/trace_optimizers.cpp)
// through T(). Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { math } from '../core/katex.js';
import { mountOptimizerRace } from '../widgets/optimizer-widgets.js';

const F = 'files.optimizers';
const V = `${F}.valley`;
// runs in the trace: sgd, momentum, adam; first_step: SGD, Adam; decay: Adam, AdamW
const RUN = { sgd: `${V}.runs.0`, mom: `${V}.runs.1`, adam: `${V}.runs.2` };
const FIRST = { sgd: `${F}.first_step.0`, adam: `${F}.first_step.1` };
const DECAY = { adam: `${F}.decay.0`, adamw: `${F}.decay.1` };
const eq = (latex) => `<div class="op-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="op" aria-labelledby="op-${id}"><h2 id="op-${id}">${title}</h2>${body}</section>`;

const STYLE = `<style>
.op p{max-width:44rem}
.op h2{margin-top:2rem}
.op-eq{overflow-x:auto;margin:.2rem 0}
.op-eq .katex-display{margin:.4em 0}
.op-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.op-code code{background:none;padding:0}
</style>`;

export default {
  id: 'optimizers',
  title: 'Better optimizers',
  trace: ['optimizers.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Better optimizers</h1>
    <p class="lead">Plain gradient descent is <code>optim::SGD</code> without momentum. It uses the same learning rate for every parameter. This lesson compares it with momentum and Adam in a narrow valley, and shows what AdamW adds.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'If the loss is steep in one direction and flat in another, plain gradient descent is slow.',
        'Momentum keeps the movement in the flat direction.',
        'Adam divides the step of each parameter by the size of its recent gradients.',
        'AdamW also makes each weight that has a gradient a little smaller at each step.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => { const c = D.files.optimizers.valley.curvature; return sec('valley', '1 · A narrow valley', `
    ${eq(`L(w) = \\tfrac{1}{2}\\,w_1^2 + \\tfrac{${c[1]}}{2}\\,w_2^2`)}
    <p>This loss is flat along ${math('w_1', false)} and steep along ${math('w_2', false)}. Plain gradient descent must keep the learning rate below two divided by the steep curvature, here ${T(`${F}.derived.sgd_lr_limit`)}. With a larger rate, the steps across the valley grow, as the lesson Using the gradient showed. With a small rate, the steps along the valley are short.</p>
    <pre class="op-code" tabindex="0" role="group" aria-label="The three optimizers in TorC"><code>optim::SGD sgd(params, 0.019f);                 // plain gradient descent
optim::SGD momentum(params, 0.019f, 0.9f);      // with momentum
optim::Adam adam(params, 0.2f);                 // beta1 0.9, beta2 0.999, eps 1e-8</code></pre>`); },
    },
    {
      kind: 'predict', id: 'p-race', title: 'Before you try it',
      prompt: () => '<p>The three optimizers start at the same point and take the same number of steps. Plain SGD and SGD with momentum use the same learning rate. Which one ends with the lowest loss?</p>',
      reveal: (D, T) => { const n = D.files.optimizers.valley.steps; return `<p>Adam ends lowest, at ${T(`${RUN.adam}.path.${n}.loss`)}. SGD with momentum ends at ${T(`${RUN.mom}.path.${n}.loss`)}. Plain SGD ends at ${T(`${RUN.sgd}.path.${n}.loss`)}, because it is still far along the valley, at ${math('w_1', false)} = ${T(`${RUN.sgd}.path.${n}.w1`)}.</p><p>This race does not compare the optimizers fairly. Adam uses a learning rate about ten times larger than the learning rate of SGD, ${T(`${RUN.adam}.lr`)} against ${T(`${RUN.sgd}.lr`)}. So this race does not show that Adam is the best optimizer for every problem.</p>`; },
    },
    {
      kind: 'widget', id: 'race', className: 'widget', label: 'Interactive: the paths of SGD, SGD with momentum and Adam in a narrow valley',
      mount: (host, D) => mountOptimizerRace(host, D),
    },
    {
      kind: 'text',
      html: () => sec('momentum', '2 · Momentum', `
    ${eq('v \\leftarrow \\mu\\,v + g \\qquad w \\leftarrow w - \\text{lr}\\cdot v')}
    <p>With momentum, each step adds the new gradient ${math('g', false)} to a velocity ${math('v', false)}. Along the valley, the gradients point the same way, so the velocity grows. Across the valley, the gradients change sign, so they cancel each other. The path moves faster along the valley, but it also swings across it.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('adam', '3 · Adam', `
    ${eq('m \\leftarrow \\beta_1 m + (1-\\beta_1)\\,g \\qquad s \\leftarrow \\beta_2 s + (1-\\beta_2)\\,g^2 \\qquad w \\leftarrow w - \\text{lr}\\cdot\\frac{\\hat m}{\\sqrt{\\hat s} + \\varepsilon}')}
    <p>Adam keeps two averages for each parameter. ${math('m', false)} is the average gradient, and ${math('s', false)} is the average squared gradient. ${math('\\hat m', false)} and ${math('\\hat s', false)} correct both averages for their start at zero. ${math('\\beta_1', false)} and ${math('\\beta_2', false)} set how fast the averages forget old gradients, and ${math('\\varepsilon', false)} prevents a division by zero. Adam divides the average gradient by its typical size. So a parameter with small gradients and a parameter with large gradients move about the same distance. In the valley, the first steps move both parameters by the same distance.</p>`),
    },
    {
      kind: 'check', id: 'c-first',
      prompt: (D, T) => `<p>Adam takes its first step with the learning rate ${T(`${FIRST.adam}.lr`)}. One parameter has the gradient ${T(`${FIRST.adam}.grad.0`)}, and another has the gradient ${T(`${FIRST.adam}.grad.1`)}. How far does each parameter move?</p>`,
      reveal: (D, T) => `<p>Both move by about the learning rate, ${T(`${FIRST.adam}.change.0`)} and ${T(`${FIRST.adam}.change.1`)}. Plain SGD with the same learning rate moves them by ${T(`${FIRST.sgd}.change.0`)} and ${T(`${FIRST.sgd}.change.1`)}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('adamw', '4 · AdamW and weight decay', `
    ${eq('w \\leftarrow w - \\text{lr}\\cdot\\frac{\\hat m}{\\sqrt{\\hat s} + \\varepsilon} - \\text{lr}\\cdot\\lambda\\cdot w')}
    <p>AdamW is Adam with weight decay. At each step, it also subtracts a small part of each weight, set by the weight decay ${math('\\lambda', false)}. A weight whose gradient is always zero keeps its value under Adam. Under AdamW, it falls from ${T(`${DECAY.adamw}.steps.0.w2`)} to ${T(`${DECAY.adamw}.steps.50.w2`)} in ${T(`${DECAY.adamw}.steps.50.step`)} steps. A parameter that the loss does not use has no gradient at all. <code>step()</code> skips it, so AdamW does not make it smaller.</p>
    <pre class="op-code" tabindex="0" role="group" aria-label="AdamW in TorC"><code>optim::AdamW adamw(params, 0.1f, /*weight_decay=*/0.1f);</code></pre>`),
    },
    {
      kind: 'check', id: 'c-decay',
      prompt: (D, T) => `<p>AdamW uses the learning rate ${T(`${F}.decay_setup.lr`)} and the weight decay ${T(`${F}.decay_setup.weight_decay`)}. A weight has the value ${T(`${DECAY.adamw}.steps.0.w2`)} and a zero gradient. What is its value after one step?</p>`,
      reveal: (D, T) => `<p>It is ${T(`${DECAY.adamw}.steps.1.w2`)}. Each step multiplies it by ${math('1 - \\text{lr}\\cdot\\lambda', false)}, which is ${T(`${F}.derived.decay_factor`)}.</p>`,
    },
    {
      kind: 'text',
      html: () => '<section class="op"><p>The next lesson, <a href="../generalization/">Generalization</a>, shows why a model can fit its training data and still fail on new data, and how weight decay helps.</p></section>',
    },
  ],
};
