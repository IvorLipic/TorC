// Lesson "why-gradients" (Part 2 `gradient`, curriculum v3, template 3): the gradient found by nudging, one step against it, the
// steep-bowl refutation (brief C3) and the bowl with fog mode. It implements the gold-standard core of brief Appendix B; the
// adaptations are listed in docs/lessons/why-gradients.md. Every number comes from data/why-gradients.json through a role-tagged
// T() (docs/authoring.md "Number roles"); formulas take their constants from the trace with T.num. The chain rule and the graph
// lab left this lesson for `backprop` (why-gradients.scenes.js keeps the graph-lab layout of scene `why` for it).
import { math } from '../core/katex.js';
import { worked, realLine, analogy, inTorc, term, sym } from '../core/blocks.js';
import { mountNudgeLens } from '../widgets/nudge-lens.js';
import { mountTwoBowls } from '../widgets/two-bowls.js';
import { mountBowl } from '../widgets/bowl.js';

const F = 'files.why-gradients';
// recorded runs, by index (the test pins this order): lr 0.05, 0.1, 0.125, 0.25 (bounces), 0.3 (diverges)
const traj = (i, rest) => `${F}.trajectories.${i}.${rest}`;
const SLOW = 0, FAST = 1, BOUNCE = 3, DIVERGE = 4;
// the steep bowl (four notebooks): its run with lr 0.05 is steep.trajectories.0
const steep = (rest) => `${F}.steep.${rest}`;
const sec = (id, title, body) => `<section class="wg" aria-labelledby="wg-${id}"><h2 id="wg-${id}">${title}</h2>${body}</section>`;
const eq = (latex, label) => `<div class="wg-eq" tabindex="0" role="group" aria-label="${label}">${math(latex)}</div>`;
const lr = '<span class="wg-lr">lr</span>';

const STYLE = `<style>
.wg p{max-width:44rem}
.wg h2{margin-top:2rem}
.wg-eq{overflow-x:auto;margin:.2rem 0;max-width:44rem}
.wg-eq .katex-display{margin:.4em 0}
.wg-lr{font-style:normal}
.wg .in-torc pre{overflow-x:auto;max-width:100%;background:var(--panel2);border-radius:6px;padding:.4rem .7rem}
</style>`;

export default {
  id: 'why-gradients',
  title: 'The gradient',
  trace: ['why-gradients.json'],
  template: 3,
  loop: 'adjust',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>The gradient</h1>
    <p class="lead">In this lesson, a model with one weight learns the price of a notebook.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'A model guesses a price, and the guess is wrong. How does the computer know which way to change it?',
        'How big a change should it make at each step?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('price', 'The price of a notebook', `
    <p>Two notebooks cost ${T(`${F}.setup.t`, 'setup')} euros. The model has one weight, <code>w</code>, which is its guess of the price of one notebook.</p>
    <p>For two notebooks the model predicts ${math(`${T.num(`${F}.setup.x`)}w`, false)} euros. The ${term('loss')} measures how wrong the prediction is. Here it is the square of the miss, ${math(`(${T.num(`${F}.setup.x`)}w - ${T.num(`${F}.setup.t`)})^2`, false)}.</p>
    <p>The model starts with the guess <code>w</code> = ${T(`${F}.probes.0.w`, 'setup')}, so it predicts ${T(`${F}.probes.0.p`, 'setup')} euros. It misses by ${T(`${F}.nudge.miss_before`, 'setup')} euros, and the loss is ${T(`${F}.probes.0.L`, 'setup')}.</p>
    <p>Here ${T(`${F}.setup.t`, 'setup')} divided by ${T(`${F}.setup.x`, 'setup')} gives the price at once. A network for handwritten digits has ${T.anchor('mlp_parameters')} weights and no such shortcut. So we practise the general method on this small problem.</p>`),
    },
    {
      kind: 'quiz', id: 'nudge', purpose: 'predict', type: 'numeric',
      answer: `${F}.nudge.stages.2.after`, tolerance: 0.01,
      prompt: (D, T) => `<p>Raise <code>w</code> from ${T(`${F}.nudge.w0`, 'setup')} to ${T(`${F}.nudge.delta`, 'setup')}. The prediction becomes ${T(`${F}.nudge.stages.0.after`, 'setup')} euros, and the miss becomes ${T(`${F}.nudge.miss_after`, 'setup')} euros. What is the new loss?</p>`,
      reveal: (D, T) => `<p>The new loss is ${T(`${F}.nudge.miss_after`, 'reveal')}² = ${T(`${F}.nudge.stages.2.after`, 'reveal')}. So the loss changed by about ${T(`${F}.nudge.actual_dL`, { role: 'reveal', sig: 2 })}.</p>
    <p>The change divided by the nudge is about ${T(`${F}.nudge.ratio`, { role: 'reveal', sig: 2 })}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('gradient', '1 · The gradient', `
    <p>With a nudge ten times smaller, the loss changes about ten times less. The change divided by the nudge stays near ${T(`${F}.probes.0.grad`, 'setup')}.</p>
    <p>This ratio is the ${term('gradient')}. It tells how much the loss changes for each unit that you move <code>w</code>. So the gradient is the slope of the loss curve at the current point.</p>
    <p>We write the gradient as ${sym('dl-dw', math('\\frac{dL}{dw}', false))}. Read it as “how much the loss changes per unit change of <code>w</code>”.</p>
    <p>A negative gradient tells you that a larger <code>w</code> gives a smaller loss. A positive gradient tells you the opposite.</p>
    <p>The widget below shows why the nudge must be small.</p>`),
    },
    {
      kind: 'widget', id: 'nudge-lens', className: 'widget', label: 'Interactive: the nudge lens',
      mount: (host, D) => mountNudgeLens(host, D),
      tasks: (D, T) => [{
        do: `Choose the nudge ${T(`${F}.secants.0.h`, 'setup')}. Then choose smaller nudges, one after the other, down to ${T(`${F}.secants.4.h`, 'setup')}.`,
        see: 'The view zooms in each time. The loss curve and the straight line come closer, until they look like one line.',
      }],
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '2 · One step', `
    <p>Training changes <code>w</code> by the gradient times a small number, in the opposite direction. That small number is the ${term('learning-rate')}, written ${lr}.</p>
    ${worked(`<p>new <code>w</code> = <code>w</code> − ${lr} × gradient = ${T(`${F}.probes.0.w`, 'worked')} − ${T(traj(SLOW, 'lr'), 'worked')} × (${T(`${F}.probes.0.grad`, 'worked')}) = <strong>${T(traj(SLOW, 'states.0.w_next'), 'worked')}</strong></p>`)}
    <p>A minus sign before a negative number adds. So <code>w</code> goes up, toward the true price.</p>
    <p>Programs often write this step with an arrow:</p>
    ${eq('w \\leftarrow w - \\mathrm{lr}\\times\\text{gradient}', 'The step with an arrow')}
    <p>Read the ${sym('arrow', 'arrow')} as “replace <code>w</code> by”.</p>
    ${realLine('L.backward();', 'In TorC the loss is a variable named <code>L</code>. This one line finds the gradient, without any nudges, and stores it in <code>w.grad()</code>.')}`),
    },
    {
      kind: 'quiz', id: 'your-step', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: traj(SLOW, 'states.1.w_next'), tolerance: 0.01,
      prompt: (D, T) => `<p>After the first step, <code>w</code> is ${T(traj(SLOW, 'states.1.w'), 'setup')}, and the gradient there is ${T(traj(SLOW, 'states.1.grad'), 'setup')}. The learning rate is still ${T(traj(SLOW, 'lr'), 'setup')}. Where does the second step land?</p>`,
      reveal: (D, T) => `${worked(`<p>new <code>w</code> = ${T(traj(SLOW, 'states.1.w'), 'worked')} − ${T(traj(SLOW, 'lr'), 'worked')} × (${T(traj(SLOW, 'states.1.grad'), 'worked')}) = <strong>${T(traj(SLOW, 'states.1.w_next'), 'worked')}</strong></p>`)}
    <p>The first step moved <code>w</code> by ${T(traj(SLOW, 'states.0.step'), 'reveal')}. This step moves it by only ${T(traj(SLOW, 'states.1.step'), { role: 'reveal', sig: 2 })}.</p>
    <p>The steps get shorter because the loss curve is flatter near the bottom. Each step keeps ${T(traj(SLOW, 'kept_percent'), { role: 'reveal', sig: 2 })}% of the distance to the bottom.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('steep', '3 · A tempting idea that is wrong', `
    <p>Many people read a large gradient as “the answer is far away”.</p>
    <p>Suppose that four notebooks cost ${T(steep('setup.t'), 'setup')} euros. The true price is still ${T(steep('setup.w_min'), 'setup')}, and the model starts again at <code>w</code> = ${T(steep('probes.0.w'), 'setup')}. So the distance to the answer is the same.</p>
    <p>The gradient at the start is four times larger. It is ${T(steep('probes.0.grad'), 'contrast')} instead of ${T(`${F}.probes.0.grad`, 'contrast')}.</p>
    <p>Two things make it larger. The miss is twice as large as before. Each nudge of <code>w</code> also changes the prediction twice as much.</p>
    <p>So the gradient tells you how steep the loss curve is where you stand. It does not tell you how far away the bottom is.</p>`),
    },
    {
      kind: 'widget', id: 'two-bowls', className: 'widget', label: 'Interactive: two bowls with the same bottom',
      mount: (host, D) => mountTwoBowls(host, D),
      tasks: (D, T) => [{
        do: `Choose the weight ${T(steep('compare.1.w'), 'setup')}, then the weight ${T(steep('compare.3.w'), 'setup')}.`,
        see: 'At each weight, both bowls have the same distance to the bottom. The gradient of the four-notebook bowl is always four times as large.',
      }],
    },
    {
      kind: 'text',
      html: () => sec('fog', '4 · Steps in the fog', `
    <p>A real network has thousands of weights, and nobody can draw its loss curve. So training measures the gradient where it is, takes one step, and measures again.</p>
    ${analogy('You stand on a hillside in thick fog. You feel which way the ground goes down under your feet, and how steep it is. You cannot see the valley.',
    'A hiker remembers the path. A training step does not.')}
    <p>In the widget below, a fog hides the loss curve. You see only <code>w</code>, the loss and the gradient at your point.</p>`),
    },
    {
      kind: 'widget', id: 'bowl', className: 'widget', label: 'Interactive: step down the bowl in the fog',
      mount: (host, D) => mountBowl(host, D),
      tasks: (D, T) => [
        { do: `Turn the fog on. Press Step with the learning rate ${T(traj(SLOW, 'lr'), 'setup')} until the gradient is near zero. Then lift the fog.`,
          see: 'The steps get shorter, and <code>w</code> comes close to the bottom. The lifted fog shows your path.' },
        { do: `Choose ${T(traj(BOUNCE, 'lr'), 'setup')}. Before you press Step, decide where <code>w</code> will land.`,
          see: '<code>w</code> lands as far past the bottom as it started before it. Then it jumps back, again and again.' },
        { do: `Choose ${T(traj(DIVERGE, 'lr'), 'setup')} and press Step a few times.`,
          see: 'Each jump lands farther from the bottom, and the loss grows.' },
      ],
    },
    {
      kind: 'quiz', id: 'best-lr', purpose: 'check', type: 'choice',
      prompt: () => '<p>You start at the same point and take eight steps. Which learning rate brings <code>w</code> closest to the bottom?</p>',
      options: [
        { html: (D, T) => T(traj(SLOW, 'lr'), 'setup'), feedback: 'A smaller learning rate is safer, but it is not always better. Its steps cover less of the distance.' },
        { html: (D, T) => T(traj(FAST, 'lr'), 'setup'), correct: true, feedback: 'Yes. Its steps are long, but they do not overshoot.' },
        { html: (D, T) => T(traj(BOUNCE, 'lr'), 'setup'), feedback: '<code>w</code> jumps across the bottom and back. It never comes closer.' },
        { html: (D, T) => T(traj(DIVERGE, 'lr'), 'setup'), feedback: 'Each jump is longer than the one before. So <code>w</code> moves away from the bottom.' },
      ],
      reveal: (D, T) => `<p>With ${T(traj(FAST, 'lr'), 'reveal')}, each step keeps only ${T.human(traj(FAST, 'kept'), 'pct')} of the distance to the bottom. With ${T(traj(SLOW, 'lr'), 'reveal')}, each step keeps ${T.human(traj(SLOW, 'kept'), 'pct')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('lr', '5 · How to choose the learning rate', `
    <p>On this bowl, each step keeps the same share of the distance to the bottom. A larger learning rate keeps a smaller share, but only up to a point.</p>
    <p>At ${lr} = ${T(traj(BOUNCE, 'lr'), 'setup')} the share is ${T.human(traj(BOUNCE, 'kept'), 'pct')}, so <code>w</code> jumps across the bottom and back forever.</p>
    <p>At ${lr} = ${T(traj(DIVERGE, 'lr'), 'setup')} the share is ${T.human(traj(DIVERGE, 'kept'), 'pct')}, and each jump is longer than the last.</p>
    <p>A rate that is too small wastes steps, and a rate that is too large overshoots. So you try a few rates and watch the loss.</p>
    ${inTorc('The step in the code', `<p>In TorC, <code>optim::SGD</code> does the same step for every weight. Its <code>step()</code> method has one line for it:</p>
    <pre tabindex="0" role="group" aria-label="One line of TorC code"><code>param-&gt;data() = param-&gt;data().sub(grad.mul(lr_));</code></pre>`)}`),
    },
    {
      kind: 'quiz', id: 'steep-step', purpose: 'transfer', type: 'numeric',
      answer: steep('trajectories.0.states.0.w_next'), tolerance: 0.01,
      prompt: (D, T) => `<p>Go back to the four notebooks. At <code>w</code> = ${T(steep('probes.0.w'), 'setup')} the gradient is ${T(steep('probes.0.grad'), 'setup')}. You use the learning rate ${T(steep('trajectories.0.lr'), 'setup')} again. Where does the first step land?</p>`,
      reveal: (D, T) => `${worked(`<p>new <code>w</code> = ${T(steep('probes.0.w'), 'worked')} − ${T(steep('trajectories.0.lr'), 'worked')} × (${T(steep('probes.0.grad'), 'worked')}) = <strong>${T(steep('trajectories.0.states.0.w_next'), 'worked')}</strong></p>`)}
    <p>The step lands past the bottom at ${T(steep('setup.w_min'), 'reveal')}. On steeper ground, the same learning rate makes a longer step. A rate that works on one loss curve can overshoot on another.</p>`,
    },
    {
      kind: 'quiz', id: 'worst-guess', purpose: 'cumulative', from: 'what-is-learning', type: 'choice',
      prompt: (D, T) => `<p>Two notebooks cost ${T(`${F}.setup.t`, 'setup')} euros. Which guess for the price of one notebook gives the largest loss?</p>`,
      options: [
        { html: (D, T) => `<code>w</code> = ${T(`${F}.probes.3.w`, 'setup')}`, feedback: 'The loss measures the miss, not the size of the guess.' },
        { html: (D, T) => `<code>w</code> = ${T(`${F}.probes.4.w`, 'setup')}`, feedback: 'This guess is the true price, so its loss is zero.' },
        { html: (D, T) => `<code>w</code> = ${T(`${F}.probes.5.w`, 'setup')}`, correct: true, feedback: 'Yes. This guess gives the largest miss.' },
      ],
      reveal: (D, T) => `<p>With <code>w</code> = ${T(`${F}.probes.5.w`, 'reveal')}, the prediction is ${T(`${F}.probes.5.p`, 'reveal')} euros. It misses by ${T(`${F}.probes.5.e`, 'reveal')} euros, as much as the start does.</p>
    <p>So its loss is ${T(`${F}.probes.5.L`, 'reveal')}. The loss squares the miss, so a miss in either direction counts.</p>`,
    },
    {
      kind: 'explain', id: 'gradient-says',
      prompt: '<p>Explain what the gradient tells you, and what it does not tell you.</p>',
      model: '<p>The gradient tells which way the loss goes down and how steep the curve is here, but not how far away the bottom is.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>The computer cannot see the whole loss curve. At its current point it measures the gradient. The gradient tells which way is downhill and how steep the curve is there.</p>
    <p>Then the computer takes a step against the gradient, scaled by the learning rate. If the learning rate is too large, each step overshoots. So you try a few learning rates and watch the loss.</p>`,
      next: () => '<p>One weight needs one gradient. What changes when the model has two weights, or thousands?</p>',
    },
  ],
};
