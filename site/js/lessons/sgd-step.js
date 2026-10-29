// Lesson "sgd-step" (template 3, curriculum v3 Part 2): gradient descent with two weights. A line y = w·x + b through three listed
// points (taxi rides), the gradient found by nudging w alone and b alone (a list of two numbers, drawn as an arrow uphill on the
// contour map), one step for both weights, the SGD::step line, fog in 2D, and why a learning rate can be too large in the steep
// direction (the zig-zag picture). Momentum and zero_grad are taught later (brief C6). Numbers come from data/optim.json and
// data/why-gradients.json through T(); widget: descent.js (on contour2d.js and the optimizer port ../optim.js).
// site/test/sgd-step.test.mjs checks every claim of the prose against the data.
import { mountDescent, LR_PRESETS } from '../widgets/descent.js';
import { SOURCES } from '../widgets/sgd-code.js';
import { math } from '../core/katex.js';
import { worked, realLine, refresher, term, sym } from '../core/blocks.js';
import { REFRESHERS } from '../content/refreshers.js';

const O = 'files.optim';
const G = 'files.why-gradients';
// recorded runs, by index: 0 lr 0.05 · 1 lr 0.1 · 2 lr 0.15 · 3 lr 0.2 (the test pins this order)
const PLAIN = 0, ZIG = 2, BIG = 3;
const lr = (run) => `${O}.runs.${run}.lr`;
const NW = `${O}.nudge_start.w.2`;   // w alone, nudge 0.1 (the predict quiz)
const NB = `${O}.nudge_start.b.3`;   // b alone, nudge 0.01 (the worked example)
// the real line of SGD::step() without momentum, copied from SOURCES (so it can never drift from the quoted library source)
export const STEP_LINE = SOURCES.step.lines[27 - SOURCES.step.from].trim();
// the contour-map refresher comes from the reference content; until it exists the page does without it
const maybeRefresher = (key) => (Object.hasOwn(REFRESHERS, key) ? refresher(key) : '');

const sec = (id, title, body) => `<section class="sg" aria-labelledby="sg-${id}"><h2 id="sg-${id}">${title}</h2>${body}</section>`;
const eq = (html, label) => `<div class="sg-eq" tabindex="0" role="group" aria-label="${label}">${html}</div>`;

const STYLE = `<style>
.sg p,.sg ul{max-width:44rem}
.sg h2{margin-top:2rem}
.sg-eq{overflow-x:auto;margin:.3rem 0 .6rem;font-size:1.05rem}
.sg-eq .katex-display{margin:.3em 0}
</style>`;

export default {
  id: 'sgd-step',
  title: 'Gradient descent with many weights',
  trace: ['optim.json', 'why-gradients.json'],
  template: 3,
  loop: 'adjust',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Gradient descent with many weights</h1>
    <p class="lead">In the last lesson, one weight learned the price of a notebook. Now two weights learn a line through three points.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        `A line has two weights. How do you find the ${term('gradient')} when the ${term('loss')} depends on both?`,
        'How does one step change both weights?',
        `Why is a larger ${term('learning-rate')} not always faster?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('rides', 'Three taxi rides', `
    <p>A taxi ride costs a fixed fee plus a price per kilometre. Three rides cost this much:</p>
    <ul>${[0, 1, 2].map((i) => `<li>A ride of ${T(`${O}.data.points.${i}.x`, 'setup')} km costs ${T(`${O}.data.points.${i}.t`, 'setup')} euros.</li>`).join('')}</ul>
    <p>The model is the line <code>y = w·x + b</code>. Here <code>x</code> is the length of a ride in kilometres, and <code>y</code> is the predicted price.</p>
    <p>The model has two weights. The weight <code>w</code> is the price per kilometre, and the weight <code>b</code> is the fixed fee. Both start at ${T(`${O}.start.w`, 'setup')}, so the model predicts that every ride is free.</p>
    <p>The loss is the mean of the three squared misses. At the start, it is ${T(`${O}.start.loss`, 'setup')}, and a typical miss is about ${T.human(`${O}.start.loss`, 'rmse')} euros.</p>`),
    },
    {
      kind: 'quiz', id: 'nudge-w', purpose: 'predict', type: 'numeric',
      answer: `${NW}.slope`, tolerance: 0.15,
      prompt: (D, T) => `<p>Keep <code>b</code> at ${T(`${O}.start.b`, 'setup')}. Raise <code>w</code> from ${T(`${O}.start.w`, 'setup')} to ${T(`${NW}.h`, 'setup')}. The loss falls from ${T(`${O}.nudge_start.L0`, 'setup')} to ${T(`${NW}.L_h`, 'setup')}.</p>
    <p>How much does the loss change per unit change of <code>w</code>? Divide the change of the loss by the nudge.</p>`,
      reveal: (D, T) => `<p>The loss changes by ${T(`${NW}.dL`, 'reveal')}. Divided by the nudge of ${T(`${NW}.h`, 'reveal')}, that is ${T(`${NW}.slope`, 'reveal')}. The minus sign says that a larger <code>w</code> gives a smaller loss.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('nudges', '1 · Two weights, two nudges', `
    <p>Your ratio is the gradient of the loss for <code>w</code>, with <code>b</code> fixed. Smaller nudges give ratios closer to ${T(`${O}.start.gw`, 'setup')}.</p>
    <p>The weight <code>b</code> needs its own nudge, with <code>w</code> fixed:</p>
    ${worked(`<p>Raise <code>b</code> from ${T(`${O}.start.b`, 'worked')} to ${T(`${NB}.h`, 'worked')}. The loss falls from ${T(`${O}.nudge_start.L0`, 'worked')} to ${T(`${NB}.L_h`, 'worked')}, a change of ${T(`${NB}.dL`, 'worked')}.</p>
    <p>The ratio is ${T(`${NB}.dL`, 'worked')} ÷ ${T(`${NB}.h`, 'worked')} = ${T(`${NB}.slope`, 'worked')}.</p>`)}
    <p>Smaller nudges give ratios closer to ${T(`${O}.start.gb`, 'setup')}, the gradient for <code>b</code>.</p>
    <p>Two weights take two nudges. <strong>The gradient is the list of both numbers, one for each weight.</strong></p>
    ${eq(`gradient = (${T(`${O}.start.gw`, 'setup')}, ${T(`${O}.start.gb`, 'setup')})`, 'The gradient at the start')}
    <p>A list of numbers in a fixed order is a ${term('vector')}. So the gradient of a model with many weights is a vector. The call <code>loss.backward()</code> finds all its numbers at once.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('map', '2 · A map of the loss', `
    <p>Each pair (<code>w</code>, <code>b</code>) gives one line and one loss. A contour map shows all these losses from above.</p>
    <p><strong>Each colour is the loss for one pair (<code>w</code>, <code>b</code>). Dark is low.</strong> Each thin contour line joins the pairs that have the same loss.</p>
    <p>A star marks the lowest loss. The gradient is an arrow from your point. Its two numbers say how far the arrow goes along <code>w</code> and along <code>b</code>.</p>
    <p>The arrow points uphill, where the loss grows fastest. A step goes the opposite way.</p>
    ${maybeRefresher('contour-map')}`),
    },
    {
      kind: 'widget', id: 'map', className: 'widget', label: 'Interactive: gradient descent on the loss map, with fog',
      mount: (host, D) => mountDescent(host, D, { lrs: LR_PRESETS.slice(0, 1), fog: true }),
      tasks: () => [
        {
          do: 'Press Step three times, and look at the two arrows after each press.',
          see: 'The orange arrow is the gradient times lr. It points uphill, at a right angle to the contour lines. The dashed step points the other way, and both arrows get shorter.',
        },
        {
          do: 'Press Reset and Add fog. Press Step ten times, then Lift the fog.',
          see: 'You reached the dark part without seeing the map. The gradient alone showed the way downhill.',
        },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '3 · One step for both weights', `
    <p>In the last lesson, one step was <code>w</code> ${sym('arrow')} <code>w</code> − lr × gradient. The arrow means "replace <code>w</code> by". With two weights, each weight takes this step with its own number:</p>
    ${eq(math('w \\leftarrow w - \\mathrm{lr} \\times (\\text{gradient for } w)', true) + math('b \\leftarrow b - \\mathrm{lr} \\times (\\text{gradient for } b)', true), 'One step for both weights')}
    <p>Both numbers come from the same point, before either weight changes.</p>
    ${worked(`<p>With lr = ${T(lr(PLAIN), 'worked')}, the first step for <code>w</code> is:</p>
    <p>new <code>w</code> = ${T(`${O}.start.w`, 'worked')} − ${T(lr(PLAIN), 'worked')} × (${T(`${O}.start.gw`, 'worked')}) = <strong>${T(`${O}.runs.${PLAIN}.steps.1.w`, 'worked')}</strong></p>`)}
    <p>A minus sign before a negative number adds. So <code>w</code> grows toward the true price of ${T(`${O}.data.true_w`, 'setup')} euros per kilometre.</p>`),
    },
    {
      kind: 'quiz', id: 'b-step', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${O}.runs.${PLAIN}.steps.1.b`, tolerance: 0.01,
      prompt: (D, T) => `<p>At the start, <code>b</code> is ${T(`${O}.start.b`, 'setup')} and the gradient for <code>b</code> is ${T(`${O}.start.gb`, 'setup')}. The learning rate is ${T(lr(PLAIN), 'setup')}.</p><p>What is <code>b</code> after the first step?</p>`,
      reveal: (D, T) => `${worked(`<p>new <code>b</code> = ${T(`${O}.start.b`, 'worked')} − ${T(lr(PLAIN), 'worked')} × (${T(`${O}.start.gb`, 'worked')}) = <strong>${T(`${O}.runs.${PLAIN}.steps.1.b`, 'worked')}</strong></p>`)}
    <p>After this one step, a typical miss is about ${T.human(`${O}.runs.${PLAIN}.steps.1.loss`, 'rmse')} euros instead of ${T.human(`${O}.start.loss`, 'rmse')} euros.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('real', '4 · The line in TorC that takes the step', `
    <p>In TorC, the ${term('optimizer')} changes the weights. TorC calls plain gradient descent SGD, and this line of <code>SGD::step()</code> changes each weight:</p>
    ${realLine(STEP_LINE, 'This line runs once for each weight. It replaces the weight by the weight minus the gradient times lr.')}
    <p>The line reads <code>grad</code>, the gradient that <code>loss.backward()</code> stored. So the optimizer does not compute the gradient. It only uses it.</p>`),
    },
    {
      kind: 'quiz', id: 'big-lr', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>The map above used lr = ${T(lr(PLAIN), 'setup')}. What happens with lr = ${T(lr(BIG), 'setup')}, four times as large?</p>`,
      options: [
        { html: 'It reaches the star about four times faster.', feedback: 'The learning rate is not a speed. A long step can jump past the low part.' },
        { html: 'It is faster, but not four times faster.', feedback: 'This learning rate is too large for that.' },
        { html: 'Each step lands further from the star.', correct: true, feedback: 'Each step starts downhill but lands higher.' },
        { html: 'It stops after one step.', feedback: 'Nothing stops the steps. Each landing point gives the next step.' },
      ],
      reveal: (D, T) => `<p>After ten steps with lr = ${T(lr(PLAIN), 'reveal')}, a typical miss is about ${T.human(`${O}.after_ten_steps.with_zero_grad.loss`, 'rmse')} euros. With lr = ${T(lr(BIG), 'reveal')}, it is about ${T.human(`${O}.runs.${BIG}.final.loss`, 'rmse')} euros.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('steep', '5 · Why a step can be too long', `
    <p>The dark part of the map is a long, narrow valley. Its walls are steep, and its floor is almost flat.</p>
    <p>On a steep wall, the gradient is large, so the step is long. A step that is too long jumps over the floor to the other wall.</p>
    <p>With lr = ${T(lr(ZIG), 'setup')}, each jump is shorter than the last, and the trail settles. With lr = ${T(lr(BIG), 'setup')}, each jump lands higher than the last.</p>
    <p><strong>So a step against the gradient does not always lower the loss.</strong> The steep direction limits the learning rate. Along the flat floor, the steps stay short, so the trail is slow there.</p>`),
    },
    {
      kind: 'widget', id: 'rates', className: 'widget', label: 'Interactive: three learning rates on the loss map',
      mount: (host, D) => mountDescent(host, D, { lrs: LR_PRESETS }),
      tasks: (D, T) => [
        {
          do: `Choose lr ${T(lr(ZIG), 'setup')} and press Step five times.`,
          see: 'The trail jumps across the valley, and each jump is shorter. The loss falls at every step.',
        },
        {
          do: `Choose lr ${T(lr(BIG), 'setup')} and press Step five times.`,
          see: 'Each jump lands higher on the other wall. The loss grows at every step, and the view zooms out.',
        },
      ],
    },
    {
      kind: 'quiz', id: 'three-weights', purpose: 'transfer', type: 'choice',
      prompt: () => '<p>A model predicts the price of a pizza from its size and its number of toppings. It has three weights. How do you find its gradient?</p>',
      options: [
        { html: 'Nudge each weight alone. The three ratios are the gradient.', correct: true, feedback: 'Each weight gets its own number.' },
        { html: 'Nudge all three weights at once.', feedback: 'One nudge mixes the effects of the three weights.' },
        { html: 'Nudge each weight alone, and add the ratios.', feedback: 'The sum hides which weight did what.' },
      ],
      reveal: () => '<p>The gradient is a list of three numbers, one for each weight.</p>',
    },
    {
      kind: 'quiz', id: 'steep-bowl', purpose: 'cumulative', from: 'why-gradients', type: 'choice',
      prompt: (D, T) => `<p>Two notebooks cost ${T(`${G}.setup.t`, 'setup')} euros, and four notebooks cost ${T(`${G}.steep.setup.t`, 'setup')} euros. The true price is ${T.anchor('bowl_w_min')} euros in both cases.</p>
    <p>From the same start, the gradients are ${T(`${G}.probes.0.grad`, 'setup')} and ${T(`${G}.steep.probes.0.grad`, 'setup')}. What does the larger gradient tell you?</p>`,
      options: [
        { html: 'The loss curve is steeper at the start.', correct: true, feedback: 'The gradient measures the steepness where you stand.' },
        { html: 'The true price is further away.', feedback: 'The distance is the same. The gradient does not measure it.' },
        { html: 'Training needs a larger learning rate.', feedback: 'Steeper ground makes each step longer, so it needs a smaller one.' },
      ],
      reveal: () => '<p>The gradient tells how steep the loss curve is at your point. The steep valley walls in this lesson work the same way.</p>',
    },
    {
      kind: 'explain', id: 'two-weights',
      prompt: () => '<p>Explain how gradient descent changes two weights in one step.</p>',
      model: () => '<p>It finds one gradient number for each weight and moves each weight against its own number, times the learning rate.</p>',
    },
    {
      kind: 'answer',
      html: () => '<p>You find the gradient by nudging one weight at a time, or with <code>loss.backward()</code>. With two weights, it is a list of two numbers. One step moves each weight against its own number, times the learning rate. A larger learning rate can be worse, because a long step on a steep wall jumps over the valley.</p>',
      next: (D, T) => `<p>A network that reads handwritten digits has ${T.anchor('mlp_parameters')} weights. What is inside a model with that many weights?</p>`,
    },
  ],
};
