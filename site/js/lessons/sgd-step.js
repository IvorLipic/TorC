// Lesson "sgd-step" (template 3, curriculum v3 Part 2): gradient descent with two parameters (the weight w and the bias b). A line y = w·x + b through three listed
// points (taxi rides). The start w = 0, b = 3 lies off the steep axis of the loss valley (optim.json `off_axis`), so the gradient
// does not point at the lowest point: the trail bends, zig-zags across the valley or creeps along its floor. The page covers the
// gradient found by nudging w alone and b alone (a list of two numbers, drawn as an arrow uphill on the contour map), one step for
// both weights, the SGD::step line, fog in 2D, and why the steep direction sets the largest safe learning rate. Momentum and
// zero_grad are taught later (brief C6). Numbers come from data/optim.json and data/why-gradients.json through T(); widget:
// descent.js (on contour2d.js and the optimizer port ../optim.js). site/test/sgd-step.test.mjs checks every claim of the prose.
import { mountDescent, LR_PRESETS } from '../widgets/descent.js';
import { SOURCES } from '../widgets/sgd-code.js';
import { math } from '../core/katex.js';
import { worked, realLine, refresher, term, sym } from '../core/blocks.js';

const O = 'files.optim';
const F = `${O}.off_axis`;
const G = 'files.why-gradients';
// the off-axis runs, by index: 0 lr 0.05 · 1 lr 0.15 · 2 lr 0.2 (the test pins this order)
const SLOW = 0, ZIG = 1, BIG = 2;
const lr = (run) => `${F}.runs.${run}.lr`;
const st = (run, i, key) => `${F}.runs.${run}.steps.${i}.${key}`;
const NW = `${F}.nudge.w.2`;   // w alone, nudge 0.1
const NB = `${F}.nudge.b.2`;   // b alone, nudge 0.1
// the real line of SGD::step() without momentum, copied from SOURCES (so it can never drift from the quoted library source)
export const STEP_LINE = SOURCES.step.lines[27 - SOURCES.step.from].trim();

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
    <p class="lead">In the last lesson, one ${term('weight')} learned the price of a notebook. Now two ${term('parameter', 'parameters')} learn a line through three points.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        `How do you find the ${term('gradient')} when the ${term('loss')} depends on two parameters?`,
        'Does a step go straight to the lowest loss?',
        `What sets the largest safe ${term('learning-rate')}?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('rides', 'Three taxi rides', `
    <p>A taxi ride costs a fixed fee plus a price per kilometre:</p>
    <ul>${[0, 1, 2].map((i) => `<li>A ride of ${T(`${O}.data.points.${i}.x`, 'setup')} km costs ${T(`${O}.data.points.${i}.t`, 'setup')} euros.</li>`).join('')}</ul>
    <p>Each extra kilometre adds ${T(`${O}.data.true_w`, 'setup')} euros, so the fee is ${T(`${O}.data.true_b`, 'setup')} euro.</p>
    <p>The ${term('model')} is the line <code>y = w·x + b</code>, with <code>x</code> in kilometres. The weight <code>w</code> is the price per kilometre, and the ${term('bias')} <code>b</code> is the fixed fee.</p>
    <p>The model starts at <code>w</code> = ${T(`${F}.w0`, 'setup')} and <code>b</code> = ${T(`${F}.b0`, 'setup')}. The loss <code>L</code> is the mean of the three squared misses. Its square root, the typical miss, is about ${T.human(`${F}.start.loss`, 'rmse')} euros.</p>`),
    },
    {
      kind: 'quiz', id: 'method', purpose: 'predict', type: 'choice',
      prompt: () => '<p>The loss now depends on <code>w</code> and on <code>b</code>. How do you find the effect of each parameter?</p>',
      options: [
        { html: 'Nudge <code>w</code> with <code>b</code> fixed, then nudge <code>b</code> with <code>w</code> fixed.', correct: true, feedback: 'Each nudge shows the effect of one parameter.' },
        { html: 'Nudge <code>w</code> and <code>b</code> together, and divide by the nudge.', feedback: 'One nudge of both mixes their effects.' },
        { html: 'Nudge each parameter alone, and add the two ratios.', feedback: 'The sum hides which parameter did what.' },
      ],
      reveal: () => '<p>Each parameter gets its own nudge and its own number.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('nudges', '1 · Two parameters, two nudges', `
    <p>Keep <code>w</code> at ${T(`${F}.w0`, 'setup')} and nudge <code>b</code> alone:</p>
    ${worked(`<p>Raise <code>b</code> by ${T(`${NB}.h`, 'worked')}. The loss falls from ${T(`${F}.nudge.L0`, 'worked')} to ${T(`${NB}.L_h`, 'worked')}, a change of ${T(`${NB}.dL`, 'worked')}.</p>
    <p>The ratio is ${T(`${NB}.dL`, 'worked')} ÷ ${T(`${NB}.h`, 'worked')} = ${T(`${NB}.slope`, 'worked')}.</p>`)}
    <p>This ratio is an estimate. Smaller nudges come closer to ${T(`${F}.start.gb`, 'setup')}, and <code>L.backward()</code> gives ${T(`${F}.start.gb`, 'setup')} exactly. That is the gradient for <code>b</code>.</p>
    <p>The same nudge of <code>w</code>, with <code>b</code> fixed, gives the ratio ${T(`${NW}.slope`, 'setup')}. The gradient for <code>w</code> is ${T(`${F}.start.gw`, 'setup')}.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('map', '2 · The gradient on a map of the loss', `
    <p>Two parameters take two nudges. <strong>The gradient is the list of both numbers, one for each parameter:</strong></p>
    ${eq(`gradient = (${T(`${F}.start.gw`, 'setup')}, ${T(`${F}.start.gb`, 'setup')})`, 'The gradient at the start')}
    <p>A list of numbers in a fixed order is a ${term('vector')}.</p>
    <p>The contour map below shows the whole loss. <strong>Each colour is the loss for one pair (<code>w</code>, <code>b</code>). Dark is low.</strong> Each thin contour line joins pairs with the same loss, given by its number.</p>
    <p>The dark region is a long, narrow valley with its lowest point at the star. The orange arrow is the gradient times lr, drawn from the current point. It points uphill.</p>
    ${refresher('contour-map')}`),
    },
    {
      kind: 'widget', id: 'map', className: 'widget', label: 'Interactive: gradient descent on the loss map, with fog',
      mount: (host, D) => mountDescent(host, D, { lrs: LR_PRESETS.slice(0, 1), fog: true }),
      tasks: () => [
        {
          do: 'Press Step once, then four more times.',
          see: 'The arrows do not point at the star. Each step goes straight downhill, across the contour lines, so the trail bends.',
        },
        {
          do: 'Press Reset, then Fog. Press Step ten times, then Fog again.',
          see: 'Every step went downhill without the map. The trail is on the valley floor, far from the star.',
        },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '3 · One step for both parameters', `
    <p>In the last lesson, a step was <code>w</code> ${sym('arrow')} <code>w</code> − lr × gradient, where the arrow means "replace <code>w</code> by". Now each parameter takes this step with its own number:</p>
    ${eq(math('w \\leftarrow w - \\mathrm{lr} \\times (\\text{gradient for } w)', true) + math('b \\leftarrow b - \\mathrm{lr} \\times (\\text{gradient for } b)', true), 'One step for both parameters')}
    <p>Both numbers come from the same point. With lr = ${T(lr(SLOW), 'setup')}, the first step is:</p>
    ${worked(`<p>new <code>w</code> = ${T(st(SLOW, 0, 'w'), 'worked')} − ${T(lr(SLOW), 'worked')} × (${T(st(SLOW, 0, 'gw'), 'worked')}) = <strong>${T(st(SLOW, 1, 'w'), 'worked')}</strong></p>
    <p>new <code>b</code> = ${T(st(SLOW, 0, 'b'), 'worked')} − ${T(lr(SLOW), 'worked')} × (${T(st(SLOW, 0, 'gb'), 'worked')}) = <strong>${T(st(SLOW, 1, 'b'), 'worked')}</strong></p>`)}
    <p>The fee <code>b</code> goes up, away from the best fee. <strong>So a step goes straight downhill, which is not straight toward the lowest point.</strong></p>`),
    },
    {
      kind: 'quiz', id: 'second-step', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: st(SLOW, 2, 'w'), tolerance: 0.01,
      prompt: (D, T) => `<p>Now <code>w</code> is ${T(st(SLOW, 1, 'w'), 'setup')}, and its gradient is ${T(st(SLOW, 1, 'gw'), 'setup')}. With lr = ${T(lr(SLOW), 'setup')}, what is <code>w</code> after the second step?</p>`,
      reveal: (D, T) => `${worked(`<p>new <code>w</code> = ${T(st(SLOW, 1, 'w'), 'worked')} − ${T(lr(SLOW), 'worked')} × (${T(st(SLOW, 1, 'gw'), 'worked')}) = <strong>${T(st(SLOW, 2, 'w'), 'worked')}</strong></p>`)}
    <p>The gradient is smaller here, so the step is shorter.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('real', '4 · The line in TorC that takes the step', `
    <p>An ${term('optimizer')} is the part of a program that changes the parameters with their gradients. TorC's optimizer for plain ${term('gradient-descent')} is ${term('stochastic-gradient-descent', 'SGD')}.</p>
    ${realLine(STEP_LINE, 'TorC keeps the parameters in groups. This line runs once for each group and changes every number in it. Here each group holds one number, <code>w</code> or <code>b</code>.')}
    <p>The line reads <code>grad</code>, the gradient that <code>L.backward()</code> stored. So the optimizer does not compute the gradient.</p>`),
    },
    {
      kind: 'quiz', id: 'big-lr', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>The map used lr = ${T(lr(SLOW), 'setup')}. Now take lr = ${T(lr(BIG), 'setup')}, four times as large.</p>
    <p>The first step for <code>w</code> subtracts ${T(lr(BIG), 'setup')} times ${T(st(BIG, 0, 'gw'), 'setup')}. Where does <code>w</code> land, compared with its best value ${T(`${O}.data.true_w`, 'setup')}? What does the trail do?</p>`,
      options: [
        { html: 'It reaches the star four times faster.', feedback: 'A long step can jump past the lowest point.' },
        { html: 'It reaches the star in one step, because <code>w</code> lands near its best value.', feedback: 'That reads a landing as arrival. But <code>b</code> moves away.' },
        { html: 'It jumps back and forth, and the loss grows.', correct: true, feedback: 'Each step starts downhill but lands higher.' },
      ],
      reveal: (D, T) => `<p>The first step takes <code>w</code> to ${T(st(BIG, 1, 'w'), 'reveal')}, but <code>b</code> up to ${T(st(BIG, 1, 'b'), 'reveal')}. After ten steps, the typical miss is about ${T.human(`${F}.runs.${BIG}.final.loss`, 'rmse')} euros.</p>`,
    },
    {
      kind: 'widget', id: 'rates', className: 'widget', label: 'Interactive: three learning rates on the loss map',
      mount: (host, D) => mountDescent(host, D, { lrs: LR_PRESETS }),
      tasks: (D, T) => [
        {
          do: `Choose lr ${T(lr(ZIG), 'setup')} and press Step ten times.`,
          see: 'The jumps across the valley shrink, and the trail creeps along the floor toward the star.',
        },
        {
          do: `Choose lr ${T(lr(BIG), 'setup')} and press Step five times.`,
          see: 'The jumps across the valley grow, and so does the loss.',
        },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('steep', '5 · The steep direction limits the learning rate', `
    <p>The valley walls are steep, so the gradient is large and the steps across are long. The floor is almost flat, so the steps along it are short.</p>
    <p>With lr = ${T(lr(ZIG), 'setup')}, each jump across shrinks. With lr = ${T(lr(BIG), 'setup')}, each jump lands higher. The trail never stops at the star, because each step knows only the gradient at its own point.</p>
    <p><strong>So the steep direction sets the largest safe learning rate.</strong> Along the floor alone, lr = ${T(lr(BIG), 'setup')} would still be safe.</p>`),
    },
    {
      kind: 'quiz', id: 'steep-first', purpose: 'transfer', type: 'choice',
      prompt: () => '<p>Another loss is steep along one parameter and almost flat along the other. You slowly raise the learning rate. What goes wrong first?</p>',
      options: [
        { html: 'The steps of the steep parameter start to jump back and forth.', correct: true, feedback: 'The steep direction gives the longest steps, so they overshoot first.' },
        { html: 'The steps of the flat parameter grow first.', feedback: 'Its gradient is small, so its steps stay short.' },
        { html: 'Both go wrong at the same learning rate.', feedback: 'Their steepness differs, so they overshoot at different rates.' },
      ],
      reveal: () => '<p>The steepest direction limits the learning rate.</p>',
    },
    {
      kind: 'quiz', id: 'steep-bowl', purpose: 'cumulative', from: 'why-gradients', type: 'choice',
      prompt: (D, T) => `<p>Two notebooks cost ${T(`${G}.setup.t`, 'setup')} euros, and four cost ${T(`${G}.steep.setup.t`, 'setup')} euros. The price is ${T.anchor('bowl_w_min')} euros in both cases.</p>
    <p>At <code>w</code> = ${T(`${G}.probes.0.w`, 'setup')}, the gradient is ${T(`${G}.probes.0.grad`, 'setup')} for two notebooks. It is ${T(`${G}.steep.probes.0.grad`, 'setup')} for four. What does the larger gradient tell you?</p>`,
      options: [
        { html: 'The loss curve is steeper there.', correct: true, feedback: 'The gradient measures steepness.' },
        { html: 'The true price is further away.', feedback: 'The distance is the same.' },
        { html: 'The model needs a larger learning rate.', feedback: 'Steeper ground gives longer steps, so it needs a smaller learning rate.' },
      ],
      reveal: () => '<p>The steep valley walls in this lesson work the same way.</p>',
    },
    {
      kind: 'explain', id: 'two-weights',
      prompt: () => '<p>Explain how one step of gradient descent changes two parameters.</p>',
      model: () => '<p>It moves each parameter against its own gradient number, times the learning rate.</p>',
    },
    {
      kind: 'answer',
      html: () => '<p>Nudge one parameter at a time, or call <code>L.backward()</code>. The gradient is then a list of two numbers. A step moves each parameter against its own number, times the learning rate. It goes downhill, not always toward the lowest point. The steep direction sets the largest safe learning rate.</p>',
      next: (D, T) => `<p>A network that reads handwritten digits has ${T.anchor('mlp_parameters')} parameters. What is inside a model with that many parameters?</p>`,
    },
  ],
};
