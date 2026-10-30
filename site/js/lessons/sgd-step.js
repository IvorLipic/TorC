// Lesson "sgd-step" (template 3, curriculum v3 Part 2): gradient descent with two weights. A line y = w·x + b through three listed
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
    <p class="lead">In the last lesson, one ${term('weight')} learned the price of a notebook. Now two weights learn a line through three points.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        `A line has two weights. How do you find the ${term('gradient')} when the ${term('loss')} depends on both?`,
        'Does a step go straight toward the lowest loss?',
        `With two weights, what sets the largest safe ${term('learning-rate')}?`,
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('rides', 'Three taxi rides', `
    <p>A taxi ride costs a fixed fee plus a price per kilometre. Three rides cost this much:</p>
    <ul>${[0, 1, 2].map((i) => `<li>A ride of ${T(`${O}.data.points.${i}.x`, 'setup')} km costs ${T(`${O}.data.points.${i}.t`, 'setup')} euros.</li>`).join('')}</ul>
    <p>Each extra kilometre adds ${T(`${O}.data.true_w`, 'setup')} euros, so the fee is ${T(`${O}.data.true_b`, 'setup')} euro. You can find this by hand. A network with thousands of weights has no such shortcut, so we practise the general method here.</p>
    <p>The ${term('model')} is the line <code>y = w·x + b</code>, where <code>x</code> is the length of a ride in kilometres. It has two weights: <code>w</code> is the price per kilometre, and <code>b</code> is the fee.</p>
    <p>The model starts at <code>w</code> = ${T(`${F}.w0`, 'setup')} and <code>b</code> = ${T(`${F}.b0`, 'setup')}. The loss <code>L</code> is the mean of the three squared misses. Its square root is the typical miss, about ${T.human(`${F}.start.loss`, 'rmse')} euros at the start.</p>`),
    },
    {
      kind: 'quiz', id: 'method', purpose: 'predict', type: 'choice',
      prompt: () => '<p>In the last lesson, a small nudge of the one weight gave the gradient. Now the loss depends on <code>w</code> and on <code>b</code>. How do you find how the loss depends on each weight?</p>',
      options: [
        { html: 'Nudge <code>w</code> with <code>b</code> fixed. Then nudge <code>b</code> with <code>w</code> fixed.', correct: true, feedback: 'Each nudge then shows the effect of one weight.' },
        { html: 'Nudge <code>w</code> and <code>b</code> together, and divide the change of the loss by the nudge.', feedback: 'One nudge of both mixes their effects. You cannot tell which weight changed the loss.' },
        { html: 'Nudge only <code>w</code>, and use the same ratio for <code>b</code>.', feedback: 'The two weights change the loss in different ways, so their ratios differ.' },
        { html: 'Nudge each weight alone, and add the two ratios into one number.', feedback: 'The sum hides which weight did what. A step needs one number for each weight.' },
      ],
      reveal: () => '<p>Nudge one weight at a time, and keep the other one fixed. Each nudge gives one number.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('nudges', '1 · Two weights, two nudges', `
    <p>Keep <code>w</code> at ${T(`${F}.w0`, 'setup')} and nudge <code>b</code> alone:</p>
    ${worked(`<p>Raise <code>b</code> by ${T(`${NB}.h`, 'worked')}. The loss falls from ${T(`${F}.nudge.L0`, 'worked')} to ${T(`${NB}.L_h`, 'worked')}, a change of ${T(`${NB}.dL`, 'worked')}.</p>
    <p>The ratio is ${T(`${NB}.dL`, 'worked')} ÷ ${T(`${NB}.h`, 'worked')} = ${T(`${NB}.slope`, 'worked')}.</p>`)}
    <p>This ratio is an estimate. Smaller nudges give ratios closer to ${T(`${F}.start.gb`, 'setup')}, and <code>L.backward()</code> gives ${T(`${F}.start.gb`, 'setup')} exactly. That number is the gradient for <code>b</code>.</p>`),
    },
    {
      kind: 'quiz', id: 'nudge-w', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${NW}.slope`, tolerance: 0.15,
      prompt: (D, T) => `<p>Keep <code>b</code> at ${T(`${F}.b0`, 'setup')} and raise <code>w</code> by ${T(`${NW}.h`, 'setup')}. The loss falls from ${T(`${F}.nudge.L0`, 'setup')} to ${T(`${NW}.L_h`, 'setup')}. What is the ratio for <code>w</code>?</p>`,
      reveal: (D, T) => `<p>The loss changes by ${T(`${NW}.dL`, 'reveal')}. Divided by ${T(`${NW}.h`, 'reveal')}, that is ${T(`${NW}.slope`, 'reveal')}. Smaller nudges come closer to ${T(`${F}.start.gw`, 'reveal')}, the value that <code>L.backward()</code> gives.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('map', '2 · The gradient on a map of the loss', `
    <p>Two weights take two nudges. <strong>The gradient is the list of both numbers, one for each weight:</strong></p>
    ${eq(`gradient = (${T(`${F}.start.gw`, 'setup')}, ${T(`${F}.start.gb`, 'setup')})`, 'The gradient at the start')}
    <p>A list of numbers in a fixed order is a ${term('vector')}. With many weights, the gradient is a vector with one number for each weight.</p>
    <p>A contour map shows the loss of every pair (<code>w</code>, <code>b</code>) in one picture. <strong>Each colour is the loss for one pair (<code>w</code>, <code>b</code>). Dark is low.</strong> Each thin contour line joins pairs with the same loss, and its number is that loss.</p>
    <p>The dark region is a long, narrow valley. The star marks its lowest point. The orange arrow is the gradient times lr, drawn from the current point, the pair (<code>w</code>, <code>b</code>) where you are now. It points uphill, and a step goes the opposite way.</p>
    ${refresher('contour-map')}`),
    },
    {
      kind: 'widget', id: 'map', className: 'widget', label: 'Interactive: gradient descent on the loss map, with fog',
      mount: (host, D) => mountDescent(host, D, { lrs: LR_PRESETS.slice(0, 1), fog: true }),
      tasks: () => [
        {
          do: 'Press Step once, and look at the two arrows. Then press Step two more times.',
          see: 'The arrows do not point at the star. The dashed step goes straight downhill, at a right angle to the contour lines, so the trail bends. Both arrows get shorter.',
        },
        {
          do: 'Press Reset, then Fog. Press Step ten times, then press Fog again.',
          see: 'Without the map, the steps still went downhill. The trail is on the floor of the valley, but it is still far from the star.',
        },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '3 · One step for both weights', `
    <p>In the last lesson, one step was <code>w</code> ${sym('arrow')} <code>w</code> − lr × gradient. The arrow means "replace <code>w</code> by". With two weights, each weight takes this step with its own number:</p>
    ${eq(math('w \\leftarrow w - \\mathrm{lr} \\times (\\text{gradient for } w)', true) + math('b \\leftarrow b - \\mathrm{lr} \\times (\\text{gradient for } b)', true), 'One step for both weights')}
    <p>Both numbers come from the same point, before either weight changes. With lr = ${T(lr(SLOW), 'setup')}, the first step is:</p>
    ${worked(`<p>new <code>w</code> = ${T(st(SLOW, 0, 'w'), 'worked')} − ${T(lr(SLOW), 'worked')} × (${T(st(SLOW, 0, 'gw'), 'worked')}) = <strong>${T(st(SLOW, 1, 'w'), 'worked')}</strong></p>
    <p>new <code>b</code> = ${T(st(SLOW, 0, 'b'), 'worked')} − ${T(lr(SLOW), 'worked')} × (${T(st(SLOW, 0, 'gb'), 'worked')}) = <strong>${T(st(SLOW, 1, 'b'), 'worked')}</strong></p>`)}
    <p>The fee <code>b</code> goes up, away from the best fee. <strong>So a step does not go straight toward the lowest point.</strong> It goes straight downhill from the current point, and downhill here is a different direction.</p>`),
    },
    {
      kind: 'quiz', id: 'second-step', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: st(SLOW, 2, 'w'), tolerance: 0.01,
      prompt: (D, T) => `<p>After the first step, <code>w</code> is ${T(st(SLOW, 1, 'w'), 'setup')}. The new gradient for <code>w</code> is ${T(st(SLOW, 1, 'gw'), 'setup')}, and lr is still ${T(lr(SLOW), 'setup')}. What is <code>w</code> after the second step?</p>`,
      reveal: (D, T) => `${worked(`<p>new <code>w</code> = ${T(st(SLOW, 1, 'w'), 'worked')} − ${T(lr(SLOW), 'worked')} × (${T(st(SLOW, 1, 'gw'), 'worked')}) = <strong>${T(st(SLOW, 2, 'w'), 'worked')}</strong></p>`)}
    <p>The gradient is smaller at the new point, so the step is shorter.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('real', '4 · The line in TorC that takes the step', `
    <p>An ${term('optimizer')} is the part of a program that changes the weights with their gradients. TorC's optimizer for plain ${term('gradient-descent')} is ${term('stochastic-gradient-descent', 'SGD')}. A later lesson explains the name. This line of <code>SGD::step()</code> does the work:</p>
    ${realLine(STEP_LINE, 'TorC keeps the weights in groups. This line runs once for each group and changes every number in it at once. Here each group holds one number, <code>w</code> or <code>b</code>.')}
    <p>The line reads <code>grad</code>, the gradient that <code>L.backward()</code> stored. So the optimizer does not compute the gradient. It reads the gradient that <code>L.backward()</code> found.</p>`),
    },
    {
      kind: 'quiz', id: 'big-lr', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>The map used lr = ${T(lr(SLOW), 'setup')}. Now take lr = ${T(lr(BIG), 'setup')}, four times as large.</p>
    <p>The first step for <code>w</code> subtracts ${T(lr(BIG), 'setup')} times ${T(st(BIG, 0, 'gw'), 'setup')}. Work out where <code>w</code> lands, and compare it with the best <code>w</code> of ${T(`${O}.data.true_w`, 'setup')}. What happens in ten steps?</p>`,
      options: [
        { html: 'The trail reaches the star about four times faster.', feedback: 'The learning rate is not a speed. A longer step can jump past the lowest point.' },
        { html: 'The trail reaches the star in one step, because <code>w</code> lands near the best value.', feedback: 'This reads a landing near the best <code>w</code> as arrival. But <code>b</code> moves away, and the next step jumps back.' },
        { html: 'The trail jumps further from the star at every step, and the loss grows.', correct: true, feedback: 'Each step starts downhill but lands higher than the one before.' },
        { html: 'The trail is faster, but not four times faster.', feedback: 'A slightly larger learning rate can do that. This one is too large.' },
      ],
      reveal: (D, T) => `<p>The first step takes <code>w</code> to ${T(st(BIG, 1, 'w'), 'reveal')}, near the best value. It takes <code>b</code> up to ${T(st(BIG, 1, 'b'), 'reveal')}.</p>
    <p>After ten steps, the typical miss is about ${T.human(`${F}.runs.${BIG}.final.loss`, 'rmse')} euros. With lr = ${T(lr(SLOW), 'reveal')}, it is about ${T.human(`${F}.runs.${SLOW}.final.loss`, 'rmse')} euros.</p>`,
    },
    {
      kind: 'widget', id: 'rates', className: 'widget', label: 'Interactive: three learning rates on the loss map',
      mount: (host, D) => mountDescent(host, D, { lrs: LR_PRESETS }),
      tasks: (D, T) => [
        {
          do: `Choose lr ${T(lr(ZIG), 'setup')} and press Step ten times.`,
          see: 'The trail jumps from one wall of the valley to the other, and the jumps shrink. It also creeps slowly along the floor toward the star.',
        },
        {
          do: `Choose lr ${T(lr(BIG), 'setup')} and press Step five times.`,
          see: 'The jumps across the valley grow, and the loss grows at every step. The trail crosses the floor at every step but never stays there.',
        },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('steep', '5 · The steep direction limits the learning rate', `
    <p>Across the valley, the walls are steep, and the gradient is large. Along the floor, the ground is almost flat, and the gradient is small. A step is lr times the gradient, so steps across the valley are long, and steps along the floor are short.</p>
    <p>With lr = ${T(lr(ZIG), 'setup')} in the widget, a long step jumps over the floor to the other wall, but each jump is shorter than the last. With lr = ${T(lr(BIG), 'setup')}, each jump lands higher than the last.</p>
    <p>The trail does not stop at the star. Each step uses only the gradient at its own point, and that gradient does not know where the star is.</p>
    <p><strong>So the steep direction sets the largest safe learning rate.</strong> Along the flat floor, lr = ${T(lr(BIG), 'setup')} would still be safe, but the steps there stay short and slow.</p>`),
    },
    {
      kind: 'quiz', id: 'steep-first', purpose: 'transfer', type: 'choice',
      prompt: () => '<p>Another model has two weights. A small change of the first weight changes the loss a lot. A small change of the second weight hardly changes it. You make the learning rate larger, a little at a time. What goes wrong first?</p>',
      options: [
        { html: 'The steps for the first weight jump back and forth, and the jumps grow.', correct: true, feedback: 'The first weight has the steep direction and the large gradient, so its steps overshoot first.' },
        { html: 'The steps for the second weight grow first.', feedback: 'The gradient for the second weight is small, so its steps stay short.' },
        { html: 'Both weights go wrong at the same learning rate.', feedback: 'The two directions have different steepness, so their steps overshoot at different learning rates.' },
        { html: 'Nothing, because a larger learning rate is always faster.', feedback: 'A step that is too long jumps over the low ground and lands higher.' },
      ],
      reveal: () => '<p>The steepest direction limits the learning rate. Here it belongs to the first weight. The second weight then moves slowly.</p>',
    },
    {
      kind: 'quiz', id: 'steep-bowl', purpose: 'cumulative', from: 'why-gradients', type: 'choice',
      prompt: (D, T) => `<p>Two notebooks cost ${T(`${G}.setup.t`, 'setup')} euros, and four notebooks cost ${T(`${G}.steep.setup.t`, 'setup')} euros. The true price is ${T.anchor('bowl_w_min')} euros in both cases, and both models start at <code>w</code> = ${T(`${G}.probes.0.w`, 'setup')}.</p>
    <p>The gradient there is ${T(`${G}.probes.0.grad`, 'setup')} with two notebooks and ${T(`${G}.steep.probes.0.grad`, 'setup')} with four. What does the larger gradient tell you?</p>`,
      options: [
        { html: 'The loss curve is steeper at the start.', correct: true, feedback: 'The gradient measures the steepness where you stand.' },
        { html: 'The true price is further away.', feedback: 'The distance is the same. The gradient does not measure it.' },
        { html: 'The model needs a larger learning rate.', feedback: 'A steeper curve makes each step longer at the same learning rate. So the steeper case needs a smaller learning rate.' },
      ],
      reveal: () => '<p>The gradient tells how steep the loss curve is at your point. The steep walls of the valley in this lesson work the same way.</p>',
    },
    {
      kind: 'explain', id: 'two-weights',
      prompt: () => '<p>Explain how one step of gradient descent changes two weights.</p>',
      model: () => '<p>It finds one gradient number for each weight and moves each weight against its own number, times the learning rate.</p>',
    },
    {
      kind: 'answer',
      html: () => '<p>You find the gradient by nudging one weight at a time, or with <code>L.backward()</code>. With two weights, it is a list of two numbers. A step moves each weight against its own number, times the learning rate. It goes straight downhill, which is often not toward the lowest point. The steep direction across the valley sets the largest safe learning rate.</p>',
      next: (D, T) => `<p>A network that reads handwritten digits has ${T.anchor('mlp_parameters')} weights. What is inside a model with that many weights?</p>`,
    },
  ],
};
