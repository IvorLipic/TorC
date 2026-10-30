// Lesson "why-gradients" (Part 2 `gradient`, curriculum v3, template 3): the gradient found by nudging, one step against it, the
// steep-bowl refutation (brief C3) and the bowl with fog mode. It implements the gold-standard core of brief Appendix B; the
// adaptations are listed in docs/lessons/why-gradients.md. Every number comes from data/why-gradients.json through a role-tagged
// T() (docs/authoring.md "Number roles"); formulas take their constants from the trace with T.num. The chain rule and the graph
// lab left this lesson for `backprop` (why-gradients.scenes.js keeps the graph-lab layout of scene `why` for it).
import { math } from '../core/katex.js';
import { worked, realLine, analogy, term, sym } from '../core/blocks.js';
import { mountNudgeLens } from '../widgets/nudge-lens.js';
import { mountTwoBowls } from '../widgets/two-bowls.js';
import { mountBowl } from '../widgets/bowl.js';
import { bowlFigure } from '../widgets/bowl-figure.js';

const F = 'files.why-gradients';
// recorded runs, by index (the test pins this order): lr 0.05, 0.1, 0.125, 0.25 (bounces), 0.3 (diverges)
const traj = (i, rest) => `${F}.trajectories.${i}.${rest}`;
const SLOW = 0, FAST = 1, BOUNCE = 3, DIVERGE = 4;
// the steep bowl (four notebooks): its run with lr 0.05 is steep.trajectories.0
const steep = (rest) => `${F}.steep.${rest}`;
const sec = (id, title, body) => `<section class="wg" aria-labelledby="wg-${id}"><h2 id="wg-${id}">${title}</h2>${body}</section>`;
const lr = '<span class="wg-lr">lr</span>';
const arrow = () => sym('arrow', '←');

const STYLE = `<style>
.wg p{max-width:44rem}
.wg h2{margin-top:2rem}
.wg-lr{font-style:normal}
.wg-fig{margin:.6rem 0;max-width:30rem}
.wg-fig figcaption{color:var(--muted);font-size:.9rem}
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
    <p class="lead">A ${term('model')} with one ${term('weight')} learns the price of a notebook.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'A model’s guess of a price is wrong. How does the computer know which way to change it?',
        'How big a change should it make at each step?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => {
        const files = D.files['why-gradients'], x = T.num(`${F}.setup.x`), t = T.num(`${F}.setup.t`);
        return sec('price', 'The price of a notebook', `
    <p>Two notebooks cost ${T(`${F}.setup.t`, 'setup')} euros. The model has one weight, <code>w</code>, which is its guess of the price of one notebook. Its ${term('prediction')} for two notebooks is ${math(`${x}w`, false)} euros.</p>
    <p>The ${term('loss')} <code>L</code> measures how wrong the prediction is. Here it is the square of the miss, ${math(`L = (${x}w - ${t})^2`, false)}.</p>
    <p>The model starts with the guess <code>w</code> = ${T(`${F}.probes.0.w`, 'setup')} and misses by ${T(`${F}.nudge.miss_before`, 'setup')} euros. So the loss is ${T(`${F}.nudge.miss_before`, 'setup')} squared, which is ${T(`${F}.probes.0.L`, 'setup')}.</p>
    <p>The true price is ${T(`${F}.setup.w_min`, 'setup')} euros. There the miss and the loss are zero.</p>
    ${bowlFigure({ grid: files.grid, start: files.probes[0], wMin: files.setup.w_min, caption: 'The loss for every guess <code>w</code>. This curve is a bowl, and its lowest point is the bottom.' })}
    <p>Here you can divide to find the price. A network for handwritten digits has ${T.anchor('mlp_parameters')} ${term('parameter', 'parameters')} and no such shortcut.</p>`);
      },
    },
    {
      kind: 'quiz', id: 'nudge', purpose: 'predict', type: 'numeric',
      answer: `${F}.nudge.stages.2.after`, tolerance: 0.01,
      prompt: (D, T) => `<p>A nudge is a small change of <code>w</code>. Raise <code>w</code> from ${T(`${F}.nudge.w0`, 'setup')} to ${T(`${F}.nudge.delta`, 'setup')}. The prediction becomes ${T(`${F}.nudge.stages.0.after`, 'setup')} euros, and the miss becomes ${T(`${F}.nudge.miss_after`, 'setup')} euros. What is the new loss? A calculator helps.</p>`,
      reveal: (D, T) => `<p>The new loss is ${T(`${F}.nudge.miss_after`, 'reveal')}² = ${T(`${F}.nudge.stages.2.after`, 'reveal')}. So the loss changed by about ${T(`${F}.nudge.actual_dL`, { role: 'reveal', sig: 2 })}.</p>
    <p>Did you subtract from the old loss? The loss is the square of the miss, so square the new miss.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('gradient', '1 · The gradient', `
    <p>Divide the change of the loss by the nudge. A nudge ten times smaller changes the loss about ten times less, so this ratio stays almost the same.</p>
    <p>As the nudge gets smaller, the ratio comes closer and closer to one number, ${T(`${F}.probes.0.grad`, 'setup')}. This number is the ${term('gradient')}. It is the slope of the loss curve at the current point.</p>
    <p>We write it ${sym('dl-dw', math('\\frac{dL}{dw}', false))} and read it as “how much the loss changes per unit change of <code>w</code>”.</p>
    <p>A negative gradient means that a larger <code>w</code> gives a smaller loss, and a positive one means the opposite.</p>`),
    },
    {
      kind: 'widget', id: 'nudge-lens', className: 'widget', label: 'Interactive: the nudge lens',
      mount: (host, D) => mountNudgeLens(host, D),
      tasks: (D, T) => [{
        do: `Choose the nudge ${T(`${F}.secants.0.h`, 'setup')}, then smaller nudges down to ${T(`${F}.secants.4.h`, 'setup')}.`,
        see: 'The loss curve and the straight line come closer, until they look like one line. The actual change divided by the nudge comes closer to the gradient.',
      }],
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '2 · One step', `
    <p>${term('training', 'Training')} changes <code>w</code> by the gradient times a small number, in the opposite direction. That small number is the ${term('learning-rate')}, written ${sym('lr', lr)}. Here we choose ${lr} = ${T(traj(SLOW, 'lr'), 'setup')}.</p>
    ${worked(`<p>new <code>w</code> = <code>w</code> − ${lr} × gradient = ${T(`${F}.probes.0.w`, 'worked')} − ${T(traj(SLOW, 'lr'), 'worked')} × (${T(`${F}.probes.0.grad`, 'worked')}) = <strong>${T(traj(SLOW, 'states.0.w_next'), 'worked')}</strong></p>`)}
    <p>Subtracting a negative number adds, so <code>w</code> goes up toward the true price.</p>
    <p>Programs write “new <code>w</code> =” as <code>w</code> ${arrow()}. Read the arrow as “replace <code>w</code> by”.</p>
    ${realLine('L.backward();', 'In this example the loss is the variable <code>L</code>. This line finds the gradient without nudges, and a later lesson shows how. TorC puts the gradient in a slot that belongs to <code>w</code>.')}`),
    },
    {
      kind: 'quiz', id: 'your-step', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: traj(SLOW, 'states.1.w_next'), tolerance: 0.01,
      prompt: (D, T) => `<p>After the first step, <code>w</code> is ${T(traj(SLOW, 'states.1.w'), 'setup')}, and the gradient there is ${T(traj(SLOW, 'states.1.grad'), 'setup')}. The learning rate is still ${T(traj(SLOW, 'lr'), 'setup')}. Where does the second step land?</p>`,
      reveal: (D, T) => `${worked(`<p><code>w</code> ${arrow()} ${T(traj(SLOW, 'states.1.w'), 'worked')} − ${T(traj(SLOW, 'lr'), 'worked')} × (${T(traj(SLOW, 'states.1.grad'), 'worked')}) = <strong>${T(traj(SLOW, 'states.1.w_next'), 'worked')}</strong></p>
    <p>The distance to the bottom is ${T(traj(SLOW, 'states.0.distance'), 'worked')}, then ${T(traj(SLOW, 'states.1.distance'), 'worked')}, then ${T(traj(SLOW, 'states.1.distance_next'), 'worked')}.</p>
    <p>${T(traj(SLOW, 'states.1.distance'), 'worked')} is ${T(traj(SLOW, 'kept_percent'), { role: 'worked', sig: 2 })}% of ${T(traj(SLOW, 'states.0.distance'), 'worked')}, and ${T(traj(SLOW, 'states.1.distance_next'), 'worked')} is ${T(traj(SLOW, 'kept_percent'), { role: 'worked', sig: 2 })}% of ${T(traj(SLOW, 'states.1.distance'), 'worked')}.</p>`)}
    <p>So each step keeps ${T(traj(SLOW, 'kept_percent'), { role: 'reveal', sig: 2 })}% of the distance that is left. The steps get shorter because the bowl is flatter near the bottom.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('steep', '3 · A tempting idea that is wrong', `
    <p>Many people read a large gradient as “the answer is far away”.</p>
    <p>Suppose that four notebooks cost ${T(steep('setup.t'), 'setup')} euros. The true price is still ${T(steep('setup.w_min'), 'setup')}, and the model starts again at <code>w</code> = ${T(steep('probes.0.w'), 'setup')}. So the distance to the answer is the same.</p>
    <p>But the miss at the start is twice as large, and a nudge changes the prediction twice as much. Twice times twice makes the gradient four times larger, ${T(steep('probes.0.grad'), 'contrast')} instead of ${T(`${F}.probes.0.grad`, 'contrast')}.</p>
    <p>So the gradient tells you how steep the loss curve is where you stand. It does not tell you how far away the bottom is.</p>`),
    },
    {
      kind: 'widget', id: 'two-bowls', className: 'widget', label: 'Interactive: two bowls with the same bottom',
      mount: (host, D) => mountTwoBowls(host, D),
      tasks: (D, T) => [{
        do: `Choose the weight ${T(steep('compare.1.w'), 'setup')}, then the weight ${T(steep('compare.3.w'), 'setup')}.`,
        see: 'At each weight both bowls are equally far from the bottom, and the four-notebook gradient is four times as large.',
      }],
    },
    {
      kind: 'text',
      html: () => sec('fog', '4 · Steps in the fog', `
    <p>Nobody can draw the loss curve of a large network. Training finds the gradient where it is, takes one step, and repeats.</p>
    ${analogy('You stand on a hillside in thick fog. You feel which way the ground goes down under your feet, and how steep it is. You cannot see the valley.',
    'A hiker takes shorter steps where the ground is steep. A training step gets longer there, because it is the learning rate times the gradient.')}
`),
    },
    {
      kind: 'widget', id: 'bowl', className: 'widget', label: 'Interactive: step down the bowl in the fog',
      mount: (host, D) => mountBowl(host, D),
      tasks: (D, T) => [
        { do: `Turn the fog on. Press Step with ${T(traj(SLOW, 'lr'), 'setup')} until the gradient is near zero, then lift the fog.`,
          see: 'In the fog you see only your point and the dashed line of its gradient. The steps get shorter, and <code>w</code> comes close to the bottom.' },
        { do: `Choose ${T(traj(BOUNCE, 'lr'), 'setup')}. Guess the new <code>w</code> out loud, then press Step.`,
          see: '<code>w</code> jumps to the other side of the bottom, exactly as far from it as the start was. The next step jumps back to the start.' },
        { do: `Choose ${T(traj(DIVERGE, 'lr'), 'setup')} and press Step a few times.`,
          see: 'Each jump lands farther from the bottom, and the loss grows.' },
      ],
    },
    {
      kind: 'quiz', id: 'best-lr', purpose: 'check', type: 'choice',
      prompt: (D, T) => `<p>You start at <code>w</code> = ${T(traj(FAST, 'states.0.w'), 'setup')} and press Step ${T(traj(FAST, 'states.8.k'), 'setup')} times. Which learning rate brings <code>w</code> closest to the bottom? Try them in the widget above.</p>`,
      options: [
        { html: (D, T) => T(traj(SLOW, 'lr'), 'setup'), feedback: 'A smaller learning rate is not always better. Its steps cover less of the distance.' },
        { html: (D, T) => T(traj(FAST, 'lr'), 'setup'), correct: true, feedback: 'Yes. Its steps are long, but they never go past the bottom.' },
        { html: (D, T) => T(traj(BOUNCE, 'lr'), 'setup'), feedback: 'It jumps across the bottom and back forever.' },
        { html: (D, T) => T(traj(DIVERGE, 'lr'), 'setup'), feedback: 'Each jump is longer than the last.' },
      ],
      reveal: (D, T) => `<p>A step with ${T(traj(FAST, 'lr'), 'reveal')} keeps ${T.human(traj(FAST, 'kept'), 'pct')} of the distance. A step with ${T(traj(SLOW, 'lr'), 'reveal')} keeps ${T.human(traj(SLOW, 'kept'), 'pct')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('lr', '5 · How to choose the learning rate', `
    <p>A larger learning rate keeps a smaller share of the distance, but only up to a point. At ${lr} = ${T(traj(BOUNCE, 'lr'), 'setup')} the share is ${T.human(traj(BOUNCE, 'kept'), 'pct')}, so <code>w</code> jumps across the bottom and back forever.</p>
    <p>At ${lr} = ${T(traj(DIVERGE, 'lr'), 'setup')} the share is ${T.human(traj(DIVERGE, 'kept'), 'pct')}, and each jump is longer than the last.</p>
    <p>A rate that is too small wastes steps. A rate that is too large overshoots, which means that its steps go past the bottom. So you try a few rates and watch the loss.</p>`),
    },
    {
      kind: 'quiz', id: 'steep-step', purpose: 'transfer', type: 'numeric',
      answer: steep('trajectories.0.states.0.w_next'), tolerance: 0.01,
      prompt: (D, T) => `<p>Go back to the four notebooks. At <code>w</code> = ${T(steep('probes.0.w'), 'setup')} the gradient is ${T(steep('probes.0.grad'), 'setup')}. You use the learning rate ${T(steep('trajectories.0.lr'), 'setup')} again. Where does the first step land?</p>`,
      reveal: (D, T) => `${worked(`<p><code>w</code> ${arrow()} ${T(steep('probes.0.w'), 'worked')} − ${T(steep('trajectories.0.lr'), 'worked')} × (${T(steep('probes.0.grad'), 'worked')}) = <strong>${T(steep('trajectories.0.states.0.w_next'), 'worked')}</strong></p>`)}
    <p>The step lands past the bottom at ${T(steep('setup.w_min'), 'reveal')}. On steeper ground the same learning rate makes a longer step, so it can overshoot.</p>`,
    },
    {
      kind: 'quiz', id: 'loss-of-lines', purpose: 'cumulative', from: 'what-is-learning', type: 'choice',
      prompt: '<p>Think of a line through the test scores of ten students. Line A misses each student by one point. Line B hits five students and misses five by two points. Which line has the smaller loss?</p>',
      options: [
        { html: 'Line A', correct: true, feedback: 'Yes. The loss squares each miss, so large misses count more.' },
        { html: 'Line B', feedback: 'Line B hits half of the students, but each of its misses adds four.' },
        { html: 'They have the same loss', feedback: 'Both lines miss by ten points in total. But the loss squares each miss first, so large misses count more.' },
      ],
      reveal: '<p>The loss of line A is the mean of ten ones, which is one. Line B has five squares of four, so its mean is two.</p>',
    },
    {
      kind: 'explain', id: 'gradient-says',
      prompt: '<p>Explain what the gradient tells you, and what it does not tell you.</p>',
      model: '<p>The gradient tells which way is downhill and how steep it is here, but not how far the bottom is.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>The computer cannot see the whole loss curve. At its current point it finds the gradient, which tells which way is downhill and how steep it is.</p>
    <p>Then it steps against the gradient, scaled by the learning rate. If the learning rate is too large, the steps overshoot.</p>`,
      next: () => '<p>One weight needs one gradient. What changes when the model has two weights, or thousands?</p>',
    },
  ],
};
