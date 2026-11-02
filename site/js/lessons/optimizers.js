// Lesson "optimizers" (template 3, curriculum v3 Part 4 `training`, brief §6, §7, C4, D18): the narrow valley of sgd-step again.
// Momentum is taught once, here (C6), with the ball analogy and its limit; its loss rises at some steps and still ends far lower
// (the race), and it survives a learning rate at which plain SGD blows up (the sweep). Adam as "a step size per parameter, scaled by
// the typical size of its recent gradients", with its first step; the hats of m and s are said in words, not drawn. AdamW and weight
// decay as a slow leak (analogy limit: TorC's step() skips a parameter with no gradient, so it does not leak). The fair race (C4):
// each optimizer at its own best learning rate from a sweep, on two problems, the narrow valley and a round bowl; no optimizer wins
// on both. Every number comes from data/optimizers.json (tools/trace/trace_optimizers.cpp) through a role-tagged T();
// site/test/optimizers.test.mjs checks each claim of the prose.
import { math } from '../core/katex.js';
import { worked, analogy, term, sym } from '../core/blocks.js';
import { mountOptimizerRace } from '../widgets/optimizer-widgets.js';

const F = 'files.optimizers';
const SW = `${F}.sweep`;
// sweep optimizers by index: 0 sgd, 1 momentum, 2 adam, 3 adamw; learning rates by index: 3 is 0.019, 4 is 0.03 (the test pins both)
const O_SGD = 0, O_MOM = 1, O_ADAM = 2;
const LR_SAFE = 3, LR_BIG = 4;
const FIRST = { sgd: `${F}.first_step.0`, adam: `${F}.first_step.1` };
const ADAMW = `${F}.decay.1`;
const lrOf = (T, i, role = 'setup') => T(`${SW}.lrs.${i}`, role);

const sec = (id, title, body) => `<section class="op" aria-labelledby="op-${id}"><h2 id="op-${id}">${title}</h2>${body}</section>`;
const eq = (latex, label) => `<div class="op-eq" tabindex="0" role="group" aria-label="${label}">${math(latex, true)}</div>`;
const STYLE = `<style>
.op p{max-width:44rem}
.op h2{margin-top:2rem}
.op-eq{overflow-x:auto;margin:.3rem 0}
.op-eq .katex-display{margin:.4em 0}
</style>`;

export default {
  id: 'optimizers',
  title: 'Better optimizers',
  trace: ['optimizers.json'],
  template: 3,
  loop: 'adjust',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Better optimizers</h1>
    <p class="lead">During ${term('training')}, plain ${term('gradient-descent', 'gradient descent')} gives every ${term('parameter')} the same ${term('learning-rate', 'learning rate')}. Other ${term('optimizer', 'optimizers')} change the step in other ways.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'Can a step remember where the last steps went?',
        'Should every parameter get the same step size?',
        'Is one optimizer the best for every problem?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => { const c = D.files.optimizers.valley.curvature; return sec('valley', 'The narrow valley again', `
    <p>The ${term('loss')} L depends on two parameters, w₁ and w₂:</p>
    ${eq(`L = \\tfrac{1}{2}\\,(w_1^2 + ${c[1]}\\,w_2^2)`, 'The loss of the narrow valley')}
    <p>It is almost flat along w₁ and steep along w₂, with the lowest point at w₁ = w₂ = ${T(`${F}.valley.surface.w1.45`, 'setup')}. So the ${term('gradient')} is large across the valley and small along its floor. Every run starts at w₁ = ${T(`${SW}.start.0`, 'setup')} and w₂ = ${T(`${SW}.start.1`, 'setup')}.</p>
    <p>Plain gradient descent is safe up to a learning rate of ${lrOf(T, LR_SAFE)}. At ${lrOf(T, LR_BIG)}, its steps across the valley grow, and the loss explodes. So it crawls along the flat floor, and after ${T(`${F}.valley.steps`, 'setup')} steps, w₁ is still about ${T(`${F}.race.0.path.50.w1`, 'contrast')}.</p>`); },
    },
    {
      kind: 'quiz', id: 'momentum', purpose: 'predict', type: 'choice',
      prompt: '<p>Now suppose that each step also adds most of the previous step. What happens in the narrow valley?</p>',
      options: [
        { html: 'It moves faster along the floor, and it swings across the valley.', correct: true, feedback: 'The kept steps add up along the floor, and they overshoot across it.' },
        { html: 'Nothing changes along the floor, because the gradient there stays small.', feedback: 'Along the floor, every gradient points the same way, so the kept steps add up.' },
        { html: 'It blows up, because the steps keep growing.', feedback: 'The path swings, but at a safe learning rate it settles. Each step keeps only a part of the last one.' },
      ],
      reveal: '<p>Along the floor, the gradients all point the same way, so the kept steps add up. Across the valley, the gradients change sign, so they partly cancel.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('momentum', '1 · Momentum', `
    <p>${term('momentum', 'Momentum')} keeps a running step, the velocity v. Each step multiplies v by ${sym('mu', 'μ')} (mu), adds the new gradient, and moves the parameter against v:</p>
    ${eq('v \\leftarrow \\mu\\, v + \\text{gradient} \\qquad w \\leftarrow w - \\mathrm{lr} \\cdot v', 'Momentum: the velocity and the step')}
    <p>Here μ = ${T(`${SW}.optimizers.${O_MOM}.momentum`, 'setup')}. In TorC, momentum is an option of ${term('stochastic-gradient-descent', 'SGD')}.</p>
    ${analogy('A ball that rolls down a slope keeps part of its speed from one moment to the next.', 'A real ball has mass and friction. Here only μ sets how much of the last step stays.')}`),
    },
    {
      kind: 'widget', id: 'ball', className: 'widget', label: 'Interactive: plain SGD and momentum in the narrow valley',
      mount: (host, D) => mountOptimizerRace(host, D, { ids: ['sgd', 'momentum'], problems: ['valley'] }),
      tasks: () => [
        { do: 'Move the slider step by step to the end. Watch the loss of momentum.', see: 'Momentum swings across the valley, and its loss rises at some steps. Still, it ends far lower than plain SGD.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('stable', '2 · Momentum does not make training unstable', `
    <p>A loss that rises for a step can look like a failure. Here it is the swing across the valley, and the loss falls again. Larger learning rates show more. At ${lrOf(T, LR_BIG)}, plain SGD blew up. Momentum ended lower than plain SGD at its own best rate.</p>`),
    },
    {
      kind: 'quiz', id: 'first-step', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>Two parameters have the gradients ${T(`${FIRST.adam}.grad.0`, 'setup')} and ${T(`${FIRST.adam}.grad.1`, 'setup')}. Plain SGD uses lr = ${T(`${FIRST.sgd}.lr`, 'setup')}. It changes them by ${T(`${FIRST.sgd}.change.0`, 'setup')} and ${T(`${FIRST.sgd}.change.1`, 'setup')}. An optimizer divides each gradient by its own typical size first. How far does its first step move each parameter?</p>`,
      options: [
        { html: (D, T) => `About ${T(`${FIRST.sgd}.lr`, 'setup')} each.`, correct: true, feedback: 'A gradient divided by its own size is about one, times the learning rate.' },
        { html: 'As far as plain SGD moves them.', feedback: 'The division removes the size of the gradient, so the steps change.' },
        { html: 'The small gradient moves its parameter more.', feedback: 'After the division, both gradients have about the same size.' },
      ],
      reveal: (D, T) => `<p>This optimizer is ${term('adam', 'Adam')}. It changed both parameters by ${T(`${FIRST.adam}.change.1`, 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('adam', '3 · Adam', `
    <p>Adam gives each parameter its own step size, scaled by the typical size of its recent gradients. For each parameter, it keeps m, the average of its recent gradients, and s, the average of their squares. So √s is their typical size.</p>
    ${eq('w \\leftarrow w - \\mathrm{lr} \\cdot \\frac{m}{\\sqrt{s} + \\varepsilon}', 'The step of Adam')}
    <p>Adam divides the average gradient by its typical size. Two numbers ${sym('beta', 'β')}₁ and β₂ (beta), just below one, set how slowly the averages forget. ${sym('epsilon', 'ε')} (epsilon) is a tiny number that prevents a division by zero. Both averages start at zero, so TorC scales them up in the first steps.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('adamw', '4 · AdamW and weight decay', `
    <p>${term('weight-decay', 'Weight decay')} pulls each parameter a little toward zero at every step. AdamW is Adam plus this pull, with a strength ${sym('lambda', 'λ')} (lambda):</p>
    ${eq('w \\leftarrow w - \\mathrm{lr} \\cdot (\\text{the Adam step}) - \\mathrm{lr} \\cdot \\lambda \\cdot w', 'The step of AdamW')}
    ${analogy(`This decay is a slow leak that pulls each ${term('weight')} toward zero, unless the data holds it up.`, 'In TorC, a parameter with no gradient is skipped by the step, so it does not leak.')}
    <p>Take a parameter whose gradient is zero, with lr = λ = ${T(`${F}.decay_setup.lr`, 'setup')}:</p>
    ${worked(`<p>each step keeps 1 − ${T(`${F}.decay_setup.lr`, 'worked')} × ${T(`${F}.decay_setup.weight_decay`, 'worked')} = ${T(`${F}.derived.decay_factor`, 'worked')} of it, so ${T(`${ADAMW}.steps.0.w2`, 'worked')} becomes ${T(`${ADAMW}.steps.1.w2`, 'worked')}</p>`)}
    <p>The next lesson shows why this pull helps.</p>`),
    },
    {
      kind: 'quiz', id: 'decay-2', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${ADAMW}.steps.2.w2`, tolerance: 0.001,
      prompt: (D, T) => `<p>The parameter is now ${T(`${ADAMW}.steps.1.w2`, 'setup')}, and its gradient is still zero. What is it after the next step? A calculator is fine.</p>`,
      reveal: (D, T) => worked(`<p>${T(`${ADAMW}.steps.1.w2`, 'worked')} × ${T(`${F}.derived.decay_factor`, 'worked')} = ${T(`${ADAMW}.steps.2.w2`, 'worked')}</p>`),
    },
    {
      kind: 'widget', id: 'race', className: 'widget', label: 'Interactive: four optimizers, each at its own best learning rate, on two problems',
      mount: (host, D) => mountOptimizerRace(host, D),
      tasks: () => [
        { do: 'With the narrow valley, move the slider to the end.', see: 'Adam and AdamW end lowest, and plain SGD ends highest.' },
        { do: 'Choose the round bowl and move the slider to the end.', see: 'Plain SGD reaches the lowest point within a few steps. The other three end higher.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('best', '5 · Which optimizer is best?', `
    <p>A fair race gives each optimizer its own best learning rate. We tried ten rates on each problem, from ${lrOf(T, 0)} to ${T(`${SW}.lrs.9`, 'setup')}, one run each.</p>
    <p>On the narrow valley, Adam ended lowest. On the round bowl, every direction is equally steep, so one learning rate suits both parameters. There, plain SGD ended lowest.</p>
    <p>Adam also needs its own learning rate. At the best rate of plain SGD, Adam ended higher than plain SGD on the valley. So no optimizer won on both problems.</p>`),
    },
    {
      kind: 'quiz', id: 'classmate', purpose: 'transfer', type: 'choice',
      prompt: '<p>A classmate trains a new network with SGD and with Adam, both with the same learning rate. Adam ends lower. What should the classmate check before calling Adam better here?</p>',
      options: [
        { html: 'Each optimizer at its own best learning rate.', correct: true, feedback: 'One shared learning rate can suit one optimizer and not the other.' },
        { html: 'Nothing, because Adam is always the best optimizer.', feedback: 'On the round bowl, plain SGD beat Adam.' },
        { html: 'A smaller learning rate for Adam only.', feedback: 'A change for one optimizer only makes the race less fair.' },
      ],
      reveal: '<p>Compare each optimizer at its own best learning rate, and on more than one problem if you can.</p>',
    },
    {
      kind: 'quiz', id: 'batch-noise', purpose: 'cumulative', from: 'training-loop', type: 'choice',
      prompt: () => `<p>In these races, each step used the gradient of the whole loss. In a real network, each step uses the gradient of one random ${term('batch')} of ${term('sample', 'samples')}. What does Adam get then?</p>`,
      options: [
        { html: 'A noisy estimate of the gradient.', correct: true, feedback: 'A batch gives an estimate that changes from batch to batch.' },
        { html: 'The exact gradient of all the samples.', feedback: 'Only all the samples give the exact gradient.' },
        { html: 'A gradient that points in a random direction.', feedback: 'The estimate points roughly the same way as the gradient of all the samples.' },
      ],
      reveal: '<p>Adam then works with noisy estimates, like plain SGD in the last lesson.</p>',
    },
    {
      kind: 'explain', id: 'adam-size',
      prompt: '<p>Explain how Adam chooses the size of the step for each parameter.</p>',
      model: '<p>Adam divides the average recent gradient of each parameter by its typical size. So small and large gradients give steps of similar size.</p>',
    },
    {
      kind: 'answer',
      html: '<p>Momentum keeps part of the last step, so it speeds along a valley floor. Adam gives each parameter its own step size, scaled by the typical size of its recent gradients. AdamW adds a slow pull toward zero. No optimizer won on both of our problems, so compare them each at its own best learning rate.</p>',
      next: () => `<p>A ${term('model')} can fit its training data very well. Does it then also work on new data?</p>`,
    },
  ],
};
