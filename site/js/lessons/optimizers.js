// Lesson "optimizers" (template 3, curriculum v3 Part 4 `training`, brief §6, §7, C4, D18): the narrow valley of sgd-step again.
// Momentum is taught once, here (C6), with the ball analogy and its limit; its loss rises at some steps and still ends far lower
// (the race), and it survives a learning rate at which plain SGD blows up (the sweep). Adam as "a step size per parameter, scaled by
// the typical size of its recent gradients", with its first step; the hats of m and s are said in words, not drawn. AdamW and weight
// decay as a slow leak (analogy limit: it never pulls exactly to zero), and TorC's rule in words: a parameter with no gradient is
// skipped, a gradient of zero still decays. The fair race (C4):
// each optimizer at its own best learning rate from a sweep, on two problems, the narrow valley and a round bowl; no optimizer wins
// on both. Every number comes from data/optimizers.json (tools/trace/trace_optimizers.cpp) through a role-tagged T();
// site/test/optimizers.test.mjs checks each claim of the prose.
import { math } from '../core/katex.js';
import { worked, analogy, term, sym } from '../core/blocks.js';
import { mountOptimizerRace } from '../widgets/optimizer-widgets.js';

const F = 'files.optimizers';
const SW = `${F}.sweep`;
// sweep optimizers by index: 0 sgd, 1 momentum, 2 adam, 3 adamw; learning rates by index (the test pins both orders)
const O_SGD = 0, O_MOM = 1;
const LR_SAFE = 3, LR_BIG = 4, LR_HUGE = 5;   // 0.019, 0.03, 0.1
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
    <p class="lead">During ${term('training')}, plain ${term('gradient-descent', 'gradient descent')} gives every ${term('parameter')} the same ${term('learning-rate', 'learning rate')}. Here you compare it with three other rules for the step.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        `Can an ${term('optimizer')} get through a narrow valley faster?`,
        'Should every parameter get the same step size?',
        'Is one optimizer the best for every problem?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => { const c = D.files.optimizers.valley.curvature; return sec('valley', 'Another narrow valley', `
    <p>The ${term('loss')} L depends on two parameters, w₁ and w₂:</p>
    ${eq(`L = \\tfrac{1}{2}\\,(w_1^2 + ${c[1]}\\,w_2^2)`, 'The loss of the narrow valley')}
    <p>It is almost flat along w₁ and steep along w₂, with the lowest point at w₁ = w₂ = ${T(`${F}.valley.surface.w1.45`, 'setup')}. So the ${term('gradient')} is large across the valley and small along its floor. Every run starts at w₁ = ${T(`${SW}.start.0`, 'setup')} and w₂ = ${T(`${SW}.start.1`, 'setup')}.</p>
    <p>Plain gradient descent is safe up to a learning rate of ${lrOf(T, LR_SAFE)}. At ${lrOf(T, LR_BIG)}, its steps across the valley grow, and the loss explodes. So it crawls along the flat floor, and after ${T(`${F}.valley.steps`, 'setup')} steps, w₁ is still about ${T(`${F}.race.0.path.50.w1`, 'contrast')}.</p>
    <p>TorC calls this optimizer ${term('stochastic-gradient-descent', 'SGD')}, although here each step uses the whole loss, so nothing is random. This lesson calls it plain SGD.</p>`); },
    },
    {
      kind: 'quiz', id: 'momentum', purpose: 'predict', type: 'choice',
      prompt: '<p>Now suppose that each step also adds most of the previous step. What happens in the narrow valley?</p>',
      options: [
        { html: 'It moves faster along the floor, and it swings across the valley.', correct: true, feedback: 'The kept steps add up along the floor, and they overshoot across it.' },
        { html: 'Nothing changes along the floor, because the gradient there stays small.', feedback: 'Along the floor, every gradient points the same way, so the kept steps add up.' },
        { html: 'It blows up, because the steps keep growing.', feedback: 'At a safe learning rate, the swings settle.' },
      ],
      reveal: '<p>Along the floor, the gradients all point the same way, so the kept steps add up. Across the valley, the gradients change sign, so they partly cancel.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('momentum', '1 · Momentum', `
    <p>${term('momentum', 'Momentum')} keeps a running step, the velocity v. Each step multiplies v by ${sym('mu', 'μ')} (mu), adds the new gradient, and moves the parameter against v. The arrow ${sym('arrow')} means “replace by”, and lr is the learning rate:</p>
    ${eq('v \\leftarrow \\mu\\, v + \\text{gradient} \\qquad w \\leftarrow w - \\mathrm{lr} \\cdot v', 'Momentum: the velocity and the step')}
    <p>Here μ = ${T(`${SW}.optimizers.${O_MOM}.momentum`, 'setup')}. In TorC, momentum is an option of SGD.</p>
    ${analogy('A ball that rolls down a slope keeps part of its speed from one moment to the next.', 'Here μ plays the role of friction, because it sets how much speed each step keeps. The velocity has no mass, so only the gradient pushes it.')}`),
    },
    {
      kind: 'widget', id: 'ball', className: 'widget', label: 'Interactive: plain SGD and momentum in the narrow valley',
      mount: (host, D) => mountOptimizerRace(host, D, { ids: ['sgd', 'momentum'], problems: ['valley'], note: true }),
      tasks: () => [
        { do: 'Press “+10 steps” until the end. Watch the loss of momentum.', see: 'Momentum swings across the valley, and its loss rises at some steps. Still, it ends far lower than plain SGD.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('stable', '2 · Momentum and larger learning rates', `
    <p>A loss that rises for a step can look like a failure. Here it is the swing across the valley, and the loss falls again.</p>
    <p>At ${lrOf(T, LR_BIG)}, plain SGD blew up. Momentum still ended lower than plain SGD at its best rate. Momentum has a limit too. At ${lrOf(T, LR_HUGE)}, it blew up as well.</p>
    <p>Its own best rate, ${T(`${SW}.optimizers.${O_MOM}.best.lr`, 'setup')}, is lower than that of plain SGD, because the kept steps add up to longer steps.</p>`),
    },
    {
      kind: 'quiz', id: 'first-step', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>Two parameters have the gradients ${T(`${FIRST.adam}.grad.0`, 'setup')} and ${T(`${FIRST.adam}.grad.1`, 'setup')}. Plain SGD uses lr = ${T(`${FIRST.sgd}.lr`, 'setup')}. It changes them by ${T(`${FIRST.sgd}.change.0`, 'setup')} and ${T(`${FIRST.sgd}.change.1`, 'setup')}. Now ${term('adam', 'Adam')}, another optimizer, takes its first step with the same lr. How far does it move each parameter?</p>`,
      options: [
        { html: (D, T) => `About ${T(`${FIRST.sgd}.lr`, 'setup')} each.`, correct: true, feedback: 'Adam divides each gradient by its typical size, so both steps are about lr.' },
        { html: 'As far as plain SGD moves them.', feedback: 'Adam does not step by the plain gradient times lr.' },
        { html: 'The small gradient moves its parameter more.', feedback: 'Adam evens out the sizes. It does not swap them.' },
      ],
      reveal: (D, T) => `<p>Adam changed both parameters by ${T(`${FIRST.adam}.change.1`, 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('adam', '3 · Adam', `
    <p>Adam gives each parameter its own step size, scaled by the typical size of its recent gradients. For each parameter, it keeps two averages that forget old steps slowly. One is the average gradient, m. The other gives the typical size of the recent gradients, √s.</p>
    ${eq('w \\leftarrow w - \\mathrm{lr} \\cdot \\frac{m}{\\sqrt{s} + \\varepsilon}', 'The step of Adam')}
    <p>Adam divides the average gradient by its typical size. ${sym('epsilon', 'ε')} (epsilon) is a tiny number that prevents a division by zero. At the start, both averages are too small, so TorC enlarges them in the first steps.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('adamw', '4 · AdamW and weight decay', `
    <p>${term('weight-decay', 'Weight decay')} pulls each parameter a little toward zero at every step. AdamW takes the same step as Adam, plus this pull with a strength ${sym('lambda', 'λ')} (lambda):</p>
    ${eq('w \\leftarrow w - \\mathrm{lr} \\cdot \\frac{m}{\\sqrt{s} + \\varepsilon} - \\mathrm{lr} \\cdot \\lambda \\cdot w', 'The step of AdamW')}
    ${analogy(`This decay is a slow leak that pulls each ${term('weight')} toward zero, unless the data holds it up.`, 'A leak can empty a tank. Weight decay takes a fixed part of the parameter at each step, so it never pulls it exactly to zero.')}
    <p>No gradient is not the same as a gradient of zero. If the loss does not use a parameter, TorC holds no gradient for it, and the step skips it. A gradient of zero still counts, so that parameter decays:</p>
    ${worked(`<p>with lr = λ = ${T(`${F}.decay_setup.lr`, 'worked')}, each step keeps 1 − ${T(`${F}.decay_setup.lr`, 'worked')} × ${T(`${F}.decay_setup.weight_decay`, 'worked')} = ${T(`${F}.derived.decay_factor`, 'worked')} of it, so ${T(`${ADAMW}.steps.0.w2`, 'worked')} becomes ${T(`${ADAMW}.steps.1.w2`, 'worked')}</p>`)}
    <p>The next lesson shows why this pull helps.</p>`),
    },
    {
      kind: 'quiz', id: 'decay-50', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${ADAMW}.steps.50.w2`, tolerance: 0.01,
      prompt: (D, T) => `<p>The same parameter starts at ${T(`${ADAMW}.steps.0.w2`, 'setup')}, and its gradient stays zero. How large is it after ${T(`${ADAMW}.steps.50.step`, 'setup')} steps? A calculator is fine.</p>`,
      reveal: (D, T) => `<p>Fifty steps multiply it by ${T(`${F}.derived.decay_factor`, 'reveal')} fifty times. So it ends at ${T(`${ADAMW}.steps.50.w2`, 'reveal')}, as in TorC.</p>`,
    },
    {
      kind: 'widget', id: 'race', className: 'widget', label: 'Interactive: four optimizers, each at its own best learning rate, on two problems',
      mount: (host, D) => mountOptimizerRace(host, D, { pickPath: true }),
      tasks: () => [
        { do: 'With the narrow valley, press “+10 steps” until the end.', see: 'Adam and AdamW end lowest, and plain SGD ends highest.' },
        { do: 'Choose the round bowl and go to the end.', see: 'Plain SGD reaches the lowest point within a few steps. The other three end higher.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('best', '5 · Which optimizer is best?', `
    <p>A fair race gives each optimizer its own best learning rate. We tried ten rates on each problem, from ${lrOf(T, 0)} to ${T(`${SW}.lrs.9`, 'setup')}, one run each. Only the learning rate changed. μ stayed at ${T(`${SW}.optimizers.${O_MOM}.momentum`, 'setup')}, and the λ of AdamW stayed at ${T(`${SW}.optimizers.3.weight_decay`, 'setup')}.</p>
    <p>On the narrow valley, Adam ended lowest. On the round bowl, every direction is equally steep, so one learning rate suits both parameters. There, plain SGD ended lowest. So no optimizer won on both problems.</p>
    <p>The best rate of Adam is much larger than that of plain SGD. Adam divides each step by the typical gradient size, so its rate is about the step size itself.</p>`),
    },
    {
      kind: 'quiz', id: 'classmate', purpose: 'transfer', type: 'choice',
      prompt: '<p>A classmate trains a new network with SGD and with Adam, both with the same learning rate. Adam ends lower. What should the classmate check before calling Adam better here?</p>',
      options: [
        { html: 'Each optimizer at its own best learning rate.', correct: true, feedback: 'One shared learning rate can suit one optimizer and not the other.' },
        { html: 'Nothing, because Adam is always the best optimizer.', feedback: 'On the round bowl, plain SGD beat Adam.' },
        { html: 'A smaller learning rate for Adam only.', feedback: 'A change for one optimizer only makes the race less fair.' },
      ],
      reveal: '<p>Compare each optimizer at its own best learning rate.</p>',
    },
    {
      kind: 'quiz', id: 'batch-noise', purpose: 'cumulative', from: 'training-loop', type: 'choice',
      prompt: () => `<p>In these races, each step used the gradient of the whole loss. In a real network, each step uses the gradient of one random ${term('batch')} of ${term('sample', 'samples')}. What does Adam get then?</p>`,
      options: [
        { html: 'A noisy estimate of the gradient.', correct: true, feedback: 'A batch gives an estimate that changes from batch to batch.' },
        { html: 'The exact gradient of all the samples.', feedback: 'Only all the samples give the exact gradient.' },
        { html: 'A gradient that points in a random direction.', feedback: 'The estimate points roughly the same way.' },
      ],
      reveal: '<p>Adam then works with noisy estimates.</p>',
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
