// Fixture lesson for the v3 machinery (template: 3). It is not in course.json: site/test/v3.test.mjs places it third on the main
// path of a copy of the course (after sgd-step) and runs every v3 rule on it, then on broken copies of it. It uses every section
// kind (text with setup, questions, quiz choice and numeric with all four purposes, explain, widget with tasks, answer), every
// number role (setup, worked, reveal, anchor, contrast, human rmse/prob/pct/per100) and the blocks of core/blocks.js within the
// palette limit (analogy and refresher; inTorc and chatgpt are covered in v3.test.mjs). The Playwright check serves it as a draft
// page in a temporary copy of the site.
import { math } from '../../js/core/katex.js';
import { worked, realLine, analogy, refresher, term, sym } from '../../js/core/blocks.js';

const F = 'files.why-gradients';
const M = 'files.mnist-mlp';
const sec = (id, title, body) => `<section class="fx" aria-labelledby="fx-${id}"><h2 id="fx-${id}">${title}</h2>${body}</section>`;

export default {
  id: 'v3-fixture',
  title: 'Machinery fixture',
  trace: ['why-gradients.json', 'mnist-mlp.json'],
  template: 3,
  loop: 'adjust',
  sections: [
    {
      kind: 'text',
      html: () => '<section class="hero"><h1>Machinery fixture</h1><p class="lead">This page tests every part of a lesson. It is not part of the course.</p></section>',
    },
    {
      kind: 'questions',
      items: () => ['How does a computer know which way to change a weight?', 'How big should each change be?'],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('setup', 'The price of a notebook', `
    <p>Two notebooks cost ${T(`${F}.setup.t`, 'setup')} euros. The model has one weight <code>w</code>, its guess of the price of one notebook. The ${term('loss')} is the square of the miss.</p>
    <p>The model starts at <code>w</code> = ${T(`${F}.probes.0.w`, 'setup')}, and its loss is ${T(`${F}.probes.0.L`, 'setup')}.</p>`),
    },
    {
      kind: 'quiz', id: 'nudge', purpose: 'predict', type: 'numeric',
      answer: `${F}.nudge.stages.2.after`, tolerance: 0.01,
      prompt: (D, T) => `<p>Raise <code>w</code> to ${T(`${F}.nudge.delta`, 'setup')}. The miss becomes ${T(`${F}.nudge.stages.1.after`, 'setup')}. What is the new loss?</p>`,
      reveal: (D, T) => `<p>The new loss is ${T(`${F}.nudge.stages.2.after`, 'reveal')}. So the loss changed by ${T(`${F}.nudge.actual_dL`, 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('gradient', '1 · What the gradient says', `
    <p>The change of the loss divided by the size of the nudge is the ${term('gradient')}. At the start, the gradient is ${T(`${F}.probes.0.grad`, 'contrast')}. At <code>w</code> = ${T(`${F}.probes.5.w`, 'setup')}, it is ${T(`${F}.probes.5.grad`, 'contrast')}.</p>
    <p>A training step changes the weight in the direction against the gradient. The arrow ${sym('arrow')} means that <code>w</code> gets a new value.</p>
    <div class="fx-eq" tabindex="0" role="group" aria-label="Formula">${math('w \\leftarrow w - \\mathit{lr}\\cdot g')}</div>
    ${refresher('slope')}`),
    },
    {
      kind: 'widget', id: 'demo', className: 'widget', label: 'Interactive: a placeholder widget',
      mount: (host) => { host.innerHTML = '<p class="hint">A widget goes in this box.</p>'; },
      tasks: () => [{ do: 'Read the text in the box above.', see: 'The box shows where a widget goes on the page.' }],
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '2 · One step', `
    <p>Training changes <code>w</code> by the gradient times a small number, in the opposite direction. That small number is the ${term('learning-rate')}.</p>
    ${worked(`<p>new <code>w</code> = ${T(`${F}.probes.0.w`, 'worked')} − ${T(`${F}.trajectories.0.lr`, 'worked')} × (${T(`${F}.probes.0.grad`, 'worked')}) = ${T(`${F}.trajectories.0.states.0.w_next`, 'worked')}</p>`)}
    ${realLine('L.backward();', 'This is the line in TorC that computes the gradient. It stores the gradient in <code>w.grad()</code>.')}
    ${analogy('You stand on a hillside in thick fog. You can feel which way the ground slopes under your feet.', 'The gradient tells the steepness at your point. It does not tell where the bottom is.')}`),
    },
    {
      kind: 'quiz', id: 'direction', purpose: 'check', type: 'choice',
      prompt: (D, T) => `<p>At <code>w</code> = ${T(`${F}.probes.5.w`, 'setup')}, the gradient is positive. In which direction does one step move <code>w</code>?</p>`,
      options: [
        { html: 'It makes <code>w</code> smaller.', correct: true, feedback: 'The step subtracts a positive number, so <code>w</code> gets smaller.' },
        { html: 'It makes <code>w</code> larger.', feedback: 'This step follows the gradient. A training step goes against the gradient, so it subtracts.' },
        { html: 'It does not change <code>w</code>.', feedback: 'Only a gradient of zero gives no step. This gradient is not zero.' },
      ],
      reveal: (D, T) => `<p>The gradient is ${T(`${F}.probes.5.grad`, 'reveal')}. The step subtracts the learning rate times the gradient. So <code>w</code> goes down toward ${T.anchor('bowl_w_min')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('human', '3 · Results that a person can feel', `
    <p>At the start, the prediction misses the price by ${T.human(`${F}.probes.0.L`, 'rmse')} euros.</p>
    <p>A network for digits has ${T.anchor('mlp_parameters')} weights. After training, it reads ${T.human(`${M}.epochs.20.test_correct`, 'pct', { of: `${M}.setup.test_samples` })} of the test digits correctly.</p>
    <p>So it makes about ${T.human(`${M}.epochs.20.test_correct`, 'per100', { of: `${M}.setup.test_samples` })} mistakes per hundred digits. On a training image, it gives about ${T.human(`${M}.epochs.20.train_loss`, 'prob')} probability to the right digit.</p>`),
    },
    {
      kind: 'quiz', id: 'steeper', purpose: 'transfer', type: 'choice',
      prompt: () => '<p>Suppose that four notebooks cost twelve euros. The true price stays the same, and the model starts with the same guess. How does the gradient at the start change?</p>',
      options: [
        { html: 'It gets larger.', correct: true, feedback: 'A change of <code>w</code> now changes the prediction more, and the miss is larger.' },
        { html: 'It stays the same, because the best weight is the same.', feedback: 'The gradient measures the steepness at your point. It does not measure the distance to the best weight.' },
        { html: 'It gets smaller, because the loss is larger.', feedback: 'A larger loss here comes with a steeper curve. So the gradient grows.' },
      ],
      reveal: () => '<p>The prediction is now four times <code>w</code>. A small change of <code>w</code> changes the prediction twice as much as before, and the miss is larger. Both make the curve steeper.</p>',
    },
    {
      kind: 'quiz', id: 'old-step', purpose: 'cumulative', from: 'why-gradients', type: 'numeric',
      answer: `${F}.trajectories.0.states.1.w_next`, tolerance: 0.01,
      prompt: (D, T) => `<p>The weight is <code>w</code> = ${T(`${F}.trajectories.0.states.1.w`, 'setup')}, and the gradient there is ${T(`${F}.trajectories.0.states.1.grad`, 'setup')}. The learning rate is ${T(`${F}.trajectories.0.lr`, 'setup')}. Where does the next step land?</p>`,
      reveal: (D, T) => `<p>The step lands at ${T(`${F}.trajectories.0.states.1.w_next`, 'reveal')}. It is shorter than the first step, because the curve is flatter there.</p>`,
    },
    {
      kind: 'explain', id: 'why-against',
      prompt: () => '<p>Explain why a training step goes against the gradient.</p>',
      model: () => '<p>The gradient points in the direction in which the loss grows. So a step in the opposite direction makes the loss smaller.</p>',
    },
    {
      kind: 'answer',
      html: () => '<p>The computer measures the gradient at its current point. The gradient tells which way the loss goes down. Then the computer takes a step against the gradient, scaled by the learning rate.</p>',
      next: () => '<p>What changes when the model has many weights?</p>',
    },
  ],
};
