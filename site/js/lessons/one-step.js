// Lesson "one-step" (Part 5 `images`, curriculum v3, template 3), title "One training step, followed end to end" (brief §6
// "one-step"). One real training step of the trained digit network on one training image of a one (data/one-step.json,
// tools/trace/trace_one_step.cpp): the forward pass to the probabilities and the surprise, the backward pass to the scores and to
// four named parameters (two output weights, a first-layer weight and a bias), plain SGD with lr 0.001 on a batch of one image so
// that new = old − lr × gradient holds exactly (the step size is lr_times_grad), and the same image again. Every number causes
// the next. The capstone: each part of the course is recalled in one clause with a link, and the lesson ends with the number of
// steps of the real run (full_run.steps). Every number comes through a role-tagged T() (docs/authoring.md "Number roles").
// Tests: site/test/one-step.test.mjs. Notes: docs/lessons/one-step.md.
import { worked, term } from '../core/blocks.js';
import { mountOneStep } from '../widgets/one-step-widgets.js';

const F = 'files.one-step';
const W = (i, rest) => `${F}.weights.${i}.${rest}`;
// the named parameters by index (the test pins their ids): the first-layer weight, the output weights to the right digit and to
// the rival, the bias of the right digit
export const FIRST = 0, TRUE = 1, RIVAL = 2;
const sec = (id, title, body) => `<section class="os1" aria-labelledby="os1-${id}"><h2 id="os1-${id}">${title}</h2>${body}</section>`;
const a = (id, text) => `<a href="../${id}/">${text}</a>`;
const p3 = (T, path) => T(path, 'contrast', { sig: 3 });

const STYLE = `<style>
.os1 p,.os1 li{max-width:44rem}
.os1 h2{margin-top:2rem}
</style>`;

export default {
  id: 'one-step',
  title: 'One training step, followed end to end',
  trace: ['one-step.json'],
  template: 3,
  loop: 'all',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>One training step, followed end to end</h1>
    <p class="lead">You follow one handwritten digit through the network, back again, and through one change of the network.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'What happens to the numbers of the network when it learns from one image?',
        `How does the miss at the end reach a ${term('weight')} at the very start?`,
        'How much does one small step change the answer of the network?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('setup', 'One image, one step', `
    <p>Take the trained digit network of the ${a('mnist-mlp', 'last lesson')}. You give it one more ${term('training')} step on one training image, a handwritten one.</p>
    <p>The real training used AdamW and a ${term('batch')} of ${T(`${F}.full_run.batch_size`, 'setup')} images per step. This step uses plain ${term('gradient-descent', 'gradient descent')} on one image, with the ${term('learning-rate', 'learning rate')} ${T(`${F}.setup.lr`, 'setup')}. We chose this rule and this value, so that you can check every number.</p>
    <p>Before the step, the network reads the image as a one. It gives the one the ${term('probability')} ${T(`${F}.forward.p_true`, 'setup', { sig: 3 })}.</p>`),
    },
    {
      kind: 'quiz', id: 'after', purpose: 'predict', type: 'choice',
      prompt: '<p>The step changes each number of the network by a tiny amount. Then the same image goes through the network again. What happens to the probability of the one?</p>',
      options: [
        { html: 'It stays almost the same, because each change is tiny.', feedback: 'This belief forgets that thousands of numbers change at once, and that each change lowers the surprise for this image.' },
        { html: 'It rises clearly, but it stays below one.', correct: true, feedback: 'Yes.' },
        { html: 'It jumps to one, because the step removes the mistake.', feedback: `A step moves each number a little way against its ${term('gradient')}. It does not solve for the right answer.` },
      ],
      reveal: (D, T) => `<p>It rises to ${T(`${F}.after.p_true`, 'reveal', { sig: 3 })}. The sections below show why.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('forward', '1 · Forward, from pixels to a surprise', `
    <p>The ${T.anchor('pixels')} grey values of the image go into the first ${term('layer')}. Each ${term('neuron')} computes a weighted sum of all ${term('pixel', 'pixels')} plus a ${term('bias')}. ${term('relu', 'ReLU')} keeps the positive sums and sets the others to zero, as in ${a('nonlinearity', 'an earlier lesson')}.</p>
    <p>The second layer does the same with these values. The last layer gives ten scores, one for each digit. ${term('softmax', 'Softmax')} turns them into probabilities, and the ${term('loss')} is the surprise, the ${term('cross-entropy', 'cross-entropy')}. It is −ln of the probability of the right digit. Both come from the lesson on ${a('losses', 'softmax and cross-entropy')}.</p>
    <p>The network gives the one the largest probability. A three comes second.</p>`),
    },
    {
      kind: 'widget', id: 'walk', className: 'widget', label: 'Interactive: one training step, stage by stage',
      mount: (host, D) => mountOneStep(host, D),
      tasks: () => [
        { do: 'Press Next until the probabilities appear. Before you press again, find the second largest bar.',
          see: 'It belongs to the three. So the network hesitates between a one and a three.' },
        { do: 'Press Next through the two stages of the backward pass.',
          see: 'The score of the one gets a negative gradient. Every other score gets a positive gradient, the largest for the three.' },
        { do: 'Press Next to the end.',
          see: 'The bar of the one grows, and the bar of the three shrinks.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('backward', '2 · Backward, from the surprise to the weights', `
    <p>Now TorC runs the backward pass, the ${term('backpropagation')} of ${a('backprop', 'the lesson on every gradient at once')}. For the score of the right digit, the gradient is its probability minus one. For every other score, it is the probability of that score.</p>
    <p>Then the ${term('chain-rule', 'chain rule')} carries the gradients back through the layers. A weight gets its input times the gradient that arrives at its neuron.</p>
    ${worked(`<p>One weight goes from the most active neuron of the second layer to the score of the one. That neuron has the value ${T(W(TRUE, 'path.unit_post'), 'worked')}. The score of the one has the gradient ${T(W(TRUE, 'path.score_grad'), 'worked')}.</p>
    <p>gradient of the weight = ${T(W(TRUE, 'path.unit_post'), 'worked')} × (${T(W(TRUE, 'path.score_grad'), 'worked')}) = <strong>${T(W(TRUE, 'grad'), 'worked')}</strong></p>`)}`),
    },
    {
      kind: 'quiz', id: 'rival', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: W(RIVAL, 'grad'), tolerance: 0.01,
      prompt: (D, T) => `<p>The same neuron also has a weight to the score of the three. That score has the gradient ${T(W(RIVAL, 'path.score_grad'), 'setup')}. What is the gradient of this weight? A calculator is fine.</p>`,
      reveal: (D, T) => `<p>Multiply the value of the neuron by the gradient of the score. The result is ${T(W(RIVAL, 'grad'), 'reveal')}. It is positive, so the step makes this weight smaller.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('deep', '3 · Deep inside the network', `
    <p>The gradients also reach the first layer. Positions count from zero here.</p>
    ${worked(`<p>Take the weight from the brightest pixel, in row ${T(W(FIRST, 'path.pixel_row'), 'worked')} and column ${T(W(FIRST, 'path.pixel_col'), 'worked')}, to neuron ${T(W(FIRST, 'path.unit'), 'worked')} of the first layer. The pixel has the value ${T(W(FIRST, 'path.pixel_value'), 'worked')}, and the gradient at the neuron is ${T(W(FIRST, 'path.unit_grad'), 'worked')}.</p>
    <p>gradient of the weight = ${T(W(FIRST, 'path.pixel_value'), 'worked')} × ${T(W(FIRST, 'path.unit_grad'), 'worked')} = <strong>${T(W(FIRST, 'grad'), 'worked')}</strong></p>`)}
    <p>A black pixel has the value zero, so its weights get the gradient zero. A neuron that ReLU sets to zero passes no gradient back. So the step changes only ${T.human(`${F}.changed_parameters`, 'pct', { of: `${F}.setup.parameters` })} of the ${T.anchor('mlp_parameters')} ${term('parameter', 'parameters')}.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '4 · The step', `
    <p>Plain gradient descent replaces each parameter <code>w</code> by <code>w</code> − lr × gradient. The size of the step is lr × gradient.</p>
    ${worked(`<p>For the weight to the one, lr × gradient = ${T(`${F}.setup.lr`, 'worked')} × (${T(W(TRUE, 'grad'), 'worked')}) = ${T(W(TRUE, 'lr_times_grad'), 'worked')}.</p>
    <p>new <code>w</code> = ${T(W(TRUE, 'before'), 'worked')} − (${T(W(TRUE, 'lr_times_grad'), 'worked')}) = <strong>${T(W(TRUE, 'after'), 'worked')}</strong></p>`)}
    <p>Subtracting a negative number adds, so this weight grows. The score of the one gets a little larger.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('rival', '5 · More than the right answer', `
    <p>Many people think that training only pushes the right answer up. The step also pushes the wrong answers down. The weight to the three had a positive gradient, so it got smaller.</p>
    <p>Then the same image goes through the network again. The probability of the one rises from ${p3(T, `${F}.forward.p_true`)} to ${p3(T, `${F}.after.p_true`)}. The probability of the three falls from ${p3(T, `${F}.forward.p_rival`)} to ${p3(T, `${F}.after.p_rival`)}.</p>
    <p>Each change is tiny, but each one lowers the surprise for this image. Thousands of them together move the answer far.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('repeat', `6 · Now repeat this ${T(`${F}.full_run.steps`, 'setup')} times`, `
    <p>This one step used every part of the course:</p>
    <ul>
    <li>The loop of data, ${term('model')}, loss and adjust step comes from ${a('what-is-learning', 'the first lesson')}.</li>
    <li>The gradient and the step against it come from ${a('why-gradients', 'the gradient')} and ${a('sgd-step', 'gradient descent')}.</li>
    <li>Neurons, ReLU, softmax and the backward pass come from ${a('linear-layer', 'the neuron and the layer')} to ${a('backprop', 'the backward pass')}.</li>
    <li>Batches and ${term('epoch', 'epochs')} come from ${a('training-loop', 'the training loop')}, and the test on new data from ${a('generalization', 'an earlier lesson')}.</li>
    <li>The trained network comes from ${a('mnist-mlp', 'the last lesson')}.</li>
    </ul>
    <p>The real training repeated a step like this for every batch of every epoch. It started from random weights, as in ${a('deep-nets', 'deep networks')}, and AdamW was its ${term('optimizer')}, from ${a('optimizers', 'the lesson after the training loop')}.</p>`),
    },
    {
      kind: 'quiz', id: 'sure', purpose: 'transfer', type: 'choice',
      prompt: '<p>Another training image shows a one, and the network is almost sure of it. The one gets a probability near one. What gradient does the score of the one get?</p>',
      options: [
        { html: 'A gradient near zero, so the step changes little', correct: true, feedback: 'Yes. The probability minus one is near zero.' },
        { html: 'A gradient as large as for our image, because the step size is fixed', feedback: 'The learning rate is fixed, but the gradient depends on how sure the network is.' },
        { html: 'A large positive gradient, because the network is very sure', feedback: 'For the right digit the gradient is the probability minus one, so it is never positive.' },
      ],
      reveal: '<p>An image that the network reads surely and correctly hardly changes the network. The unsure images teach it most.</p>',
    },
    {
      kind: 'quiz', id: 'surprise', purpose: 'cumulative', from: 'losses', type: 'numeric',
      answer: `${F}.forward.loss`, tolerance: 0.01,
      prompt: (D, T) => `<p>Before the step, the one has the probability ${T(`${F}.forward.p_true`, 'setup')}. What is the surprise, the cross-entropy of this image? Use ln.</p>`,
      reveal: (D, T) => `<p>−ln(${T(`${F}.forward.p_true`, 'reveal')}) = ${T(`${F}.forward.loss`, 'reveal')}. After the step it is ${T(`${F}.after.loss`, 'reveal')}.</p>`,
    },
    {
      kind: 'explain', id: 'why-rise',
      prompt: '<p>Explain why one step on this image makes the one more probable.</p>',
      model: '<p>Each parameter moves a little against its gradient, so the score of the one rises and the other scores fall.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>The pixels go forward to ten scores, probabilities and a surprise. The backward pass sends a gradient to every parameter, and each weight gets its input times the gradient at its neuron.</p>
    <p>The step moves each parameter a little against its gradient. Each tiny change lowers the surprise for this image, so together they make the right digit clearly more probable.</p>`,
      next: () => '<p>The first layer gives every pixel its own weights. Can a network use one small pattern of weights all over the image?</p>',
    },
  ],
};
