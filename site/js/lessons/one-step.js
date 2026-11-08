// Lesson "one-step" (Part 5 `images`, curriculum v3, template 3), title "One training step, followed end to end" (brief §6
// "one-step"). One real training step of the trained digit network on one training image of the digit one (data/one-step.json,
// tools/trace/trace_one_step.cpp): the forward pass to the probabilities and the loss, the backward pass to the scores, through
// the hidden neurons (chain: a hidden neuron's gradient is the sum of its weights times the gradients after it) to four named
// parameters, plain SGD with lr 0.001 on a batch of one image so that new = old − lr × gradient holds exactly (the step size is
// lr_times_grad), and the same image again. Every number causes the next. The capstone maps each part of the step to its lesson,
// keeps what the real run added apart, and ends with the steps of the real run (full_run.steps). Every number comes through a
// role-tagged T() (docs/authoring.md "Number roles"). Tests: site/test/one-step.test.mjs. Notes: docs/lessons/one-step.md.
import { worked, term } from '../core/blocks.js';
import { mountOneStep } from '../widgets/one-step-widgets.js';

const F = 'files.one-step';
const W = (i, rest) => `${F}.weights.${i}.${rest}`;
// the named parameters by index (the test pins their ids): the first-layer weight, the output weights to the right digit and to
// the rival, the bias of the right digit
export const FIRST = 0, TRUE = 1, RIVAL = 2;
const sec = (id, title, body) => `<section class="os1" aria-labelledby="os1-${id}"><h2 id="os1-${id}">${title}</h2>${body}</section>`;
const a = (id, text) => `<a href="../${id}/">${text}</a>`;
const p4 = (T, path) => T(path, 'contrast', { sig: 4 });
// the three largest parts of the first-layer neuron's gradient, as trace paths (the trace ranks them)
const topParts = (D) => D.files['one-step'].chain.first.ranked.slice(0, 3).map((k) => `${F}.chain.first.terms.${k}`);

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
        `How does the ${term('loss')} at the end reach a ${term('weight')} at the very start?`,
        'How much does one small step change the answer of the network?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('setup', 'One image, one step', `
    <p>Take the trained digit network of the ${a('mnist-mlp', 'last lesson')}. You give it one more ${term('training')} step on a training image of the digit one.</p>
    <p>The real training used AdamW and a ${term('batch')} of ${T(`${F}.full_run.batch_size`, 'setup')} images per step. This step uses plain ${term('gradient-descent', 'gradient descent')} on one image, with the ${term('learning-rate', 'learning rate')} ${T(`${F}.setup.lr`, 'setup')}. We chose both, so that you can check every number.</p>
    <p>Before the step, the network reads the image as a one. It gives the one the ${term('probability')} ${T(`${F}.forward.p_true`, 'setup')}.</p>`),
    },
    {
      kind: 'quiz', id: 'after', purpose: 'predict', type: 'choice',
      prompt: '<p>The step changes many numbers of the network, each by a tiny amount. Then the same image goes through again. What happens to the probability of the one?</p>',
      options: [
        { html: 'It stays almost the same, because each change is tiny.', feedback: 'This belief forgets that thousands of numbers change together.' },
        { html: 'It rises clearly, but it stays below one.', correct: true, feedback: 'Yes.' },
        { html: 'It jumps to one, because the step removes the mistake.', feedback: `A step moves each number a little way against its ${term('gradient')}. It does not solve for the right answer.` },
      ],
      reveal: (D, T) => `<p>For this image and our learning rate, it rises to ${T(`${F}.after.p_true`, 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('forward', '1 · Forward, from pixels to the loss', `
    <p>The ${T.anchor('pixels')} grey values of the image go into the first ${term('layer')}. Each ${term('neuron')} computes a weighted sum of all ${term('pixel', 'pixels')} plus a ${term('bias')}. ${term('relu', 'ReLU')} keeps the positive sums and sets the others to zero, as in ${a('nonlinearity', 'an earlier lesson')}.</p>
    <p>The second layer does the same with these values. The last layer gives ten scores, one for each digit. ${term('softmax', 'Softmax')} turns them into probabilities. The loss is the surprise, the ${term('cross-entropy', 'cross-entropy')}, which is −ln of the probability of the right digit.</p>
    <p>The one gets the largest probability, and a three comes second.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('backward', '2 · Backward, from the loss to the last weights', `
    <p>Now TorC runs the backward pass, the ${term('backpropagation')} of ${a('backprop', 'the lesson on every gradient at once')}. For one image, the gradient of each score is its probability minus its ${term('target')}. The target is one for the right digit and zero for the others.</p>
    <p>A weight gets its input times the gradient that arrives at the end of the weight.</p>
    ${worked(`<p>The gradient of the score of the one is ${T(`${F}.forward.p_true`, 'worked')} minus one = ${T(W(TRUE, 'path.score_grad'), 'worked')}.</p>
    <p>One weight goes from the most active neuron of the second layer to this score. That neuron has the value ${T(W(TRUE, 'path.unit_post'), 'worked')}.</p>
    <p>gradient of the weight = ${T(W(TRUE, 'path.unit_post'), 'worked')} × (${T(W(TRUE, 'path.score_grad'), 'worked')}) = <strong>${T(W(TRUE, 'grad'), 'worked')}</strong></p>`)}`),
    },
    {
      kind: 'quiz', id: 'rival', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: W(RIVAL, 'after'), tolerance: 0.0003,
      prompt: (D, T) => `<p>The same neuron also has a weight to the score of the three. The score of the three has the gradient ${T(W(RIVAL, 'path.score_grad'), 'setup')}. Before the step, the weight has the value ${T(W(RIVAL, 'before'), 'setup')}.</p>
    <p>Find its gradient, then take the step new = old − ${T(`${F}.setup.lr`, 'setup')} × gradient. What is the new value? A calculator is fine.</p>`,
      reveal: (D, T) => `<p>The gradient is ${T(W(RIVAL, 'grad'), 'reveal')}, so the new value is ${T(W(RIVAL, 'after'), 'reveal')}. The gradient is positive, so the weight gets smaller.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => { const [x, y, z] = topParts(D); return sec('deep', '3 · Back to the first layer', `
    <p>A hidden neuron passes the gradient on. It adds up its weight to each later neuron times the gradient of that neuron. If ReLU let its sum through, this total is its gradient. This is the ${term('chain-rule', 'chain rule')} again. Positions count from zero here.</p>
    ${worked(`<p>Neuron ${T(`${F}.chain.first.unit`, 'worked')} of the first layer is on. From the second layer, its three largest parts are ${T(`${x}.weight`, 'worked')} × ${T(`${x}.grad`, 'worked')}, ${T(`${y}.weight`, 'worked')} × ${T(`${y}.grad`, 'worked')} and ${T(`${z}.weight`, 'worked')} × (${T(`${z}.grad`, 'worked')}). The other parts add up to ${T(`${F}.chain.first.rest_after_3`, 'worked')}.</p>
    <p>gradient of the neuron = ${T(`${x}.product`, 'worked')} + ${T(`${y}.product`, 'worked')} + ${T(`${z}.product`, 'worked')} + ${T(`${F}.chain.first.rest_after_3`, 'worked')} = ${T(`${F}.chain.first.grad`, 'worked')}</p>
    <p>One of the brightest pixels has the value ${T(W(FIRST, 'path.pixel_value'), 'worked')}. Its weight to this neuron gets ${T(W(FIRST, 'path.pixel_value'), 'worked')} × ${T(W(FIRST, 'path.unit_grad'), 'worked')} = <strong>${T(W(FIRST, 'grad'), 'worked')}</strong>.</p>`)}
    <p>The gradients of the second layer came the same way from the ten scores.</p>`); },
    },
    {
      kind: 'widget', id: 'walk', className: 'widget', label: 'Interactive: one training step, stage by stage',
      mount: (host, D) => mountOneStep(host, D),
      tasks: () => [
        { do: 'Press Next until the probabilities appear. Compare the bar of the one with the others.',
          see: 'The one has the tallest bar. Most of the rest goes to a three.' },
        { do: 'Press Next through the backward pass.',
          see: 'The gradients move from the scores to the second layer, then to the first. Neurons that are off get zero.' },
        { do: 'Press Next to the end.',
          see: 'The bar of the one grows, and the bar of the three shrinks.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('step', '4 · The step', `
    <p>Plain gradient descent replaces each ${term('parameter')} <code>w</code> by <code>w</code> − lr × gradient.</p>
    <p>Before the step, the weight to the score of the one has the value ${T(W(TRUE, 'before'), 'setup')}.</p>
    ${worked(`<p>lr × gradient = ${T(`${F}.setup.lr`, 'worked')} × (${T(W(TRUE, 'grad'), 'worked')}) = ${T(W(TRUE, 'lr_times_grad'), 'worked')}</p>
    <p>new <code>w</code> = ${T(W(TRUE, 'before'), 'worked')} − (${T(W(TRUE, 'lr_times_grad'), 'worked')}) = <strong>${T(W(TRUE, 'after'), 'worked')}</strong></p>`)}
    <p>Subtracting a negative number adds, so this weight grows.</p>
    <p>A black pixel has the value zero, so its weights get the gradient zero. A neuron that ReLU turned off passes no gradient back. A few changes are too small to store. So the step changes only about ${T.human(`${F}.changed_parameters`, 'pct', { of: `${F}.setup.parameters` })} of the ${T.anchor('mlp_parameters')} parameters. This is also why the edges of the weight images in the last lesson look speckled.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('rival', '5 · More than the right answer', `
    <p>Many people think that training only pushes the right answer up. But the score of the three had a positive gradient. So its weight from the most active neuron got smaller, and its score fell.</p>
    <p>A few other wrong scores rose a little. But the score of the one rose much more, so the probability of every wrong digit fell. The probability of the one rose from ${p4(T, `${F}.forward.p_true`)} to ${p4(T, `${F}.after.p_true`)}.</p>
    <p>A step that helps one image can hurt others. So the real training averages the gradient over a batch.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('repeat', `6 · Now repeat this ${T(`${F}.full_run.steps`, 'setup')} times`, `
    <p>Each part of this step comes from an earlier lesson:</p>
    <ul>
    <li>The loop of data, ${term('model')}, loss and adjust step comes from ${a('what-is-learning', 'the first lesson')}.</li>
    <li>The layers and neurons come from ${a('linear-layer', 'the neuron and the layer')}, ReLU from ${a('nonlinearity', 'the lesson on curves')}, and softmax from ${a('losses', 'the lesson on the loss')}.</li>
    <li>The gradients come from ${a('why-gradients', 'the gradient')} and ${a('backprop', 'the backward pass')}.</li>
    <li>The step against the gradient comes from ${a('sgd-step', 'gradient descent')}.</li>
    </ul>
    <p>The real run started from random weights, as in ${a('deep-nets', 'deep networks')}. It used batches and ${term('epoch', 'epochs')} from ${a('training-loop', 'the training loop')}, and the ${term('optimizer')} AdamW from ${a('optimizers', 'the lesson after it')}. A separate run with a ${term('validation-set', 'validation set')}, as in ${a('generalization', 'an earlier lesson')}, showed that this training ${term('overfitting', 'overfits')} only a little.</p>
    <p>One epoch took ${T(`${F}.full_run.steps_per_epoch`, 'setup')} steps, and the run had ${T(`${F}.full_run.epochs`, 'setup')} epochs. So it took a step like this one ${T(`${F}.full_run.steps`, 'setup')} times.</p>`),
    },
    {
      kind: 'quiz', id: 'sure', purpose: 'transfer', type: 'choice',
      prompt: '<p>Another training image shows a one, and the network is almost sure of it. The one gets a probability near one. What gradient does the score of the one get?</p>',
      options: [
        { html: 'Near zero, so the step changes little', correct: true, feedback: 'Yes.' },
        { html: 'Near minus one, the strongest push, because the network is sure', feedback: 'This belief says that a sure answer earns a strong push. A sure right answer gets almost none.' },
        { html: 'A large positive gradient', feedback: 'The probability minus one is never positive.' },
      ],
      reveal: '<p>The gradient is the probability minus one. It is largest in size when the right digit gets a probability near zero. So a wrong answer changes the network most, and a sure right answer least.</p>',
    },
    {
      kind: 'quiz', id: 'surprise', purpose: 'cumulative', from: 'losses', type: 'numeric',
      answer: `${F}.forward.loss`, tolerance: 0.01,
      prompt: (D, T) => `<p>Before the step, the one has the probability ${T(`${F}.forward.p_true`, 'setup')}. What is the loss, the cross-entropy of this image? Use ln.</p>`,
      reveal: (D, T) => `<p>−ln(${T(`${F}.forward.p_true`, 'reveal')}) = ${T(`${F}.forward.loss`, 'reveal')}. After the step it is ${T(`${F}.after.loss`, 'reveal')}.</p>`,
    },
    {
      kind: 'explain', id: 'why-rise',
      prompt: '<p>Explain why one step on this image makes the one more probable.</p>',
      model: '<p>Each parameter moves a little against its gradient, and together the changes raise the score of the one far more than any other score.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>The pixels go forward to ten scores, probabilities and the loss. The backward pass sends a gradient to every parameter, and each neuron adds up its weights times the gradients after it.</p>
    <p>The step moves each parameter a little against its gradient. Together the changes make the right digit clearly more probable.</p>`,
      next: () => '<p>The first layer gives every pixel its own weights. Can a network use one small pattern of weights all over the image?</p>',
    },
  ],
};
