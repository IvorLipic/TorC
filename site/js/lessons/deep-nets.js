// Lesson "deep-nets" (Part 3 `networks`, curriculum v3, template 3; brief §6 "deep-nets"): short. Twins (two neurons that start
// equal get equal gradients and stay equal; zero weights as the extreme case), vanishing gradients with sigmoid, and the size of
// the start with the reason for the 2 in √(2/in). The last lesson of Part 3, so it ends with a short review of the part.
// Every number comes from data/deep.json (tools/trace/trace_deep.cpp) through a role-tagged T() (docs/authoring.md "Number roles").
// Notes: docs/lessons/deep-nets.md.
import { math } from '../core/katex.js';
import { worked, term } from '../core/blocks.js';
import { mountTwins, mountVanishing, mountSignal } from '../widgets/deep-widgets.js';

const F = 'files.deep';
const TWINS = 0, ZEROS = 2;                                 // twins[]: twins, different, zeros
const KAIMING = 0, TOO_SMALL = 1, NO_TWO = 3;               // init[]: kaiming, too_small, too_large, no_two
const SIGMOID = 0, RELU = 1;                                // vanishing[]
const LAST = 4;                                             // long_run[4] is step 2000
const sec = (id, title, body) => `<section class="dn" aria-labelledby="dn-${id}"><h2 id="dn-${id}">${title}</h2>${body}</section>`;
const kaiming = () => math('\\sqrt{2 / \\text{in}}', false);

const STYLE = `<style>
.dn p{max-width:44rem}
.dn h2{margin-top:2rem}
.dn-review{margin:.3rem 0 .3rem 1.2rem;padding:0;max-width:42rem}
.dn-review li{margin:.25rem 0}
</style>`;

export default {
  id: 'deep-nets',
  title: 'Why deep networks are hard to start',
  trace: ['deep.json'],
  template: 3,
  loop: 'model',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Why deep networks are hard to start</h1>
    <p class="lead">Before ${term('training')} starts, every ${term('weight')} of a network needs a first value.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        `Two ${term('neuron', 'neurons')} start with the same weights. Does training make them different?`,
        `Why can the first ${term('layer')} of a deep network learn almost nothing?`,
        'How large should the first weights be?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: () => sec('start', 'Where training starts', `
    <p>A network with many ${term('hidden-layer', 'hidden layers')} is a <strong>deep network</strong>. The choice of the first weights is the ${term('initialization')}. TorC fills each layer with random numbers of a chosen size.</p>
    <p>Take a small network with two inputs, two neurons and one output. It must learn four points. Its output must be one when exactly one input is one, and zero otherwise.</p>`),
    },
    {
      kind: 'quiz', id: 'twins', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>The two neurons start with the same weights, the same ${term('bias')} and the same output weight. Training runs for ${T(`${F}.twins.${TWINS}.long_run.${LAST}.step`, 'setup')} steps. What happens to the two neurons?</p>`,
      options: [
        { html: 'They stay exactly equal.', correct: true, feedback: 'Yes. Every step changes them in the same way.' },
        { html: 'The four points push them apart.', feedback: 'Each point reaches both neurons through the same weights, so it pushes both in the same way.' },
        { html: 'One takes over, and the other goes to zero.', feedback: 'Nothing in a step tells the two neurons apart.' },
      ],
      reveal: (D, T) => `<p>They stay equal to the last digit after all ${T(`${F}.twins.${TWINS}.long_run.${LAST}.step`, 'reveal')} steps.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('twins', '1 · Twins stay twins', `
    <p>The two neurons get the same inputs and have the same weights, so they compute the same output. They have the same output weight, so the backward pass gives them the same ${term('gradient')}. A weight's gradient is its neuron's gradient times its input, so matching weights get equal gradients. The step changes both neurons in the same way, and they are equal again.</p>
    <p>So the layer acts like one neuron forever. With the equal start, the network keeps a typical miss of about ${T.human(`${F}.twins.${TWINS}.long_run.${LAST}.loss`, 'rmse')}, the square root of the mean squared miss. With different starts, it learns all four points.</p>`),
    },
    {
      kind: 'widget', id: 'twins-widget', className: 'widget', label: 'Interactive: two neurons from three starts',
      mount: (host, D) => mountTwins(host, D),
      tasks: (D, T) => [
        { do: `Choose the equal start, then step to ${T(`${F}.twins.${TWINS}.long_run.${LAST}.step`, 'setup')}.`,
          see: 'The two neurons keep the same bars, and the typical miss stops falling.' },
        { do: 'Choose the different start and step again.',
          see: `The neurons move apart, and the ${term('prediction', 'predictions')} reach the ${term('target', 'targets')}.` },
        { do: 'Choose all zero.', see: 'No weight moves, and the network predicts the same number for every point.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('zero', '2 · A tempting idea that is wrong', `
    <p>Many people start every weight at zero, because zero looks neutral. But zero makes every neuron a twin of every other.</p>
    <p>It is even worse. With zero output weights, each neuron gets a gradient of zero, so no weight ever changes. In TorC's run only the output bias learns, and the network predicts ${T(`${F}.twins.${ZEROS}.long_run.${LAST}.prediction.0`, 'contrast')} for every point. So the start must be random.</p>`),
    },
    {
      kind: 'quiz', id: 'deepest', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>A stack has ${T(`${F}.setup.depth`, 'setup')} hidden layers of ${T(`${F}.setup.width`, 'setup')} neurons, each with a ${term('sigmoid')}. After one backward pass, which layer has the largest gradients?</p>`,
      options: [
        { html: 'The first layer, next to the input', feedback: 'The gradient starts at the ' + term('loss') + ' and passes the other layers first.' },
        { html: 'The last hidden layer, next to the output', correct: true, feedback: 'Yes. The gradient reaches it before it passes the other layers.' },
        { html: 'All layers about the same, because they have the same size', feedback: 'Each layer on the way back multiplies the gradient again.' },
      ],
      reveal: (D, T) => `<p>The last hidden layer gets about ${T(`${F}.vanishing.${SIGMOID}.derived.last_over_first_hidden`, { role: 'reveal', sig: 2 })} times more than the first.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('vanish', '3 · Gradients that fade', `
    <p>The backward pass multiplies one ratio per layer, by the ${term('chain-rule')}. The sigmoid is never steeper than ${T(`${F}.setup.sigmoid_max_slope`, 'setup')}, so in this stack each sigmoid layer shrank the gradient.</p>
    <p>In the stack, the first layer's gradient is about ${T(`${F}.vanishing.${SIGMOID}.derived.last_over_first_hidden`, 'contrast')} times smaller than the last layer's. With ${term('relu', 'ReLU')} it is about ${T(`${F}.vanishing.${RELU}.derived.last_over_first_hidden`, 'contrast')} times smaller. ReLU has the slope one for every positive input, so it does not shrink the gradient by itself.</p>
    <p>These are ${term('vanishing-gradients')}. A layer with almost no gradient barely learns, so more layers do not always learn better. These numbers come from one random start, and another start gives slightly different ones.</p>`),
    },
    {
      kind: 'widget', id: 'vanishing', className: 'widget', label: 'Interactive: the gradient of each layer, sigmoid and ReLU',
      mount: (host, D) => mountVanishing(host, D),
      tasks: () => [{
        do: 'Compare the first and the eighth layer, for sigmoid and then for ReLU.',
        see: 'The sigmoid bars grow at every layer toward the output. The ReLU bars stay at about the same height.',
      }],
    },
    {
      kind: 'text',
      html: (D, T) => sec('size', '4 · How large the start should be', `
    <p>The spread of some numbers is their standard deviation, a typical distance from their mean. Each neuron adds up its inputs times its weights. With weights that are too small, the spread of the signal shrinks at every layer. With weights that are too large, it grows. The gradients then shrink or grow too.</p>
    <p>TorC starts the weights of a layer with the spread ${kaiming()}, where <code>in</code> is the number of inputs. Without the 2, each sum would keep the spread of its inputs. But ReLU sets about ${T.human(`${F}.init.${KAIMING}.layers.1.frac_zero`, 'pct')} of the values to zero, so the signal would weaken at every layer. The 2 makes up for that lost half.</p>`),
    },
    {
      kind: 'widget', id: 'signal', className: 'widget', label: 'Interactive: the spread of the signal through eight layers',
      mount: (host, D) => mountSignal(host, D),
      tasks: () => [
        { do: 'Look at the four lines.', see: 'Only TorC’s start stays level. The too small start falls toward zero, and the too large start climbs.' },
        { do: 'Turn off the too small and the too large start.', see: 'Without the 2, the spread falls at every layer.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('worked', '5 · The spread for one layer', `
    <p>After eight layers, the start without the 2 has a spread of about ${T(`${F}.init.${NO_TWO}.layers.8.std`, 'contrast')}, and TorC’s start keeps about ${T(`${F}.init.${KAIMING}.layers.8.std`, 'contrast')}.</p>
    ${worked(`<p>Each layer of the stack has ${T(`${F}.setup.width`, 'worked')} inputs. So TorC starts its weights with the spread ${math(`\\sqrt{2 / ${T.num(`${F}.setup.width`)}}`, false)} = <strong>${T(`${F}.setup.kaiming_std`, 'worked')}</strong>.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'digit-start', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${F}.setup.kaiming_digit.std`, tolerance: 0.001,
      prompt: (D, T) => `<p>The first layer of the digit network has ${T(`${F}.setup.kaiming_digit.inputs`, 'setup')} inputs. What spread does TorC give its start weights? A calculator helps.</p>`,
      reveal: (D, T) => `<p>The square root of two divided by ${T(`${F}.setup.kaiming_digit.inputs`, 'reveal')} is about ${T(`${F}.setup.kaiming_digit.std`, 'reveal')}. A neuron with more inputs adds more products, so each weight must be smaller.</p>`,
    },
    {
      kind: 'quiz', id: 'twenty', purpose: 'transfer', type: 'choice',
      prompt: (D, T) => `<p>A friend builds a ReLU network with twenty layers. Every weight starts with the spread ${T(`${F}.init.${TOO_SMALL}.init_std`, 'setup')}, whatever the number of inputs. What happens to the signal?</p>`,
      options: [
        { html: 'It shrinks toward zero, and the first layers get almost no gradient.', correct: true, feedback: 'Yes. Small weights shrink the spread at every layer, and twenty layers repeat it.' },
        { html: 'It keeps its size, because ReLU keeps positive numbers.', feedback: 'ReLU keeps them, but the small weights of the next layer shrink the sum again.' },
        { html: 'It grows, because the network has many layers.', feedback: 'Small weights make each sum smaller, and many layers repeat that.' },
      ],
      reveal: (D, T) => `<p>In this lesson's stack, the start ${T(`${F}.init.${TOO_SMALL}.init_std`, 'reveal')} made the spread fall at every layer.</p>`,
    },
    {
      kind: 'quiz', id: 'tiny-gradient', purpose: 'cumulative', from: 'why-gradients', type: 'choice',
      prompt: '<p>A weight in the first layer has a gradient of almost zero. What does one training step do to it?</p>',
      options: [
        { html: 'It barely changes the weight.', correct: true, feedback: `Yes. The step is the ${term('learning-rate')} times the gradient, which is almost zero.` },
        { html: 'It sets the weight to zero.', feedback: 'A step subtracts the learning rate times the gradient. It does not set a value.' },
        { html: 'It makes a large step, because the loss is still high.', feedback: 'The size of a step follows the gradient, not the loss.' },
      ],
      reveal: '<p>So with vanishing gradients, the first layers almost stop learning.</p>',
    },
    {
      kind: 'explain', id: 'why-twins',
      prompt: '<p>Explain why two neurons that start with the same weights never become different.</p>',
      model: '<p>They compute the same output and get the same gradient, so every step changes them in the same way.</p>',
    },
    {
      kind: 'text',
      html: () => sec('review', 'Part 3 in review', `
    <p>This lesson ends the part about networks. If one of these questions is hard, go back to its lesson.</p>
    <ul class="dn-review">
      <li>What does a layer compute from its inputs? (<a href="../linear-layer/">layers</a>)</li>
      <li>Why does a network need an ${term('activation-function')} between its layers? (<a href="../nonlinearity/">nonlinearity</a>)</li>
      <li>How does ${term('backpropagation')} find every gradient in one backward pass? (<a href="../backprop/">backpropagation</a>)</li>
    </ul>`),
    },
    {
      kind: 'answer',
      html: () => `<p>No. Two neurons that start equal get equal gradients, so they stay twins, and the start must be random.</p>
    <p>With sigmoid, the backward pass multiplies small ratios, so the first layers of a deep stack get almost no gradient. ReLU and the start spread ${kaiming()} keep the signal and the gradient steady.</p>`,
      next: () => '<p>Every part of a training step is now in place. How does a program repeat it over thousands of examples?</p>',
    },
  ],
};
