// Lesson "nonlinearity" (Part 3 `networks`, curriculum v3, template 3): "Activation functions". Two linear layers act like one,
// first with plain numbers (f(x) = 2x + 1, g(x) = 3x + 4, g(f(x)) = 6x + 7, then ReLU between them breaks this), then for layers;
// the no-activation model fitted to t = x² ends as the flat line at the mean (the committed predict, without the old hint); ReLU
// and the hinge builder (each hidden ReLU neuron is a hinge, their sum bends); sigmoid and the consequence of its small gradient.
// Brief §6 Part 2 and §7. Softmax moved to `losses`; the overflow fact left the page. Every number comes from
// data/nonlinearity.json (`collapse`, `models`, `hinges`, `activations`, `setup`) through a role-tagged T(); formulas take their
// constants from the trace with T.num.
import { math } from '../core/katex.js';
import { worked, analogy, term, sym } from '../core/blocks.js';
import { mountActivationPlot, mountCurveFit } from '../widgets/nonlinear-widgets.js';
import { mountHingeBuilder } from '../widgets/hinge-builder.js';

const F = 'files.nonlinearity';
const C = `${F}.collapse`;
const LIN = `${F}.models.0`, NET = `${F}.models.1`;
const END = (m) => `${m}.checkpoints.3`;                 // after the last recorded step (400)
const SIG = `${F}.activations.1`;
const Z = { zero: 40, four: 80 };                        // indices into the grid z = -4, -3.9, …, 4
const X0 = 22;                                           // index of x = 0 in the prediction grid
const XM2 = 0, X2 = 4;                                   // indices of x = -2 and x = 2 in collapse.x
const UNIT = (k) => `${F}.hinges.units.${k}.unit`;

const sec = (id, title, body) => `<section class="nlx" aria-labelledby="nl-${id}"><h2 id="nl-${id}">${title}</h2>${body}</section>`;
const eq = (latex, label = 'Formula') => `<div class="nl-eq" tabindex="0" role="group" aria-label="${label}">${math(latex, true)}</div>`;

const STYLE = `<style>
.nlx p{max-width:44rem}
.nlx h2{margin-top:2rem}
.nl-eq{overflow-x:auto;margin:.2rem 0}
.nl-eq .katex-display{margin:.4em 0}
</style>`;

export default {
  id: 'nonlinearity',
  title: 'Activation functions',
  trace: ['nonlinearity.json'],
  template: 3,
  loop: 'model',
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Activation functions</h1>
    <p class="lead">A small function between two ${term('layer', 'layers')} lets a network draw curves.</p></section>`,
    },
    {
      kind: 'questions',
      items: () => [
        'You stack two layers to fit points that lie on a curve. Does the second layer let the network bend?',
        'What does a small function between the layers change, and why does it help?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('points', 'Points on a curve', `
    <p>There are ${T(`${F}.setup.points`, 'setup')} points on the curve <i>t</i> = <i>x</i>². Their <i>x</i> goes from ${T(`${F}.data.x.0`, 'setup')} to ${T(`${F}.data.x.20`, 'setup')}.</p>
    <p>The ${term('model')} has two layers. The first layer has ${T(`${F}.setup.hidden`, 'setup')} ${term('neuron', 'neurons')}. The second layer has one neuron, and its output is the ${term('prediction')}.</p>
    <p>The first layer is a ${term('hidden-layer')}, because the data never shows its outputs. Each of its outputs goes straight into the second layer.</p>
    <p>${term('training', 'Training')} runs for ${T(`${F}.setup.steps`, 'setup')} steps. Each step changes the ${term('weight', 'weights')} with the ${term('gradient')}, by an improved kind of ${term('gradient-descent')} that a later lesson explains.</p>`),
    },
    {
      kind: 'quiz', id: 'flat', purpose: 'predict', type: 'choice',
      prompt: '<p>What does this model draw after its training?</p>',
      options: [
        { html: 'A parabola that follows the points',
          feedback: 'A second layer does not bend the line by itself. Each layer only multiplies and adds.' },
        { html: 'A tilted straight line',
          feedback: 'The model does draw a straight line. But the points are the same on the left and on the right, so a tilt helps one side and hurts the other.' },
        { html: 'A flat straight line at the average height of the points', correct: true,
          feedback: 'Yes. The two layers act like one straight line, and the best straight line through these points is flat.' },
        { html: 'A line with a few corners, one for each hidden neuron',
          feedback: 'Corners need a function between the layers. Without one, the neurons only multiply and add.' },
      ],
      reveal: (D, T) => `<p>After training, the model draws a flat line at ${T(`${END(LIN)}.prediction.${X0}`, 'reveal')}. This is the average height of the points.</p>
    <p>It misses the points by about ${T.human(`${END(LIN)}.loss`, 'rmse')} on average.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => {
        const n = (k) => T.num(`${C}.${k}`);
        return sec('collapse', '1 · Two layers act like one', `
    <p>Here is the reason, first with plain numbers. Take two functions that only multiply and add:</p>
    ${eq(`f(x) = ${n('f.w')}x + ${n('f.b')} \\qquad g(x) = ${n('g.w')}x + ${n('g.b')}`, 'The two functions')}
    <p>Now feed the output of <i>f</i> into <i>g</i>.</p>
    ${worked(`<p><i>g</i>(<i>f</i>(<i>x</i>)) = ${T(`${C}.g.w`, 'worked')} × (${T(`${C}.f.w`, 'worked')}<i>x</i> + ${T(`${C}.f.b`, 'worked')}) + ${T(`${C}.g.b`, 'worked')} = <strong>${T(`${C}.combined.w`, 'worked')}<i>x</i> + ${T(`${C}.combined.b`, 'worked')}</strong></p>
    <p>At <i>x</i> = ${T(`${C}.x.${X2}`, 'worked')}, <i>f</i> gives ${T(`${C}.f_x.${X2}`, 'worked')} and <i>g</i> gives ${T(`${C}.g_f_x.${X2}`, 'worked')}. The one line gives ${T(`${C}.combined.w`, 'worked')} × ${T(`${C}.x.${X2}`, 'worked')} + ${T(`${C}.combined.b`, 'worked')} = ${T(`${C}.combined_x.${X2}`, 'worked')} too.</p>`)}
    <p>Layers do the same with many inputs. The second layer makes a weighted sum of weighted sums, and that is again one weighted sum plus a ${term('bias')}.</p>
    <p>So two layers act like one layer, and they can only draw a straight line.</p>`);
      },
    },
    {
      kind: 'text',
      html: () => sec('relu', '2 · A function between the layers', `
    <p>An ${term('activation-function')} is a function that each neuron applies to its weighted sum. Its plot is not a straight line, so the layers stop acting like one.</p>
    <p>The simplest one is ${term('relu', 'ReLU')}. It keeps a positive number and changes a negative number to zero:</p>
    ${eq('\\mathrm{relu}(z) = \\max(0,\\ z)', 'The ReLU function')}
    <p>Here <i>z</i> is the weighted sum of one neuron.</p>`),
    },
    {
      kind: 'quiz', id: 'relu-between', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${C}.g_relu_f_x.${XM2}`, tolerance: 0.01,
      prompt: (D, T) => `<p>Put ReLU between <i>f</i> and <i>g</i>. What is <i>g</i>(relu(<i>f</i>(<i>x</i>))) at <i>x</i> = ${T(`${C}.x.${XM2}`, 'setup')}?</p>`,
      reveal: (D, T) => `${worked(`<p><i>f</i>(${T(`${C}.x.${XM2}`, 'worked')}) = ${T(`${C}.f_x.${XM2}`, 'worked')}. ReLU changes it to ${T(`${C}.relu_f_x.${XM2}`, 'worked')}, and <i>g</i>(${T(`${C}.relu_f_x.${XM2}`, 'worked')}) = <strong>${T(`${C}.g_relu_f_x.${XM2}`, 'worked')}</strong>.</p>`)}
    <p>The one line gives ${T(`${C}.combined_x.${XM2}`, 'reveal')} there. With ReLU between them, the two functions no longer act like one line.</p>`,
    },
    {
      kind: 'widget', id: 'fit', className: 'widget', label: 'Interactive: two models fitted to t = x squared, with no activation and with ReLU',
      mount: (host, D) => mountCurveFit(host, D),
      tasks: (D, T) => [
        { do: 'Choose no activation, then go through the training steps from the first to the last.',
          see: `The tilted start line turns flat. It never bends, and it misses the points by about ${T.human(`${END(LIN)}.loss`, 'rmse')} on average.` },
        { do: 'Choose with ReLU and go through the steps again.',
          see: `This model bends toward the points. At the end it misses them by about ${T.human(`${END(NET)}.loss`, 'rmse')} on average.` },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('hinges', '3 · Each ReLU neuron is a hinge', `
    <p>Look at one hidden neuron with ReLU. Its output is zero on one side of a point and a straight line on the other side. We call that point its kink.</p>
    <p>The second layer multiplies each bent line by a weight and adds them to its bias. So each neuron bends the sum at its own kink. Training moves the kinks until the sum follows the points.</p>
    ${analogy('Each ReLU neuron is a hinge, and a sum of hinges bends a straight line into a curve.',
    `The sum has straight pieces only. With ${T(`${F}.setup.hidden`, 'setup')} hinges it has at most ${T(`${F}.setup.hidden`, 'setup')} bends, so it only comes close to a smooth curve.`)}
    <p>Many people think that ReLU itself is the curve. But ReLU is two straight pieces with one bend. In the widget, one neuron alone is a single hinge, and the curve appears only in the sum.</p>`),
    },
    {
      kind: 'widget', id: 'hinges', className: 'widget', label: 'Interactive: the hinge builder',
      mount: (host, D) => mountHingeBuilder(host, D),
      tasks: (D, T) => [
        { do: `Press All off, then switch on neuron ${T(UNIT(4), 'setup')} only.`,
          see: 'You see one hinge. It is zero on the left of its kink and a straight line on the right. ReLU alone makes no curve.' },
        { do: 'Switch the other neurons on, one at a time.',
          see: 'Each neuron adds a bend at its kink. With all of them on, the sum follows the points.' },
        { do: `Switch off neuron ${T(UNIT(1), 'setup')} only.`,
          see: 'The left arm of the curve drops, because this neuron gave much of the rise on the left. The right side stays the same.' },
      ],
    },
    {
      kind: 'quiz', id: 'bends', purpose: 'transfer', type: 'choice',
      prompt: '<p>A network has one hidden layer with three ReLU neurons. At most how many bends can its curve have?</p>',
      options: [
        { html: 'Three, one for each neuron', correct: true,
          feedback: 'Yes. Each hidden neuron bends the sum at its own kink.' },
        { html: 'One, because ReLU bends once',
          feedback: 'One neuron bends once. But the second layer adds three hinges with three kinks.' },
        { html: 'As many as it needs, because ReLU makes curves',
          feedback: 'ReLU is two straight pieces. The sum has straight pieces only, one more than the number of bends.' },
        { html: 'None, because each neuron is a straight line',
          feedback: 'Without ReLU this is true. With ReLU each neuron is a hinge with one bend.' },
      ],
      reveal: '<p>Three neurons give three kinks, so the sum has at most three bends. Two kinks at the same place give fewer.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('sigmoid', '4 · Sigmoid, and a gradient that gets small', `
    <p>Another activation function is the ${term('sigmoid')}, written ${sym('sigma', 'σ')}, the Greek letter sigma. It changes any number into a number between zero and one:</p>
    ${eq('\\sigma(z) = \\frac{1}{1 + e^{-z}}', 'The sigmoid function')}
    <p>Here ${sym('exp', '<i>e</i>')} is a fixed number a little below three.</p>
    <p>Training also needs the gradient of the sigmoid. It is largest at <i>z</i> = ${T(`${F}.z.${Z.zero}`, 'setup')}, where it is ${T(`${SIG}.grad.${Z.zero}`, 'contrast')}. At <i>z</i> = ${T(`${F}.z.${Z.four}`, 'setup')} it is only ${T(`${SIG}.grad.${Z.four}`, 'contrast')}.</p>
    <p>So a neuron whose weighted sum is far from zero sends back a tiny gradient. The weights that feed it then get tiny steps, so they barely learn. The lesson on ${term('backpropagation', 'backprop')} shows why.</p>
    <p>The gradient of ReLU is one for every positive weighted sum.</p>`),
    },
    {
      kind: 'widget', id: 'activation', className: 'widget', label: 'Interactive: ReLU and sigmoid with their gradients',
      mount: (host, D) => mountActivationPlot(host, D),
      tasks: (D, T) => [
        { do: `Choose Sigmoid and move <i>z</i> to ${T(`${F}.z.${Z.four}`, 'setup')}.`,
          see: 'The dashed gradient curve is almost zero there, and sigmoid is almost one.' },
      ],
    },
    {
      kind: 'quiz', id: 'tiny-step', purpose: 'cumulative', from: 'why-gradients', type: 'choice',
      prompt: (D, T) => `<p>A training step changes a weight by the ${term('learning-rate')} times its gradient. Suppose that a weight’s gradient is ${T(`${SIG}.grad.${Z.four}`, { role: 'setup', sig: 2 })} where it was ${T(`${SIG}.grad.${Z.zero}`, 'setup')}. The learning rate stays the same. What happens to the step?</p>`,
      options: [
        { html: 'It gets much smaller, so the weight barely changes', correct: true,
          feedback: 'Yes. The step is the learning rate times the gradient.' },
        { html: 'It stays the same, because the learning rate sets the step',
          feedback: 'The learning rate is only one factor. The step is the learning rate times the gradient.' },
        { html: 'It gets larger, because the model is still far off',
          feedback: 'The step does not look at how wrong the model is. It is the learning rate times the gradient.' },
      ],
      reveal: '<p>A much smaller gradient gives a much smaller step, with the same learning rate.</p>',
    },
    {
      kind: 'explain', id: 'why-activation',
      prompt: '<p>Explain why a network needs an activation function between two layers.</p>',
      model: '<p>Without it, two layers multiply out to one weighted sum, which draws only a straight line. The activation function lets each neuron bend it.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>Two layers with no activation function act like one layer, so they draw only a straight line. ReLU between the layers turns each hidden neuron into a hinge.</p>
    <p>The second layer adds the hinges, and the sum bends to follow the points. Sigmoid also bends, but its gradient is tiny far from zero, so those weights barely learn.</p>`,
      next: () => `<p>This network gives one number for each input. How does a network choose between two ${term('class', 'classes')}, such as two colours of points?</p>`,
    },
  ],
};
