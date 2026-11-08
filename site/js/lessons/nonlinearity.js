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
        'What does a small function between the layers change?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: (D, T) => sec('points', 'Points on a curve', `
    <p>There are ${T(`${F}.setup.points`, 'setup')} points on the curve <i>t</i> = <i>x</i>². Here <i>t</i> is the ${term('target')}, the value to predict. Their <i>x</i> goes from ${T(`${F}.data.x.0`, 'setup')} to ${T(`${F}.data.x.20`, 'setup')}.</p>
    <p>The ${term('model')} has two layers. The first layer has ${T(`${F}.setup.hidden`, 'setup')} ${term('neuron', 'neurons')}. The second layer has one neuron, and its output is the ${term('prediction')}. Nothing sits between the layers, so each layer only multiplies and adds, as in the last lesson.</p>
    <p>The first layer is a ${term('hidden-layer')}, because its outputs are neither the inputs nor the prediction.</p>
    <p>${term('training', 'Training')} runs for ${T(`${F}.setup.steps`, 'setup')} steps. Each step changes the ${term('parameter', 'parameters')} with the ${term('gradient')}, by an improved kind of ${term('gradient-descent')} that a later lesson explains.</p>`),
    },
    {
      kind: 'quiz', id: 'flat', purpose: 'predict', type: 'choice',
      prompt: '<p>What does this model draw after its training?</p>',
      options: [
        { html: 'A parabola that follows the points',
          feedback: 'A second layer does not bend the line by itself. Each layer only multiplies and adds.' },
        { html: 'A tilted straight line',
          feedback: 'It is a straight line, but a tilt helps one side and hurts the other.' },
        { html: 'A flat straight line at the average height of the points', correct: true,
          feedback: 'Yes. The two layers act like one straight line, and the best one here is flat.' },
        { html: 'A line with a few corners, one for each hidden neuron',
          feedback: 'Corners need a function between the layers. Here nothing sits there.' },
      ],
      reveal: (D, T) => `<p>After training, the model draws a flat line at ${T(`${END(LIN)}.prediction.${X0}`, 'reveal')}, the average height of the points.</p>
    <p>Its typical miss, the square root of the ${term('mean-squared-error', 'mean squared error')}, is about ${T.human(`${END(LIN)}.loss`, 'rmse')}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => {
        const n = (k) => T.num(`${C}.${k}`);
        return sec('collapse', '1 · Two layers act like one', `
    <p>Here is the reason, first with plain numbers. Take two functions that only multiply and add:</p>
    ${eq(`f(x) = ${n('f.w')}x + ${n('f.b')} \\qquad g(x) = ${n('g.w')}x + ${n('g.b')}`, 'The two functions')}
    <p>Now feed the output of <i>f</i> into <i>g</i>.</p>
    ${worked(`<p><i>g</i>(<i>f</i>(<i>x</i>)) = ${T(`${C}.g.w`, 'worked')} × (${T(`${C}.f.w`, 'worked')}<i>x</i> + ${T(`${C}.f.b`, 'worked')}) + ${T(`${C}.g.b`, 'worked')} = <strong>${T(`${C}.derived.w`, 'worked')}<i>x</i> + ${T(`${C}.derived.b`, 'worked')}</strong></p>
    <p>At <i>x</i> = ${T(`${C}.x.${X2}`, 'worked')}, <i>f</i> gives ${T(`${C}.f_x.${X2}`, 'worked')} and <i>g</i> gives ${T(`${C}.g_f_x.${X2}`, 'worked')}. The one line gives ${T(`${C}.derived.w`, 'worked')} × ${T(`${C}.x.${X2}`, 'worked')} + ${T(`${C}.derived.b`, 'worked')} = ${T(`${C}.combined_x.${X2}`, 'worked')} too.</p>`)}
    <p>Layers do the same with many inputs. The second layer makes a weighted sum of weighted sums, which is again one weighted sum plus a ${term('bias')}. So two layers act like one layer and can only draw a straight line.</p>`);
      },
    },
    {
      kind: 'text',
      html: () => sec('relu', '2 · A function between the layers', `
    <p>An ${term('activation-function')} is a function that each neuron applies to its weighted sum. So a hidden neuron now does one more thing than in the last lesson. The plot of this function is not a straight line, so the layers stop acting like one.</p>
    <p>The simplest one is ${term('relu', 'ReLU')}. It keeps a positive number and changes a negative number to zero:</p>
    ${eq('\\mathrm{relu}(z) = \\max(0,\\ z)', 'The ReLU function')}
    <p>Here <i>z</i> is the weighted sum of one neuron, and max gives the larger of zero and <i>z</i>.</p>`),
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
        { do: 'With no activation, go through the training steps.',
          see: `The tilted start line turns flat. It never bends, and its typical miss ends near ${T.human(`${END(LIN)}.loss`, 'rmse')}.` },
        { do: 'Choose with ReLU and go through the steps again.',
          see: `This model bends toward the points. In this one training run, its typical miss ends near ${T.human(`${END(NET)}.loss`, 'rmse')}.` },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('hinges', '3 · Each ReLU neuron is a hinge', `
    <p>One hidden neuron with ReLU gives zero on one side of a point and a straight line on the other side. This is a hinge, and the point is its bend.</p>
    <p>The second layer multiplies each hinge by a ${term('weight')}, adds them, and adds its own bias. So each neuron can bend the sum at its own point. A neuron with a small weight in the second layer changes the sum only a little.</p>
    ${analogy('A sum of hinges bends a straight line into a curve.',
    `The sum has straight pieces only. With ${T(`${F}.setup.hidden`, 'setup')} hinges it has at most ${T(`${F}.setup.hidden`, 'setup')} bends, so it only comes close to a smooth curve.`)}
    <p>Many people think that ReLU itself is the curve. But ReLU is two straight pieces with one bend. In the widget, one neuron alone is one hinge, and the curve appears only in the sum.</p>`),
    },
    {
      kind: 'widget', id: 'hinges', className: 'widget', label: 'Interactive: the hinge builder',
      mount: (host, D) => mountHingeBuilder(host, D),
      tasks: (D, T) => [
        { do: `Press All off, then switch on neuron ${T(UNIT(4), 'setup')} only.`,
          see: 'You see one hinge. The thick sum line sits a little lower, because it also adds the bias of the second layer.' },
        { do: `Switch on neuron ${T(UNIT(0), 'setup')}, then neuron ${T(UNIT(1), 'setup')}.`,
          see: 'Each one adds a bend. The first lifts the right side, and the second lifts the left side.' },
        { do: `Press All on, then switch off neuron ${T(UNIT(1), 'setup')}.`,
          see: 'The left arm of the curve drops, and the right side stays the same.' },
      ],
    },
    {
      kind: 'quiz', id: 'bends', purpose: 'transfer', type: 'choice',
      prompt: '<p>You want a curve shaped like the letter W. It goes down, up, down and up again, so it has three bends. What is the smallest number of hidden ReLU neurons that can draw it?</p>',
      options: [
        { html: 'Three, one for each bend', correct: true,
          feedback: 'Yes. Each hidden neuron adds at most one bend.' },
        { html: 'One, because ReLU makes the bends',
          feedback: 'One ReLU neuron gives one bend only.' },
        { html: 'Two, one for each low point',
          feedback: 'The high point in the middle is a bend too.' },
        { html: 'Four, one for each straight piece',
          feedback: 'The bends make the pieces. Three bends give four pieces.' },
      ],
      reveal: '<p>Each hidden neuron adds at most one bend, so three bends need at least three neurons.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('sigmoid', '4 · Sigmoid, and a gradient that gets small', `
    <p>Another activation function is the ${term('sigmoid')}, written ${sym('sigma', 'σ')}. This small sigma is not the Σ of sums. The sigmoid changes any number into a number between zero and one:</p>
    ${eq('\\sigma(z) = \\frac{1}{1 + e^{-z}}', 'The sigmoid function')}
    <p>Here ${sym('exp', '<i>e</i>')} is a fixed number, <code>2.718</code> to four figures.</p>
    <p>The gradient of the sigmoid tells how much σ changes when <i>z</i> changes a little. It is largest at <i>z</i> = ${T(`${F}.z.${Z.zero}`, 'setup')}, where it is ${T(`${SIG}.grad.${Z.zero}`, 'contrast')}. At <i>z</i> = ${T(`${F}.z.${Z.four}`, 'setup')} the curve is flat, and it is only ${T(`${SIG}.grad.${Z.four}`, 'contrast')}.</p>
    <p>On the way back, training multiplies by this gradient. So a weight before a flat part of the sigmoid gets a tiny gradient. The lesson on ${term('backpropagation', 'backprop')} shows how.</p>
    <p>The gradient of ReLU is one for every positive weighted sum and zero for every negative one.</p>`),
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
      prompt: () => `<p>A training step changes a weight by the ${term('learning-rate')} times its gradient. Weight A sits before a flat part of a sigmoid, so its gradient is much smaller than that of weight B. With the same learning rate, which weight moves further over many steps?</p>`,
      options: [
        { html: 'Weight B, so weight A barely learns', correct: true,
          feedback: 'Yes. Each step of A is much smaller.' },
        { html: 'Both the same, because the learning rate sets the step',
          feedback: 'The step is the learning rate times the gradient.' },
        { html: 'Weight A, because it has more to learn',
          feedback: 'A step follows the gradient, not what the weight still has to learn.' },
      ],
      reveal: '<p>Every step of A is much smaller, so after many steps A has moved much less.</p>',
    },
    {
      kind: 'explain', id: 'why-activation',
      prompt: '<p>Explain why a network needs an activation function between two layers.</p>',
      model: '<p>Without it, two layers multiply out to one weighted sum, which draws only a straight line. The activation function lets each neuron bend it.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>Two layers with no activation function act like one layer, so they draw only a straight line. ReLU turns each hidden neuron into a hinge, and the sum of the hinges bends to follow the points.</p>
    <p>Sigmoid also bends, but its gradient is tiny where it is flat, so the weights before it barely learn.</p>`,
      next: () => `<p>This network gives one number for each input. How does a network choose between two ${term('class', 'classes')}, such as two colours of points?</p>`,
    },
  ],
};
