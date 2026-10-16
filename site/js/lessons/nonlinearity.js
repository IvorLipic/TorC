// Lesson "nonlinearity": two linear layers in a row are one linear layer; relu and sigmoid with their derivatives; two models
// fitted to t = x^2 with and without relu; softmax. Every number comes from data/nonlinearity.json
// (tools/trace/trace_nonlinear.cpp) through T(). Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { math } from '../core/katex.js';
import { mountActivationPlot, mountCurveFit } from '../widgets/nonlinear-widgets.js';

const F = 'files.nonlinearity';
const RELU = `${F}.activations.0`, SIG = `${F}.activations.1`;
const LIN = `${F}.models.0`, NET = `${F}.models.1`;
const last = (m) => `${m}.checkpoints.3`;
// indices into the recorded grid z = -4, -3.9, ..., 4
const Z = { m2: 20, zero: 40, p3: 70, p4: 80 };
const eq = (latex) => `<div class="nl-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="nlx" aria-labelledby="nl-${id}"><h2 id="nl-${id}">${title}</h2>${body}</section>`;
const row = (T, k) => [0, 1, 2].map((i) => T(`${F}.softmax.${k}.output.${i}`)).join(', ');
const input = (T, k) => [0, 1, 2].map((i) => T(`${F}.softmax.${k}.input.${i}`)).join(', ');

const STYLE = `<style>
.nlx p{max-width:44rem}
.nlx h2{margin-top:2rem}
.nl-eq{overflow-x:auto;margin:.2rem 0}
.nl-eq .katex-display{margin:.4em 0}
.nl-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.nl-code code{background:none;padding:0}
</style>`;

export default {
  id: 'nonlinearity',
  title: 'Nonlinearity',
  trace: ['nonlinearity.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Nonlinearity</h1>
    <p class="lead">A network needs a function between its linear layers. Without one, many layers can do no more than one layer. This lesson shows why, and what ReLU, sigmoid and softmax do.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'Two linear layers in a row are the same as one linear layer.',
        'An activation function between the layers lets the model fit curves.',
        'ReLU sets negative values to zero. Sigmoid maps every value to a number between zero and one.',
        'Softmax turns a row of numbers into positive numbers that add up to one.',
      ],
    },
    {
      kind: 'text',
      html: () => sec('collapse', '1 · Two linear layers are one linear layer', `
    <p>Put one linear layer after another and multiply out the result.</p>
    ${eq('W_2\\,(W_1 x + b_1) + b_2 \\;=\\; (W_2 W_1)\\,x + (W_2 b_1 + b_2)')}
    <p>The product ${math('W_2 W_1', false)} is one matrix, and ${math('W_2 b_1 + b_2', false)} is one vector. So the two layers compute one linear function. More linear layers add parameters, but they do not let the model bend. An activation function between the layers changes this.</p>
    <pre class="nl-code" tabindex="0" role="group" aria-label="Two layers with an activation in TorC"><code>Variable h = l1(x);
Variable r = relu(h);    // without this line, the model is linear
Variable y = l2(r);</code></pre>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('functions', '2 · ReLU and sigmoid', `
    ${eq('\\mathrm{relu}(z) = \\max(0, z) \\qquad \\sigma(z) = \\frac{1}{1 + e^{-z}}')}
    <p>The derivative of ReLU is one for a positive input and zero for a negative input. At exactly zero, TorC uses zero. The derivative of sigmoid is ${math('\\sigma(z)\\,(1 - \\sigma(z))', false)}. It is largest at zero, where it is ${T(`${SIG}.grad.${Z.zero}`)}. At <code>z</code> = ${T(`${F}.z.${Z.p4}`)} it is only ${T(`${SIG}.grad.${Z.p4}`)}. So a large input to sigmoid sends back a very small gradient.</p>`),
    },
    {
      kind: 'widget', id: 'activation', className: 'widget', label: 'Interactive: ReLU and sigmoid with their derivatives',
      mount: (host, D) => mountActivationPlot(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => sec('fit', '3 · Fitting a curve', `
    <p>Two models learn the curve ${math('t = x^2', false)} from ${T(`${F}.setup.points`)} points. Each model has a hidden layer with ${T(`${F}.setup.hidden`)} units, and both start from the same weights. Only one of them has a ReLU after the hidden layer. TorC trains each model for ${T(`${F}.setup.steps`)} steps with <code>optim::Adam</code> and <code>nn::MSELoss</code>.</p>`),
    },
    {
      kind: 'predict', id: 'p-line', title: 'Before you try it',
      prompt: () => '<p>What does the model without ReLU learn? Think about the best straight line through a curve that is symmetric around zero.</p>',
      reveal: (D, T) => `<p>It learns a horizontal line at ${T(`${LIN}.derived.b_eff`)}, which is the mean of the targets. Its loss stays at ${T(`${last(LIN)}.loss`)}. The model with ReLU bends, and its loss falls to ${T(`${last(NET)}.loss`)}.</p>`,
    },
    {
      kind: 'widget', id: 'fit', className: 'widget', label: 'Interactive: two models fitted to t = x squared, with and without ReLU',
      mount: (host, D) => mountCurveFit(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => sec('softmax', '4 · Softmax', `
    ${eq('\\mathrm{softmax}(z)_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}')}
    <p>Softmax makes every entry positive, and the entries add up to one. So a classifier can read them as probabilities. <code>softmax(z, 1)</code> works on each row. For the row ${input(T, 0)} it gives ${row(T, 0)}.</p>
    <p>If you add the same number to every entry, the result does not change. TorC uses this fact. It subtracts the largest entry before it computes the exponentials. So a row of very large numbers gives the same result, and no exponential overflows.</p>`),
    },
    {
      kind: 'check', id: 'c-relu',
      prompt: (D, T) => `<p>In TorC, what is the derivative of ReLU at <code>z</code> = ${T(`${F}.z.${Z.m2}`)}, and at <code>z</code> = ${T(`${F}.z.${Z.p3}`)}?</p>`,
      reveal: (D, T) => `<p>The derivative is ${T(`${RELU}.grad.${Z.m2}`)} at the first input, because that input is negative. It is ${T(`${RELU}.grad.${Z.p3}`)} at the second input. A negative input stops the gradient completely.</p>`,
    },
    {
      kind: 'check', id: 'c-softmax',
      prompt: (D, T) => `<p>What does softmax give for the row ${input(T, 3)}?</p>`,
      reveal: (D, T) => `<p>Each entry is ${T(`${F}.softmax.3.output.0`)}. The entries are equal, so each gets the same share of one.</p>`,
    },
    {
      kind: 'text',
      html: () => '<section class="nlx"><p>The next lesson, <a href="../modules/">Modules</a>, puts layers and activations together into one model.</p></section>',
    },
  ],
};
