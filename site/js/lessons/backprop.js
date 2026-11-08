// Lesson "backprop" (Part 3 `networks`, curriculum v3, template 3; brief §6 "backprop"): why nudging every parameter is too slow,
// the chain rule on the notebook model (with C7 fixed), the scene `reuse` of the Backprop lab (one graph lab per page), and one
// small network backward with real numbers (scene `net`, one sample). ∂ and the matrix rule of a layer are in the In TorC panel.
// Every number comes from data/autograd_trace.json, data/why-gradients.json or data/backprop-timing.json (a timing, see
// docs/trace-schema.md) through a role-tagged T() (docs/authoring.md "Number roles"). Notes: docs/lessons/backprop.md.
import { math } from '../core/katex.js';
import { worked, analogy, inTorc, term, sym } from '../core/blocks.js';
import { defaultFooter } from '../core/sections.js';
import { mountGraphLab } from '../widgets/graph-lab.js';
import { mountBackpropCost } from '../widgets/backprop-cost.js';
import { mountBlameNet } from '../widgets/blame-net.js';
import { reuseLayout } from './backprop.scenes.js';

const F = 'files.why-gradients';
const A = 'files.autograd_trace';
const TM = 'files.backprop-timing.derived';
const stage = (i, rest) => `${F}.nudge.stages.${i}.${rest}`;
const sec = (id, title, body) => `<section class="bp" aria-labelledby="bp-${id}">${title ? `<h2 id="bp-${id}">${title}</h2>` : ''}${body}</section>`;
const part = (body) => `<section class="bp">${body}</section>`;
const dd = (top, bottom) => sym('partial', math(`\\frac{\\partial ${top}}{\\partial ${bottom}}`, false));
const NOTES = 'https://github.com/IvorLipic/TorC/blob/web/docs/lessons/backprop.md#the-cost-of-a-backward-pass';

const STYLE = `<style>
.bp p{max-width:44rem}
.bp h2{margin-top:2rem}
.bp-chain{margin:.3rem 0 .3rem 1.2rem;padding:0;max-width:42rem}
.bp-chain li{margin:.15rem 0}
.bp-eq{overflow-x:auto;margin:.3rem 0}
#reuse-lab #scene-tabs{display:none}
#reuse-lab .lab-top{justify-content:flex-end}
</style>`;

const lesson = {
  id: 'backprop',
  title: 'How the computer finds every gradient at once',
  trace: ['autograd_trace.json', 'why-gradients.json', 'backprop-timing.json'],
  template: 3,
  loop: 'adjust',
  footer: (D) => `${defaultFooter(lesson, D)}<p>The cost of a backward pass comes from a TorC timing of three networks on one computer, which <a href="${NOTES}">the notes of this lesson</a> record. The general idea is the “cheap gradient principle” in A. Griewank and A. Walther, <cite>Evaluating Derivatives</cite>, 2nd edition, SIAM, 2008.</p>`,
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>How the computer finds every gradient at once</h1>
    <p class="lead">${term('training', 'Training')} needs the ${term('gradient')} of every ${term('parameter')}, at every step.</p></section>`,
    },
    {
      kind: 'questions',
      items: (D, T) => [
        `The digit network has ${T.anchor('mlp_parameters')} parameters. Does the computer nudge each one to find its gradient?`,
        'How can one pass backward give the gradient of every parameter?',
        'A formula uses one value twice. How does TorC find the gradient of that value?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: () => sec('nudge', 'The price of a nudge', `
    <p>You found a gradient by nudging. You changed the ${term('weight')} <code>w</code> a little, computed the ${term('loss')} again, and divided the change of the loss by the nudge.</p>
    <p>To compute the ${term('prediction')} and the loss from the input is a <strong>forward pass</strong>.</p>`),
    },
    {
      kind: 'quiz', id: 'cost', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>The digit network has ${T.anchor('mlp_parameters')} parameters. To find all their gradients by nudging, how many forward passes do you need?</p>`,
      options: [
        { html: 'Two. One at the start, and one after nudging all the parameters together.', feedback: 'One change of the loss cannot tell how much of it each parameter caused.' },
        { html: 'One at the start, and one more for each parameter.', correct: true, feedback: 'Yes. Each nudge needs its own forward pass.' },
        { html: 'One. The gradient comes out of the forward pass.', feedback: 'A forward pass gives only the loss, not its gradient.' },
      ],
      reveal: '<p>Each nudge needs a forward pass to compute the new loss, and the start needs one more.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('cost', '1 · One pass backward', `
    <p>${term('backpropagation', '<strong>Backpropagation</strong>')} works differently. After one forward pass, it makes one <strong>backward pass</strong> from the loss back through the ${term('model')} to every parameter. The line <code>L.backward()</code> starts it.</p>
    <p>In one TorC timing on one computer, a forward and a backward pass together cost as much as ${T(`${TM}.backprop_over_nudge_min`, { role: 'contrast', sig: 1 })} to ${T(`${TM}.backprop_over_nudge_max`, { role: 'contrast', sig: 1 })} nudges. That held for networks up to ${T(`${TM}.largest_over_digit`, { role: 'contrast', sig: 2 })} times the size of the digit network.</p>`),
    },
    {
      kind: 'widget', id: 'cost-counter', className: 'widget', label: 'Interactive: nudging and backpropagation, counted in forward passes',
      mount: (host, D) => mountBackpropCost(host, D),
      tasks: () => [{
        do: 'Choose each network in turn.',
        see: 'The block for nudging grows with the parameters, and the red bar stays small.',
      }],
    },
    {
      kind: 'text',
      html: (D, T) => {
        const x = T.num(`${F}.setup.x`), t = T.num(`${F}.setup.t`);
        return sec('chain', '2 · The chain rule', `
    <p>Go back to the notebook model, ${math(`L = (${x}w - ${t})^2`, false)}, at <code>w</code> = ${T('why.w.value', 'setup')}. Call the prediction minus the price the miss, ${math(`${x}w - ${t}`, false)}. At the start it is ${T('why.e.value', 'setup')}.</p>
    ${worked(`<p>Raise <code>w</code> by ${T(`${F}.nudge.delta`, 'worked')} and follow each link.</p>
    <ul class="bp-chain">
      <li>The prediction rises by ${T(stage(0, 'change'), 'worked')}, which is ${T(stage(0, 'local_slope'), 'worked')} times the nudge.</li>
      <li>The miss rises by ${T(stage(1, 'change'), 'worked')}, which is ${T(stage(1, 'local_slope'), 'worked')} times the change of the prediction.</li>
      <li>The loss changes by ${T(stage(2, 'change'), 'worked')}, about ${T(stage(2, 'local_slope'), 'worked')} times the change of the miss. That is twice the miss.</li>
    </ul>
    <p>${T(stage(0, 'local_slope'), 'worked')} × ${T(stage(1, 'local_slope'), 'worked')} × (${T(stage(2, 'local_slope'), 'worked')}) = <strong>${T(`${F}.nudge.local_slope_product`, 'worked')}</strong></p>`)}
    <p>This is the gradient that you found by nudging. The ${term('chain-rule')} says that along a chain, the ratios of the links multiply.</p>`);
      },
    },
    {
      kind: 'text',
      html: (D, T) => part(`
    ${analogy('Gear B turns twice as fast as gear A, and gear C three times as fast as B. Then C turns six times as fast as A.',
    'A gear keeps its ratio. The ratio of a link changes when <code>w</code> changes, so the product holds only at the current point.')}
    <p>The backward pass computes this product from the loss back. Each link multiplies the gradient that it gets by its ratio and passes the result on. So each ratio is computed once, and every parameter behind that link reuses it.</p>
    ${worked(`<p>The loss is the miss times the miss. In a product, each factor gets the other factor times the gradient that arrives. One arrives at the loss, so each use of the miss gets ${T('why.e.value', 'worked')}. TorC adds the two and gets ${T('why.e.grad', 'worked')}.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'reuse', purpose: 'predict', type: 'numeric',
      answer: 'reuse.x.grad', tolerance: 0.01,
      prompt: (D, T) => `<p>Here <code>y</code> = <code>x</code>·<code>x</code> + <code>x</code> with <code>x</code> = ${T('reuse.x.value', 'setup')}, so the formula uses <code>x</code> three times. What is the gradient of <code>y</code> for <code>x</code>?</p>`,
      reveal: (D, T) => `<p>The gradient is ${T('reuse.x.grad', 'reveal')}. Each use in the multiplication sends back ${T(`${A}.reuse_uses.mul_first`, 'reveal')}, and the addition sends back ${T(`${A}.reuse_uses.add`, 'reveal')}.</p>`,
    },
    {
      kind: 'widget', id: 'reuse-lab', className: 'lab', label: 'Interactive lab: one value with three uses',
      mount: (host, D, ctx) => mountGraphLab(host, { scenes: { reuse: D.scenes.reuse }, layouts: { reuse: reuseLayout(ctx.makeT()) }, defaultScene: 'reuse' }),
      tasks: () => [{
        do: 'Press Next until the backward pass reaches <code>x</code>.',
        see: 'Three gradients come back to <code>x</code>, one per use, and they add up to its gradient.',
      }],
    },
    {
      kind: 'text',
      html: () => sec('adds', '3 · One value, several uses', `
    <p>When a computation uses a value in several places, the value gets a gradient from each place, and TorC adds them. This is right, because a nudge of the value acts through every use at once.</p>
    <p>TorC also adds across training steps. After a new forward pass, <code>L.backward()</code> adds the new gradient to the one that a parameter already holds. The lesson on the training loop shows how to clear it.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('net', '4 · Blame flows back through a small network', `
    <p>Now take a ${term('neural-network')} with two inputs, a ${term('layer')} of two ${term('neuron', 'neurons')} with ${term('relu', 'ReLU')}, and one output. Its inputs are ${T('net.x.value.0', 'setup')} and ${T('net.x.value.1', 'setup')}. Its ${term('target')} is ${T('net.t.value', 'setup')}, and the miss, prediction minus target, is ${T('net.d.value', 'setup')}.</p>
    <p>Read a value's gradient as its share of the blame for the miss. Its sign says which way to move the value to lower the loss.</p>
    <p>A neuron gets the gradient of the prediction times its output weight, times the gradient of its ReLU. That gradient is one here, because both neurons are on. A weight gets the gradient of its neuron times its input.</p>
    ${worked(`<p>The gradient of the prediction is twice the miss, ${T('net.o.grad', 'worked')}. The second neuron has the output weight ${T('net.W2.value.1', 'worked')}. So its gradient is ${T('net.o.grad', 'worked')} × (${T('net.W2.value.1', 'worked')}) = <strong>${T('net.h.grad.1', 'worked')}</strong>.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'your-weight', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: 'net.W1.grad.3', tolerance: 0.01,
      prompt: (D, T) => `<p>One weight goes from the second input, ${T('net.x.value.1', 'setup')}, to the second neuron. The gradient of that neuron is ${T('net.h.grad.1', 'setup')}. What is the gradient of this weight?</p>`,
      reveal: (D, T) => `<p>Multiply the input by the neuron's gradient. The result is ${T('net.W1.grad.3', 'reveal')}.</p><p>A nudge of this weight changes the neuron's sum by ${T('net.x.value.1', 'reveal')} times the nudge.</p>`,
    },
    {
      kind: 'widget', id: 'blame-net', className: 'widget', label: 'Interactive: the small network, forward and backward',
      mount: (host, D) => mountBlameNet(host, D),
      tasks: () => [
        { do: 'Press Next until the backward pass reaches the neurons.',
          see: 'The second neuron gets a positive gradient, because its output weight is negative.' },
        { do: 'Press Next once more and compare the four weights of the first layer.',
          see: 'The weights from the second input get twice the gradient of the weights from the first input.' },
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('wrong', '5 · A tempting idea that is wrong', `
    <p>Many people think that every weight gets an equal share of the blame.</p>
    <p>The network above shows otherwise. The weight from the second input to the first neuron gets ${T('net.W1.grad.2', { role: 'contrast', sig: 3 })}, and the weight from the first input to the second neuron gets ${T('net.W1.grad.1', { role: 'contrast', sig: 3 })}.</p>
    <p>A gradient depends on the input that the weight carries and on its path to the loss. So one step changes some weights much more than others.</p>
    ${inTorc('The rule for a whole layer', `<p>For a loss of many parameters, the gradient for one of them is written ${dd('L', 'w')}, where ∂ means the same as d.</p>
    <p><code>nn::Linear</code> computes <code>y</code> = <code>x</code>·<code>W</code>ᵀ + <code>b</code> as <code>matmul(x, transpose(W))</code> plus the ${term('bias')}. Its backward pass gives three rules:</p>
    <div class="bp-eq" tabindex="0" role="group" aria-label="The backward rules of a layer">${math(`\\frac{\\partial L}{\\partial W} = \\Big(\\frac{\\partial L}{\\partial y}\\Big)^{\\top} x \\qquad \\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y}\\, W \\qquad \\frac{\\partial L}{\\partial b} = \\text{the sum of the rows of } \\frac{\\partial L}{\\partial y}`)}</div>
    <p>Each entry of the first rule is a neuron's gradient times an input. The backward steps of <code>matmul</code> and <code>transpose</code> produce it, and <code>accumulate_grad</code> adds it to the stored gradient.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'switched-off', purpose: 'transfer', type: 'choice',
      prompt: '<p>Take another input. This time the sum of the first neuron is negative, so ReLU makes its output zero. What gradient do the two weights into the first neuron get?</p>',
      options: [
        { html: 'Zero', correct: true, feedback: 'Yes. A small change of a negative sum does not change the ReLU output.' },
        { html: 'The same as before, because the weights did not change', feedback: 'The path from these weights to the loss is now blocked.' },
        { html: 'A larger one, because this neuron is wrong', feedback: 'A neuron that is off gets no blame, because it passes no gradient back.' },
      ],
      reveal: '<p>For a negative sum, the gradient of ReLU is zero. So the neuron’s gradient is zero, and zero times any input is zero.</p>',
    },
    {
      kind: 'quiz', id: 'count', purpose: 'cumulative', from: 'linear-layer', type: 'numeric',
      answer: `${A}.net_counts.parameters`, tolerance: 0.5,
      prompt: () => `<p>The small network has two inputs, a layer of two neurons and one output. Each neuron and the output have a ${term('bias')}. How many parameters does it have?</p>`,
      reveal: (D, T) => `<p>Each neuron and the output have two weights and a bias. That makes ${T(`${A}.net_counts.parameters`, 'reveal')} parameters.</p>`,
    },
    {
      kind: 'explain', id: 'why-one-pass',
      prompt: '<p>Explain why one backward pass can replace a nudge of every parameter.</p>',
      model: '<p>The backward pass computes each link’s ratio once and reuses it for every parameter behind that link.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>No. Nudging needs one forward pass per parameter, and backpropagation one forward and one backward pass.</p>
    <p>The backward pass computes each link’s ratio once and reuses it for every parameter behind it. A value with several uses gets a gradient from each use, and TorC adds them.</p>`,
      next: () => '<p>Backpropagation finds every gradient without the small error of a nudge. Why can a network with many layers still be hard to train?</p>',
    },
  ],
};

export default lesson;
