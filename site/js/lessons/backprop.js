// Lesson "backprop" (Part 3 `networks`, curriculum v3, template 3; brief §6 "backprop"): why nudging every parameter is too slow,
// the chain rule on the notebook model (with C7 fixed), the scene `reuse` of the Backprop lab (one graph lab per page), and one
// small network backward with real numbers (scene `net`, one sample). ∂ is introduced here; the matrix rule of a layer is in the
// In TorC panel. Every number comes from data/autograd_trace.json or data/why-gradients.json through a role-tagged T()
// (docs/authoring.md "Number roles"). Notes: docs/lessons/backprop.md.
import { math } from '../core/katex.js';
import { worked, analogy, inTorc, term, sym } from '../core/blocks.js';
import { defaultFooter } from '../core/sections.js';
import { mountGraphLab } from '../widgets/graph-lab.js';
import { mountBackpropCost } from '../widgets/backprop-cost.js';
import { mountBlameNet } from '../widgets/blame-net.js';
import { reuseLayout } from './backprop.scenes.js';

const F = 'files.why-gradients';
const stage = (i, rest) => `${F}.nudge.stages.${i}.${rest}`;
const sec = (id, title, body) => `<section class="bp" aria-labelledby="bp-${id}">${title ? `<h2 id="bp-${id}">${title}</h2>` : ''}${body}</section>`;
const part = (body) => `<section class="bp">${body}</section>`;
const dd = (top, bottom) => sym('partial', math(`\\frac{\\partial ${top}}{\\partial ${bottom}}`, false));
const TIMING = 'https://github.com/IvorLipic/TorC/blob/web/docs/lessons/backprop.md#the-cost-of-a-backward-pass';

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
  trace: ['autograd_trace.json', 'why-gradients.json'],
  template: 3,
  loop: 'adjust',
  footer: (D) => `${defaultFooter(lesson, D)}<p>The cost of a backward pass comes from a TorC timing of three networks on one computer, which <a href="${TIMING}">the notes of this lesson</a> record.</p>`,
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
        'Why does TorC add gradients together?',
      ],
    },
    {
      kind: 'text', setup: true,
      html: () => sec('nudge', 'The price of a nudge', `
    <p>You found a gradient by nudging. You changed the ${term('weight')} <code>w</code> a little, computed the ${term('loss')} again, and divided the change of the loss by the nudge.</p>
    <p>To compute the ${term('prediction')} and the loss from the input is a <strong>forward pass</strong>. So every nudge costs one forward pass.</p>`),
    },
    {
      kind: 'quiz', id: 'cost', purpose: 'predict', type: 'choice',
      prompt: (D, T) => `<p>The digit network has ${T.anchor('mlp_parameters')} parameters. To find all their gradients by nudging, how many forward passes do you need?</p>`,
      options: [
        { html: 'Two. One at the start, and one after you nudge all the parameters together.', feedback: 'One change of the loss cannot tell how much of it each parameter caused.' },
        { html: 'One at the start, and one more for each parameter.', correct: true, feedback: 'Yes. Each nudge needs its own forward pass.' },
        { html: 'One. The gradient comes out of the forward pass.', feedback: 'A forward pass gives only the loss, not its gradient.' },
      ],
      reveal: '<p>Nudging needs one forward pass more than the network has parameters.</p>',
    },
    {
      kind: 'text',
      html: () => sec('cost', '1 · One pass backward', `
    <p>Nudging costs one forward pass per parameter, so ten times more parameters need ten times more passes.</p>
    <p>${term('backpropagation', '<strong>Backpropagation</strong>')} works differently. After one forward pass, it makes one <strong>backward pass</strong> from the loss back through the ${term('model')} to every parameter.</p>
    <p>One backward pass costs about as much as a few forward passes, also in a large network. The line <code>L.backward()</code> starts it.</p>`),
    },
    {
      kind: 'widget', id: 'cost-counter', className: 'widget', label: 'Interactive: nudging and backpropagation, counted in passes',
      mount: (host, D) => mountBackpropCost(host, D),
      tasks: () => [{
        do: 'Choose the notebook model, then the small network, then the digit network.',
        see: 'The pile for nudging grows with the number of parameters. Backpropagation always needs two passes.',
      }],
    },
    {
      kind: 'text',
      html: (D, T) => sec('chain', '2 · The chain rule', `
    <p>How can one pass backward find a gradient? Go back to the notebook model at <code>w</code> = ${T('why.w.value', 'setup')}. The weight changes the prediction, the prediction changes the miss, and the miss changes the loss.</p>
    ${worked(`<p>Raise <code>w</code> by ${T(`${F}.nudge.delta`, 'worked')} and follow each link.</p>
    <ul class="bp-chain">
      <li>The prediction rises by ${T(stage(0, 'change'), 'worked')}, which is ${T(stage(0, 'local_slope'), 'worked')} times the nudge.</li>
      <li>The miss rises by ${T(stage(1, 'change'), 'worked')}, which is ${T(stage(1, 'local_slope'), 'worked')} times the change of the prediction.</li>
      <li>The loss changes by ${T(stage(2, 'change'), 'worked')}, which is about ${T(stage(2, 'local_slope'), 'worked')} times the change of the miss.</li>
    </ul>
    <p>${T(stage(0, 'local_slope'), 'worked')} × ${T(stage(1, 'local_slope'), 'worked')} × (${T(stage(2, 'local_slope'), 'worked')}) = <strong>${T(`${F}.nudge.local_slope_product`, 'worked')}</strong></p>`)}
    <p>This is the gradient that you found by nudging <code>w</code> directly. The ${term('chain-rule')} says that along a chain, the ratios of the links multiply.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => part(`
    ${analogy('Gear B turns twice as fast as gear A, and gear C turns three times as fast as B. Then C turns six times as fast as A.',
    'A gear keeps its ratio. The ratio of a link changes when <code>w</code> changes, so the product holds only at the current point.')}
    <p>The backward pass computes this product from the loss back to <code>w</code>. Each link multiplies the gradient that it gets by its ratio and passes the result on. So every value on the way gets its gradient, and the earlier links reuse the work of the later ones.</p>
    ${worked(`<p>The loss is the miss times the miss. In calculus, the gradient of a square is twice the miss, ${T('why.e.grad', 'worked')}. TorC treats each use of the miss separately. Each use gives ${T('why.e.value', 'worked')}, and TorC adds them.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'reuse', purpose: 'predict', type: 'numeric',
      answer: 'reuse.x.grad', tolerance: 0.01,
      prompt: (D, T) => `<p>Here <code>y</code> = <code>x</code>·<code>x</code> + <code>x</code> with <code>x</code> = ${T('reuse.x.value', 'setup')}, so the formula uses <code>x</code> three times. What is the gradient of <code>y</code> for <code>x</code>?</p>`,
      reveal: (D, T) => `<p>The gradient is ${T('reuse.x.grad', 'reveal')}. The two uses in the multiplication each send back ${T('reuse.x.value', 'reveal')}. The addition sends back ${T('reuse.m.grad', 'reveal')}, and TorC adds the three.</p>`,
    },
    {
      kind: 'widget', id: 'reuse-lab', className: 'lab', label: 'Interactive lab: one value with three uses',
      mount: (host, D, ctx) => mountGraphLab(host, { scenes: { reuse: D.scenes.reuse }, layouts: { reuse: reuseLayout(ctx.makeT()) }, defaultScene: 'reuse' }),
      tasks: () => [{
        do: 'Press Next until the backward pass reaches <code>x</code>.',
        see: 'Three gradients come back to <code>x</code>, one for each use, and their sum is the gradient of <code>x</code>.',
      }],
    },
    {
      kind: 'text',
      html: () => sec('adds', '3 · One value, several uses', `
    <p>When a computation uses a value in several places, the value gets a gradient from each place, and TorC adds them. This is right, because a nudge of the value acts through every use at once.</p>
    <p>For the same reason, TorC adds every new gradient to the gradient that a parameter already holds. So a second <code>L.backward()</code> adds to the first. The lesson on the training loop shows how to clear it.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('net', '4 · Blame flows back through a small network', `
    <p>Now take a ${term('neural-network')} with two inputs, a ${term('layer')} of two ${term('neuron', 'neurons')}, and one output. Its inputs are ${T('net.x.value.0', 'setup')} and ${T('net.x.value.1', 'setup')}. Its ${term('target')} is ${T('net.t.value', 'setup')}, and the miss, prediction minus target, is ${T('net.d.value', 'setup')}.</p>
    <p>When the loss depends on many parameters, we write the gradient for one of them as ${dd('L', 'w')}. The sign ∂ means the same as d.</p>
    <p>Read the gradient of a value as its share of the blame for the miss. A neuron gets the gradient of the prediction times its output weight. A weight gets the gradient of its neuron times its input.</p>
    ${worked(`<p>The gradient of the prediction is twice the miss, ${T('net.o.grad', 'worked')}. The second neuron has the output weight ${T('net.W2.value.1', 'worked')}. So its gradient is ${T('net.o.grad', 'worked')} × (${T('net.W2.value.1', 'worked')}) = <strong>${T('net.h.grad.1', 'worked')}</strong>.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'your-weight', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: 'net.W1.grad.3', tolerance: 0.01,
      prompt: (D, T) => `<p>One weight goes from the second input, ${T('net.x.value.1', 'setup')}, to the second neuron. The gradient of that neuron is ${T('net.h.grad.1', 'setup')}. What is the gradient of this weight?</p>`,
      reveal: (D, T) => `<p>Multiply the input by the neuron's gradient. The result is ${T('net.W1.grad.3', 'reveal')}.</p><p>A nudge of this weight changes the neuron's sum by ${T('net.x.value.1', 'reveal')} times the nudge, and the chain rule multiplies.</p>`,
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
    ${inTorc('The rule for a whole layer', `<p>A layer of TorC (<code>nn::Linear</code>) computes <code>y</code> = <code>x</code>·<code>W</code>ᵀ + <code>b</code> for a row of inputs <code>x</code>. Its backward pass uses three rules:</p>
    <div class="bp-eq" tabindex="0" role="group" aria-label="The backward rules of a layer">${math(`\\frac{\\partial L}{\\partial W} = \\Big(\\frac{\\partial L}{\\partial y}\\Big)^{\\top} x \\qquad \\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y}\\, W \\qquad \\frac{\\partial L}{\\partial b} = \\text{the sum of the rows of } \\frac{\\partial L}{\\partial y}`)}</div>
    <p>Each entry of the first rule is the gradient of a neuron times an input. The backward step of <code>matmul</code> computes these products, and <code>accumulate_grad</code> adds each one to the stored gradient.</p>`)}`),
    },
    {
      kind: 'quiz', id: 'switched-off', purpose: 'transfer', type: 'choice',
      prompt: () => `<p>Take another input. This time the sum of the first neuron is negative, so ${term('relu', 'ReLU')} makes its output zero. What gradient do the two weights into the first neuron get?</p>`,
      options: [
        { html: 'Zero', correct: true, feedback: 'Yes. A small change of a negative sum does not change the ReLU output.' },
        { html: 'The same as before, because the weights did not change', feedback: 'The path from these weights to the loss is now blocked.' },
        { html: 'A larger one, because this neuron is wrong', feedback: 'A neuron that is off gets no blame, because it passes no gradient back.' },
      ],
      reveal: '<p>The neuron’s gradient is zero, and zero times any input is zero. These two weights do not change in this step.</p>',
    },
    {
      kind: 'quiz', id: 'count', purpose: 'cumulative', from: 'linear-layer', type: 'numeric',
      answer: 'files.autograd_trace.net_counts.parameters', tolerance: 0.5,
      prompt: () => `<p>The small network has two inputs, a layer of two neurons and one output. Each neuron and the output have a ${term('bias')}. How many parameters does it have?</p>`,
      reveal: (D, T) => `<p>Each neuron and the output have two weights and a bias. That makes ${T('files.autograd_trace.net_counts.parameters', 'reveal')} parameters.</p>`,
    },
    {
      kind: 'explain', id: 'why-one-pass',
      prompt: '<p>Explain why one backward pass can replace a nudge of every parameter.</p>',
      model: '<p>The backward pass multiplies the ratios of the links from the loss back, so all parameters share the work.</p>',
    },
    {
      kind: 'answer',
      html: () => `<p>No. Nudging needs one forward pass per parameter, and backpropagation needs one forward and one backward pass.</p>
    <p>The backward pass multiplies the ratios of the links from the loss back, so every value gets its gradient. A value with several uses gets one from each use, and TorC adds them.</p>`,
      next: () => '<p>Every gradient is now exact and cheap. Why can a network with many layers still be hard to train?</p>',
    },
  ],
};

export default lesson;
