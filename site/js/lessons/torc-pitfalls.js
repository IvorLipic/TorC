// Lesson "torc-pitfalls" (appendix "Using TorC", curriculum v3, template 3, new): "Why did my X fail?". The silent mistakes (a
// missing zero_grad, a step that skips a parameter, a missing reset) and the errors that TorC reports (a graph that lost a
// Variable, shape errors of MSELoss, CrossEntropyLoss, Linear and indexing, wrong settings), each with the real TorC message.
// Brief §6 appendix, D6 and decision 22 (the MSE shape pitfall lives only here). Numbers and messages come from
// autograd_trace.json (accumulation, lifetime, unused_parameter), optim.json, training.json, losses.json, linear.json and
// tensors.json (all written by tools/trace/*) through a role-tagged T() or the recorded text.
// Widget: widgets/loop-timeline.js (the zero_grad loop, kept for this page). Writing: docs/authoring.md "Writing rules".
import { escapeHtml } from '../core/dom.js';
import { term, worked } from '../core/blocks.js';
import { mountLoopTimeline } from '../widgets/loop-timeline.js';

const A = 'files.autograd_trace';
const L = 'files.losses';
const TR = 'files.training';
const OP = 'files.optim';
const sec = (id, title, body) => `<section class="pf" aria-labelledby="pf-${id}"><h2 id="pf-${id}">${title}</h2>${body}</section>`;
const err = (r) => `<code>${escapeHtml(r.exception)}: ${escapeHtml(r.what)}</code>`;
const code = (label, body) => `<pre class="pf-code" tabindex="0" role="group" aria-label="${label}"><code>${body}</code></pre>`;
const lifetime = (D, id) => D.files.autograd_trace.lifetime.find((c) => c.id === id);
const lerr = (D, id) => D.files.losses.errors.find((e) => e.id === id);
const lin = (D, id) => D.files.linear.shape_errors.find((e) => e.id === id);

const STYLE = `<style>
.pf p{max-width:44rem}
.pf h2{margin-top:2rem}
.pf-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.pf-code code{background:none;padding:0}
.gotcha ul{max-width:44rem}
</style>`;

export default {
  id: 'torc-pitfalls',
  title: 'Common errors in TorC programs',
  trace: ['autograd_trace.json', 'optim.json', 'training.json', 'losses.json', 'linear.json', 'tensors.json'],
  template: 3,
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Common errors in TorC programs</h1>
    <p class="lead">Some mistakes in a TorC program give an error message, and some give no message at all. Here you meet both kinds, with the real TorC message where there is one.</p></section>`,
    },
    {
      kind: 'quiz', id: 'p-twice', purpose: 'predict', type: 'choice', title: 'Before you read on',
      prompt: (D, T) => `<p>The ${term('weight')} is <code>w</code> = ${T(`${A}.accumulation.w`, 'setup')}, and the ${term('loss')} is <code>L = w · w</code>. After <code>L.backward()</code>, the stored ${term('gradient')} of <code>w</code> is ${T(`${A}.accumulation.after_first_backward`, 'setup')}. You call <code>L.backward()</code> a second time on the same <code>L</code>. What does the stored gradient hold now?</p>`,
      options: [
        { html: (D, T) => `${T(`${A}.accumulation.after_new_forward_and_backward`, 'setup')}, because each <code>backward()</code> adds to the stored gradient.`, feedback: () => 'The gradients add up across passes. But the first call already used up the record of this loss, so the second call adds nothing.' },
        { html: (D, T) => `It still holds ${T(`${A}.accumulation.after_second_backward_same_loss`, 'setup')}, because the second call does nothing.`, correct: true, feedback: () => 'Right. When <code>backward()</code> ends, TorC clears the record that the loss itself holds.' },
        { html: 'TorC throws an error, because the record is already used.', feedback: () => 'TorC returns without a message, so a repeated call can hide a mistake.' },
      ],
      reveal: (D, T) => `<p>It still holds ${T(`${A}.accumulation.after_second_backward_same_loss`, 'reveal')}. A new forward pass builds a new loss, and its <code>backward()</code> adds to the stored gradient. Then it holds ${T(`${A}.accumulation.after_new_forward_and_backward`, 'reveal')}.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('zero', '1 · Gradients add up', `
    <p>A new <code>backward()</code> adds to the stored gradient of a ${term('parameter')}. It does not replace it. <code>step()</code> then reads the sum. So the loop must remove the old gradient before the next pass, and <code>zero_grad()</code> does that.</p>
    ${code('A training step in TorC', `Variable out = model(x);
Variable L = mse(out, t);
L.backward();
optimizer.step();
optimizer.zero_grad();    // without this line, the gradients add up`)}`),
    },
    {
      kind: 'widget', id: 'zero-loop', className: 'widget', label: 'Interactive: ten training steps with and without zero_grad',
      mount: (host, D) => mountLoopTimeline(host, D),
      tasks: () => [
        { do: 'Leave the box ticked and read both charts.', see: 'The gradient bars get smaller from one iteration to the next. The loss falls toward zero.' },
        { do: 'Clear the box.', see: 'The first iteration stays the same. Later the bars grow and change sign, and the loss jumps up and down. The dashed line is the loop with <code>zero_grad()</code>.' },
      ],
    },
    {
      kind: 'gotcha', title: 'Why does my loss jump up and down?',
      html: () => '<p>Check that <code>zero_grad()</code> runs once in every iteration. Without it, each step uses the sum of all earlier gradients. The steps become too large, and TorC gives no message.</p>',
    },
    {
      kind: 'text',
      html: (D, T) => sec('skip', '2 · A step that changes nothing', `
    <p><code>step()</code> skips every parameter that has no stored gradient. It gives no message. This happens when you call <code>step()</code> before the first <code>backward()</code>, right after <code>zero_grad()</code>, or for a parameter that the loss does not use. ${term('adam', 'Adam')} and AdamW skip such a parameter too, so ${term('weight-decay', 'weight decay')} does not change it either.</p>
    <p>Take a loss that uses only one of two parameters, both at ${T(`${A}.unused_parameter.start`, 'setup')}. The step changes the parameter that the loss uses. The other parameter stays where it was.</p>
    ${worked(`<p>${T(`${A}.unused_parameter.start`, 'worked')} − ${T(`${A}.unused_parameter.lr`, 'worked')} × ${T(`${A}.unused_parameter.used_grad`, 'worked')} = ${T(`${A}.unused_parameter.used_after`, 'worked')}</p>`)}`),
    },
    {
      kind: 'gotcha', title: 'Why does my model not learn?',
      html: () => `<p>Check the order of the calls. It is forward pass, loss, <code>backward()</code>, <code>step()</code> and <code>zero_grad()</code>. Check that every parameter takes part in the loss. A parameter that the forward pass never uses gets no gradient.</p>`,
    },
    {
      kind: 'quiz', id: 'p-lifetime', purpose: 'predict', type: 'choice', title: 'Before you read on',
      prompt: (D) => `<p>The parameter <code>w</code> needs a gradient. What does this code do?</p>${code('A loss in one statement', escapeHtml(lifetime(D, 'temporary').code))}`,
      options: [
        { html: 'It works. The graph keeps the product <code>mul(w, w)</code> alive.', feedback: () => 'The graph points to its in-between results. It does not own them, and this one is gone when the statement ends.' },
        { html: 'TorC throws a <code>TorcError</code> in <code>backward()</code>.', correct: true, feedback: () => 'Right. TorC checks that every input that needs a gradient still exists.' },
        { html: 'It works, but the gradient is wrong.', feedback: () => 'TorC does not guess. It stops with an error before it computes anything.' },
      ],
      reveal: (D) => `<p>The message is ${err(lifetime(D, 'temporary'))}.</p>`,
    },
    {
      kind: 'text',
      html: (D) => sec('lifetime', '3 · A graph that lost a Variable', `
    <p>The graph of a loss does not own its Variables. Each node points to its inputs. Before <code>backward()</code> uses an input that needs a gradient, it checks that the input still exists.</p>
    <p>A temporary dies at the end of its statement. A local Variable dies when its function returns. So give every Variable of the graph a name, and keep it until after <code>backward()</code>. A ${term('model')} call such as <code>layer(x)</code> returns a temporary too.</p>
    ${code('Fails and works', `${escapeHtml(lifetime(D, 'model-temporary').code)}   // fails

${escapeHtml(lifetime(D, 'model-named').code)}   // works`)}`),
    },
    {
      kind: 'gotcha', title: 'Why did backward() throw a TorcError?',
      html: (D) => `<p>TorC reports ${err(lifetime(D, 'local'))}. Look for a temporary in the loss expression. Look for a Variable that needs a gradient and is made inside a function. Look for a second call of the same module. The page <a href="../modules/">Modules</a> explains the last case.</p>`,
    },
    {
      kind: 'quiz', id: 'p-reset', purpose: 'predict', type: 'numeric', title: 'Before you read on',
      answer: `${TR}.no_reset.1.steps`, tolerance: 0.01,
      prompt: (D, T) => `<p>A ${term('data-set', 'data set')} has ${T(`${TR}.setup.samples`, 'setup')} ${term('sample', 'samples')}, and the ${term('batch')} size is ${T(`${TR}.loaders.0.batch_size`, 'setup')}. The first ${term('epoch')} takes ${T(`${TR}.no_reset.0.steps`, 'setup')} steps. You train for three epochs and never call <code>reset()</code> on the loader. How many steps does the second epoch take?</p>`,
      reveal: (D, T) => `<p>The second epoch takes ${T(`${TR}.no_reset.1.steps`, 'reveal')} steps, and so does the third. After the last batch, <code>has_next()</code> stays false.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('reset', '4 · An epoch that does nothing', `
    <p>A loader gives each batch once. After the last batch, <code>has_next()</code> is false. <code>reset()</code> moves the loader back to the first batch, and it shuffles again when shuffling is on.</p>
    ${code('One epoch in TorC', `loader.reset();                    // start the epoch again
while (loader.has_next()) {
    auto [x, t] = loader.next_batch();
    // forward pass, loss, backward, step, zero_grad
}`)}`),
    },
    {
      kind: 'gotcha', title: 'Why does the second epoch do nothing?',
      html: (D) => `<p>Without <code>reset()</code>, the <code>while</code> loop runs zero times from the second epoch on. TorC gives no message, and the loss stays where it was. A direct call of <code>next_batch()</code> after the last batch throws ${err(D.files.training.exhausted)}.</p>`,
    },
    {
      kind: 'quiz', id: 'p-mse', purpose: 'predict', type: 'choice', title: 'Before you read on',
      prompt: (D) => `<p>A <code>Linear</code> ${term('layer')} with one output gives a ${term('prediction')} of the ${term('shape')} <code>(${D.files.losses.mse_mismatch.prediction_shape.join(', ')})</code>. Your ${term('target')} has the shape <code>(${D.files.losses.mse_mismatch.target_shape.join(', ')})</code>. What does <code>MSELoss</code> do?</p>`,
      options: [
        { html: 'It combines the two shapes and returns a loss.', feedback: () => 'Element-wise operations broadcast, but <code>MSELoss</code> checks that its two shapes are equal.' },
        { html: 'It throws a <code>ShapeError</code>.', correct: true, feedback: () => 'Right. The two shapes must be equal.' },
        { html: 'It reshapes the target to fit and returns a loss.', feedback: () => '<code>MSELoss</code> never reshapes your data.' },
      ],
      reveal: (D) => `<p>The message is ${err(D.files.losses.mse_mismatch)}.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('shapes', '5 · Errors about shapes and settings', `
    <p>Most errors that TorC reports are a <code>ShapeError</code> or a <code>TorcError</code>. The message names the shapes or the value that does not fit. Read the shapes first.</p>`),
    },
    {
      kind: 'gotcha', title: 'Why did MSELoss fail?',
      html: (D) => `<p>The prediction and the target must have the same shape. A <code>Linear</code> layer with one output gives <code>(batch, 1)</code>, so the target needs the shape <code>(batch, 1)</code> too. TorC reports ${err(D.files.losses.mse_mismatch)}.</p>`,
    },
    {
      kind: 'gotcha', title: 'Why did CrossEntropyLoss fail?',
      html: (D) => `<p>The ${term('logit', 'logits')} need the shape <code>(batch, classes)</code>. The targets are ${term('class')} numbers from zero up to one less than the number of classes. These are the messages.</p>
    <ul>
      <li>${err(lerr(D, 'not-integer'))} The target is not a whole number.</li>
      <li>${err(lerr(D, 'out-of-range'))} The class number is too large.</li>
      <li>${err(lerr(D, 'rank-1-logits'))} The logits have no batch dimension.</li>
      <li>${err(lerr(D, 'length'))} The number of targets differs from the batch size.</li>
    </ul>`,
    },
    {
      kind: 'gotcha', title: 'Why did my Linear layer fail?',
      html: (D) => `<p>The last size of the input must equal the number of inputs of the layer. The layer multiplies the input by its transposed weight, so the message names <code>matmul</code> and the shape <code>(in, out)</code>. For <code>Linear(784, 32)</code> and an input of the shape <code>(8, 10)</code>, TorC reports ${err(lin(D, 'linear-784x32-(8, 10)'))}. An input without a batch dimension gives ${err(lin(D, 'linear-3x2-(3)'))}. Here rank means the number of dimensions.</p>`,
    },
    {
      kind: 'gotcha', title: 'Why did my index or axis fail?',
      html: (D) => `<p>Positions and axes count from zero. The ${term('tensor')} <code>a</code> has the shape <code>(2, 3)</code>. For <code>a[2, 0]</code>, TorC reports ${err(D.files.tensors.index_errors[0])}. For <code>a.sum(2)</code>, it reports ${err(D.files.tensors.reduce_error)}.</p>`,
    },
    {
      kind: 'gotcha', title: 'Why did a constructor throw?',
      html: (D) => `<p>Sizes and ${term('learning-rate', 'learning rates')} must be positive. For a learning rate of zero, <code>SGD</code> reports ${err(D.files.optim.ctor_errors[0])}. For <code>Linear(0, 2)</code>, TorC reports ${err(lin(D, 'linear-zero-features'))}.</p>`,
    },
    {
      kind: 'text',
      html: () => '<section class="pf"><p>The next page, <a href="../notation/">The symbols of this course</a>, explains each symbol and how to read it aloud.</p></section>',
    },
  ],
};
