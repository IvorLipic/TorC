// Lesson "tensors": shape and row-major storage, reshape (a copy in TorC), broadcasting, reductions along an axis, and matmul.
// Every number and error message comes from data/tensors.json (tools/trace/trace_tensors.cpp) through T() or the recorded text.
// Widgets: tensor-widgets.js. Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { mountBroadcastGrid, mountAxisExplorer } from '../widgets/tensor-widgets.js';

const F = 'files.tensors';
const eq = (latex) => `<div class="tn-eq">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="tn" aria-labelledby="tn-${id}"><h2 id="tn-${id}">${title}</h2>${body}</section>`;
const err = (r) => `<code>${escapeHtml(r.exception)}: ${escapeHtml(r.what)}</code>`;
const list = (T, path, n) => Array.from({ length: n }, (_, k) => T(`${path}.${k}`)).join(', ');
// reductions in the trace, by index: axis 0 sum, mean, max, then axis 1 sum, mean, max
const RED = { sum0: 0, mean0: 1, max0: 2, sum1: 3, mean1: 4, max1: 5 };

const STYLE = `<style>
.tn p{max-width:44rem}
.tn h2{margin-top:2rem}
.tn-eq{overflow-x:auto;margin:.2rem 0}
.tn-eq .katex-display{margin:.4em 0}
.tn-code{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.6rem 1rem;overflow-x:auto;max-width:44rem;line-height:1.5}
.tn-code code{background:none;padding:0}
</style>`;

export default {
  id: 'tensors',
  title: 'Tensors',
  trace: ['tensors.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Tensors</h1>
    <p class="lead">A tensor is a block of numbers with a shape. Layers, losses and optimizers in TorC all work on tensors. This lesson shows how TorC stores them and how shapes combine.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        'A tensor has a shape and one list of numbers, stored row after row.',
        'For element-wise operations, TorC compares the two shapes from the right. A size of one repeats to fit the other size.',
        'A reduction such as <code>sum(axis)</code> removes one dimension from the shape.',
        'A matrix product needs equal inner sizes.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('storage', '1 · Shape and storage', `
    <p>The tensor <code>a</code> has shape <code>(2, 3)</code>, which means two rows and three columns. It holds ${T(`${F}.example.numel`)} numbers.</p>
    <pre class="tn-code" tabindex="0" role="group" aria-label="The tensor a in TorC"><code>Tensor a({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3});
float v = a[1, 2];               // row 1, column 2
Tensor r = a.reshape({3, 2});    // the same numbers, a new shape</code></pre>
    <p>TorC stores the numbers in one list, row after row. The element in row <code>i</code> and column <code>j</code> is at position <code>3·i + j</code>. So <code>a[1, 2]</code> is at position ${T(`${F}.example.index.5.flat`)}, and its value is ${T(`${F}.example.index.5.value`)}.</p>
    <p><code>reshape</code> gives the same numbers a new shape. In TorC, <code>reshape</code> copies the numbers. A change to <code>r</code> does not change <code>a</code>.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('broadcast', '2 · Broadcasting', `
    <p>An element-wise operation such as <code>add</code> combines the numbers at equal positions. TorC also accepts two different shapes if they fit. It compares the shapes from the right, one dimension at a time. Two sizes fit if they are equal or if one of them is one. A size of one repeats to fill the other size, and a missing dimension counts as one. If a pair of sizes does not fit, TorC throws a <code>ShapeError</code>.</p>`),
    },
    {
      kind: 'predict', id: 'p-row', title: 'Before you try it',
      prompt: () => '<p>Can TorC add a tensor of shape <code>(2)</code> to <code>a</code>, which has shape <code>(2, 3)</code>? Compare the shapes from the right.</p>',
      reveal: (D) => `<p>No. From the right, the sizes are <code>3</code> and <code>2</code>. They are not equal, and neither of them is one. TorC throws ${err(D.files.tensors.broadcast.find((b) => b.id === 'wrong-row').sum)}. To add one number to each row, use shape <code>(2, 1)</code>.</p>`,
    },
    {
      kind: 'widget', id: 'broadcast', className: 'widget', label: 'Interactive: a plus b for several shapes of b',
      mount: (host, D) => mountBroadcastGrid(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => sec('reduce', '3 · Reductions', `
    <p>A reduction combines the numbers along one axis. Axis <code>0</code> goes down the rows, and axis <code>1</code> goes across the columns. The axis disappears from the shape.</p>
    <p><code>a.sum(0)</code> adds each column and gives ${list(T, `${F}.reductions.${RED.sum0}.result.values`, 3)}, with shape <code>(3)</code>. <code>a.sum(1)</code> adds each row and gives ${list(T, `${F}.reductions.${RED.sum1}.result.values`, 2)}, with shape <code>(2)</code>. <code>mean</code> and <code>max</code> work the same way. Without an axis, <code>a.sum()</code> gives one number, ${T(`${F}.whole.sum`)}.</p>`),
    },
    {
      kind: 'widget', id: 'axis', className: 'widget', label: 'Interactive: sum, mean and max along axis 0 or axis 1',
      mount: (host, D) => mountAxisExplorer(host, D),
    },
    {
      kind: 'text',
      html: (D, T) => sec('matmul', '4 · The matrix product', `
    ${eq('\\underset{(m,\\,k)}{A}\\;\\cdot\\;\\underset{(k,\\,n)}{B}\\;=\\;\\underset{(m,\\,n)}{C}, \\qquad C_{ij} = \\sum_{l} A_{il}\\,B_{lj}')}
    <p>Each output is the sum of the products of one row of the first tensor and one column of the second. The two inner sizes must be equal. For <code>a</code> and its transpose, <code>a.matmul(a.transpose({1, 0}))</code> gives ${list(T, `${F}.matmul.a_at.values`, 4)}, with shape <code>(2, 2)</code>. The product of <code>a</code> with itself fails with this message.</p>
    <p>${err(D.files.tensors.matmul.a_a)}</p>
    <p>For tensors with more than two dimensions, TorC multiplies the last two dimensions and broadcasts the others.</p>`),
    },
    {
      kind: 'check', id: 'c-sum',
      prompt: () => '<p>The tensor <code>x</code> has shape <code>(4, 5)</code>. What is the shape of <code>x.sum(0)</code>?</p>',
      reveal: () => '<p>The shape is <code>(5)</code>. Axis <code>0</code> has four entries, and the sum removes it.</p>',
    },
    {
      kind: 'check', id: 'c-shapes',
      prompt: () => '<p>Can TorC add two tensors of shapes <code>(4, 1)</code> and <code>(3)</code>? If it can, what is the shape of the result?</p>',
      reveal: (D) => { const p = D.files.tensors.broadcast_shapes[0]; return `<p>Yes. From the right, <code>1</code> and <code>3</code> fit because one of them is one. Then <code>4</code> meets a missing dimension, which counts as one. The result has shape <code>(${p.result.join(', ')})</code>.</p>`; },
    },
    {
      kind: 'text',
      html: () => '<section class="tn"><p>The next lesson, <a href="../linear-layer/">Layers</a>, uses <code>matmul</code> and broadcasting to build a layer of a neural network.</p></section>',
    },
  ],
};
