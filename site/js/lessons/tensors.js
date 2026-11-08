// Lesson "tensors" (appendix "Using TorC", curriculum v3, template 3): how TorC stores a tensor (row-major, a flat position from
// the index), a 3-D example (a colour image), reshape (a copy), the full broadcasting rules, reductions and axes, the matrix
// product with one worked 2 x 2 example before the index formula, and the batched matrix product. Brief §6 appendix and §7.
// Every number and message comes from data/tensors.json (tools/trace/trace_tensors.cpp) through a role-tagged T() or the recorded
// text. Widgets: tensor-widgets.js. Writing: docs/authoring.md "Writing rules" (ASD-STE100, STE-flavored mode).
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { worked, term } from '../core/blocks.js';
import { mountBroadcastGrid, mountAxisExplorer } from '../widgets/tensor-widgets.js';

const F = 'files.tensors';
const eq = (latex) => `<div class="tn-eq" tabindex="0" role="group" aria-label="Formula">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="tn" aria-labelledby="tn-${id}"><h2 id="tn-${id}">${title}</h2>${body}</section>`;
const err = (r) => `<code>${escapeHtml(r.exception)}: ${escapeHtml(r.what)}</code>`;
const shp = (a) => `<code>(${a.join(', ')})</code>`;
const bc = (D, id) => D.files.tensors.broadcast.find((b) => b.id === id);
const mat = (T, base, n) => `\\begin{bmatrix} ${Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => T.num(`${base}.${i * n + j}`)).join(' & ')).join(' \\\\ ')} \\end{bmatrix}`;

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
  template: 3,
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Tensors</h1>
    <p class="lead">This page is for the reader who wants to use TorC in C++. It shows how TorC stores a ${term('tensor')} and how the ${term('shape', 'shapes')} of two tensors combine. Every message quoted here is a real TorC message.</p></section>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('storage', '1 · Shape and storage', `
    <p>The tensor <code>a</code> has the shape <code>(2, 3)</code>, which means two rows and three columns. Each entry of a shape is the size of one dimension. So <code>a</code> has two dimensions and holds ${T(`${F}.example.numel`, 'setup')} numbers.</p>
    <p>A dimension is named by its position in the shape. Dimension <code>0</code> is the first entry, and dimension <code>1</code> is the second. TorC counts every position from zero.</p>
    <pre class="tn-code" tabindex="0" role="group" aria-label="The tensor a in TorC"><code>Tensor a({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {2, 3});
float v = a[1, 2];     // row 1, column 2</code></pre>
    <p>TorC keeps the numbers in one list, row after row. The entry in row <code>i</code> and column <code>j</code> is at position <code>3·i + j</code>, where <code>3</code> is the number of columns. So <code>a[1, 2]</code> is at position ${T(`${F}.example.index.5.flat`, 'setup')}, and its value is ${T(`${F}.example.index.5.value`, 'setup')}.</p>`),
    },
    {
      kind: 'text',
      html: (D, T) => sec('image', '2 · Three dimensions, and reshape', `
    <p>A colour image has three dimensions. They are the ${term('channel', 'channels')} (red, green, blue), the rows and the columns. The tensor <code>image</code> has the shape ${shp(D.files.tensors.image.shape)} and holds ${T(`${F}.image.numel`, 'setup')} brightness values that we made up.</p>
    <p>The rule for a position works for any number of dimensions. Start at zero. For each dimension, multiply the running position by the size of that dimension, and then add the index. So <code>image[2, 1, 0]</code> is at position <code>(${D.files.tensors.image.index[10].c}·${D.files.tensors.image.shape[1]} + ${D.files.tensors.image.index[10].i})·${D.files.tensors.image.shape[2]} + ${D.files.tensors.image.index[10].j}</code>, which is ${T(`${F}.image.index.10.flat`, 'setup')}, and its value is ${T(`${F}.image.index.10.value`, 'setup')}.</p>
    <p><code>reshape</code> gives the same numbers a new shape, as in <code>a.reshape({3, 2})</code>. In TorC, <code>reshape</code> copies the numbers. A change to the copy does not change <code>a</code>.</p>`),
    },
    {
      kind: 'gotcha', title: 'Why did my reshape fail?',
      html: (D) => `<p>The new shape must hold as many numbers as the old one. TorC reports ${err(D.files.tensors.reshape.error)}.</p>`,
    },
    {
      kind: 'quiz', id: 'p-row', purpose: 'predict', type: 'choice', title: 'Before you read the rule',
      prompt: () => '<p>The tensor <code>a</code> has shape <code>(2, 3)</code>. Your tensor <code>b</code> has shape <code>(2)</code>, one number for each row. What does <code>a.add(b)</code> do?</p>',
      options: [
        { html: 'It adds the first number of <code>b</code> to row <code>0</code> and the second number to row <code>1</code>.', feedback: () => 'TorC does not match the sizes by guessing. It compares the shapes from the right, and the rows are on the left.' },
        { html: 'It stops with a <code>ShapeError</code>.', correct: true, feedback: () => 'Right. The last sizes, <code>3</code> and <code>2</code>, are different and neither is <code>1</code>.' },
        { html: 'It adds both numbers of <code>b</code> to every row.', feedback: () => 'A size repeats only when it is <code>1</code>. A size of <code>2</code> never repeats to fill a size of <code>3</code>.' },
      ],
      reveal: (D) => `<p>The sizes at the right end are <code>${D.files.tensors.example.shape.at(-1)}</code> and <code>${bc(D, 'wrong-row').b.shape.at(-1)}</code>. They differ and neither is one. The next section gives the full rule. To add one number to each row, use the shape ${shp(bc(D, 'column').b.shape)}.</p>`,
    },
    {
      kind: 'text',
      html: () => sec('broadcast', '3 · Broadcasting', `
    <p>An element-wise operation such as <code>add</code>, <code>sub</code> or <code>mul</code> combines the numbers at equal positions. TorC also accepts two different shapes, and the word for this is ${term('broadcasting')}.</p>
    <p>TorC lines the two shapes up from the right. A shape that is shorter gets sizes of 1 on its left. Two sizes fit if they are equal or if one of them is 1. A size of 1 repeats to match the other size. So if one of the two sizes is 1, the result takes the other size.</p>
    <p>If one pair of sizes does not fit, TorC throws a <code>ShapeError</code>. Select a shape for <code>b</code> below, and then select a result entry.</p>`),
    },
    {
      kind: 'widget', id: 'broadcast', className: 'widget', label: 'Interactive: a plus b for several shapes of b',
      mount: (host, D) => mountBroadcastGrid(host, D),
      tasks: () => [
        { do: 'Select the shape <code>b (3)</code>. Then select the result entry in row 1, column 2.', see: 'The entry adds the number of <code>a</code> in row 1, column 2 to the last number of <code>b</code>. The row of <code>b</code> repeats for both rows of <code>a</code>.' },
        { do: 'Select the shape <code>b (2, 1)</code>.', see: 'Each row of <code>a</code> gets one number of <code>b</code>. The column of <code>b</code> repeats across the columns.' },
        { do: 'Select the shape <code>b (2)</code>, and then <code>b (3, 1)</code>.', see: 'TorC reports a <code>ShapeError</code> for both. The table above the grids marks the pair of sizes that does not fit.' },
      ],
    },
    {
      kind: 'gotcha', title: 'Why did my addition fail?',
      html: (D) => `<p>Check the last sizes of both shapes first. For <code>b</code> of shape ${shp(bc(D, 'wrong-row').b.shape)}, TorC reports ${err(bc(D, 'wrong-row').sum)}. To add one number to each row, give <code>b</code> the shape ${shp(bc(D, 'column').b.shape)}, for example with <code>b.reshape({2, 1})</code>.</p>`,
    },
    {
      kind: 'quiz', id: 'q-shapes', purpose: 'transfer', type: 'choice', title: 'Check yourself',
      prompt: (D) => { const p = D.files.tensors.broadcast_shapes[1]; return `<p>A tensor of shape ${shp(p.x)} meets a tensor of shape ${shp(p.y)} in an <code>add</code>. What is the shape of the result?</p>`; },
      options: [
        { html: (D) => shp(D.files.tensors.broadcast_shapes[1].result), correct: true, feedback: () => 'Right. From the right, <code>3</code> meets <code>1</code>, <code>1</code> meets <code>2</code>, and <code>4</code> meets a missing size.' },
        { html: 'None. TorC throws a <code>ShapeError</code>, because the shapes have different numbers of dimensions.', feedback: () => 'A missing size on the left counts as <code>1</code>, so shapes with different numbers of dimensions can fit.' },
        { html: (D) => shp(D.files.tensors.broadcast_shapes[1].x), feedback: () => 'A size of <code>1</code> repeats to match the other size. Here the <code>1</code> in the second position meets a <code>2</code>, so the result has a <code>2</code> there.' },
      ],
      reveal: (D) => `<p>The result has the shape ${shp(D.files.tensors.broadcast_shapes[1].result)}. In each pair, a size of <code>1</code> gives way to the other size, and a missing size counts as <code>1</code>.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('reduce', '4 · Reductions and axes', `
    <p>A reduction combines the numbers along one dimension. The position of that dimension is called its axis. The reduced axis disappears from the shape. TorC has no option to keep it.</p>
    <p><code>a.sum(0)</code> adds down the rows and gives ${shp(D.files.tensors.reductions[0].result.shape)}. <code>a.sum(1)</code> adds across the columns and gives ${shp(D.files.tensors.reductions[3].result.shape)}. <code>mean</code> and <code>max</code> work the same way. <code>a.sum()</code> without an axis gives one number, ${T(`${F}.whole.sum`, 'setup')}.</p>
    <p>For the image, <code>image.sum(0)</code> adds the three channels, so it gives one brightness for each ${term('pixel')}, with the shape ${shp(D.files.tensors.image.reductions[0].result.shape)}. An axis of <code>-1</code> means the last axis.</p>`),
    },
    {
      kind: 'widget', id: 'axis', className: 'widget', label: 'Interactive: sum, mean and max along one axis of a matrix and of an image',
      mount: (host, D) => mountAxisExplorer(host, D),
      tasks: () => [
        { do: 'Choose <code>a</code>, <code>sum</code> and axis <code>0</code>. Select the result entries one by one.', see: 'Each entry adds one column of <code>a</code>. The result has one entry for each column.' },
        { do: 'Choose axis <code>1</code>.', see: 'Each entry now adds one row of <code>a</code>. The result has one entry for each row.' },
        { do: 'Choose <code>image</code>, <code>sum</code> and axis <code>0</code>. Select the first entry.', see: 'The entry adds the brightness of one pixel in the three channels. The channels disappear from the shape.' },
      ],
    },
    {
      kind: 'quiz', id: 'c-sum', purpose: 'check', type: 'choice', title: 'Check yourself',
      prompt: (D) => `<p>The tensor <code>x</code> has the shape ${shp(D.files.tensors.check_sum.x_shape)}. What is the shape of <code>x.sum(0)</code>?</p>`,
      options: [
        { html: (D) => shp(D.files.tensors.check_sum.sum0_shape), correct: true, feedback: () => 'Right. Axis <code>0</code> is removed, and the other size stays.' },
        { html: (D) => `<code>(${D.files.tensors.check_sum.x_shape[0]})</code>`, feedback: () => 'That is the shape of <code>x.sum(1)</code>. The sum along axis <code>0</code> removes the first size, not the second.' },
        { html: (D) => `<code>(1, ${D.files.tensors.check_sum.x_shape[1]})</code>`, feedback: () => 'TorC does not keep the reduced axis with a size of <code>1</code>. The axis disappears.' },
      ],
      reveal: (D) => `<p>The shape is ${shp(D.files.tensors.check_sum.sum0_shape)}. The sum removes axis <code>0</code> from the shape.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('matmul', '5 · The matrix product', `
    <p>The ${term('matrix')} product of two tensors <code>P</code> and <code>Q</code> fills each entry of the result with one row of <code>P</code> times one column of <code>Q</code>. Here is a small example.</p>
    ${eq(`P = ${mat(T, `${F}.matmul_2x2.p.values`, 2)}, \\quad Q = ${mat(T, `${F}.matmul_2x2.q.values`, 2)}`)}
    ${worked(`<p>Row <code>0</code> of <code>P</code> times column <code>0</code> of <code>Q</code> gives ${T(`${F}.matmul_2x2.p.values.0`, 'worked')} × ${T(`${F}.matmul_2x2.q.values.0`, 'worked')} + ${T(`${F}.matmul_2x2.p.values.1`, 'worked')} × ${T(`${F}.matmul_2x2.q.values.2`, 'worked')} = ${T(`${F}.matmul_2x2.c.values.0`, 'worked')}.</p>`)}
    <p>This is the entry in row <code>0</code> and column <code>0</code> of the result.</p>`),
    },
    {
      kind: 'quiz', id: 'n-entry', purpose: 'check', type: 'numeric', title: 'Your turn',
      answer: `${F}.matmul_2x2.c.values.3`, tolerance: 0.01,
      prompt: (D, T) => `<p>Use the same <code>P</code> and <code>Q</code>. Row <code>1</code> of <code>P</code> is ${T(`${F}.matmul_2x2.p.values.2`, 'setup')} and ${T(`${F}.matmul_2x2.p.values.3`, 'setup')}. Column <code>1</code> of <code>Q</code> is ${T(`${F}.matmul_2x2.q.values.1`, 'setup')} and ${T(`${F}.matmul_2x2.q.values.3`, 'setup')}. What is the entry in row <code>1</code> and column <code>1</code> of the result?</p>`,
      reveal: (D, T) => `<p>Row <code>1</code> times column <code>1</code> gives the entry.</p>${worked(`<p>${T(`${F}.matmul_2x2.p.values.2`, 'worked')} × ${T(`${F}.matmul_2x2.q.values.1`, 'worked')} + ${T(`${F}.matmul_2x2.p.values.3`, 'worked')} × ${T(`${F}.matmul_2x2.q.values.3`, 'worked')} = ${T(`${F}.matmul_2x2.c.values.3`, 'worked')}</p>`)}`,
    },
    {
      kind: 'text',
      html: (D) => sec('matmul-rule', 'The index formula and the shapes', `
    ${eq('C_{ij} = \\sum_{l} P_{il}\\,Q_{lj}')}
    <p>A tensor <code>P</code> of shape <code>(m, k)</code> times a tensor <code>Q</code> of shape <code>(k, n)</code> gives the shape <code>(m, n)</code>. The letter <code>l</code> runs from <code>0</code> to <code>k − 1</code>, along the row of <code>P</code> and down the column of <code>Q</code>. So <code>P</code> needs as many columns as <code>Q</code> has rows.</p>
    <p><code>a.transpose({1, 0})</code> swaps the two dimensions of <code>a</code> and gives the shape ${shp(D.files.tensors.matmul.transpose.shape)}. So <code>a.matmul(a.transpose({1, 0}))</code> has the shape ${shp(D.files.tensors.matmul.a_at.shape)}.</p>`),
    },
    {
      kind: 'gotcha', title: 'Why did my matmul fail?',
      html: (D) => `<p>Compare the last size of the first tensor with the second-last size of the second tensor. The call <code>a.matmul(a)</code> reports ${err(D.files.tensors.matmul.a_a)}. The usual repair is <code>a.matmul(a.transpose({1, 0}))</code>, which swaps the two dimensions of the second tensor.</p>`,
    },
    {
      kind: 'text',
      html: (D) => sec('batched', '6 · Batches of matrices', `
    <p>A tensor with more than two dimensions holds a ${term('batch')} of matrices. TorC multiplies only the last two dimensions and treats the others as the batch.</p>
    <p>The batch shapes broadcast as in the broadcasting section. A tensor of shape ${shp(D.files.tensors.matmul.batched_a_shape)} times one of shape ${shp(D.files.tensors.matmul.transpose.shape)} gives ${shp(D.files.tensors.matmul.batched.shape)}, four separate products.</p>`),
    },
    {
      kind: 'quiz', id: 'c-batch', purpose: 'check', type: 'choice', title: 'Check yourself',
      prompt: () => '<p>A tensor of shape <code>(2, 1, 2, 3)</code> meets a tensor of shape <code>(3, 3, 2)</code> in a <code>matmul</code>. What happens?</p>',
      options: [
        { html: (D) => `The result has the shape ${shp(D.files.tensors.matmul.batched_both.shape)}.`, correct: true, feedback: () => 'Right. The batch shapes <code>(2, 1)</code> and <code>(3)</code> broadcast to <code>(2, 3)</code>, and each product has the shape <code>(2, 2)</code>.' },
        { html: 'TorC throws a <code>ShapeError</code>, because the batch shapes <code>(2, 1)</code> and <code>(3)</code> have different lengths.', feedback: () => 'The batch shapes broadcast by the rule of the broadcasting section. A missing size counts as <code>1</code>.' },
        { html: 'The result has the shape <code>(2, 1, 2, 2)</code>, because the first tensor decides the batch.', feedback: () => 'Both tensors have a batch. The <code>1</code> meets a <code>3</code>, so the result has a <code>3</code> there.' },
      ],
      reveal: (D) => `<p>The batch shape is <code>(2, 3)</code>, and each product has the shape <code>(2, 2)</code>. The result has the shape ${shp(D.files.tensors.matmul.batched_both.shape)}.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => `<section class="tn"><p>The next page, <a href="../modules/">Modules</a>, shows how TorC puts ${term('layer', 'layers')} together into one ${term('model')}.</p></section>`,
    },
  ],
};
