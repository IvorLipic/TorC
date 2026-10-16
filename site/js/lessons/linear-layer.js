// Lesson "linear-layer": nn::Linear as matmul plus bias. Shapes and parameter counts, one neuron = one row of W, the backward
// formulas, and why the weights start at a particular size. Widgets: matmul-grid (live arithmetic) and init-signal (recorded runs).
// Every number from the real run goes into prose with T('path'), or with sci() for very small/large recorded values.
// Style budget: docs/authoring.md.
import { escapeHtml } from '../core/dom.js';
import { math } from '../core/katex.js';
import { resolvePath } from '../core/trace.js';
import { mountMatmulGrid } from '../widgets/matmul-grid.js';
import { mountInitSignal } from '../widgets/init-signal.js';
import { linearShapes, fmtShape } from '../widgets/linear-shapes.js';
import { fmtSci } from '../widgets/linear-signal-data.js';

const P = 'files.linear';                                  // prefix of the extra tables of linear.json
const WEIGHT = `${P}.param_counts.0.layers.0.params.1`;    // Linear(784, 32): weight record (params are sorted by name: bias, weight)
const BIAS = `${P}.param_counts.0.layers.0.params.0`;
const SIG = (i, layer) => `${P}.signal.${i}.layers.${layer}.post_std`;   // i: 0 kaiming, 1 std 0.01, 2 std 1.0
const recorded = (D, id) => D.files.linear.shape_errors.find((r) => r.id === id);
const ERR = 'linear-784x32-(8, 10)';
// very small or very large recorded numbers in scientific form (T() would print 0.000002173); bound to the trace like T()
const sci = (D, path) => `<span class="rec" data-trace="${path}" data-fmt="sci">${fmtSci(resolvePath(D, path))}</span>`;
const shp = (T, base) => `(${T(`${base}.0`)}, ${T(`${base}.1`)})`;
const eq = (latex) => `<div class="ll-eq">${math(latex, true)}</div>`;
const sec = (id, title, body) => `<section class="ll" aria-labelledby="ll-${id}"><h2 id="ll-${id}">${title}</h2>${body}</section>`;

const STYLE = `<style>
.ll p{max-width:44rem}
.ll h2{margin-top:2rem}
.ll-eq{overflow-x:auto;margin:.2rem 0}
.ll-eq .katex-display{margin:.4em 0}
.rec{color:var(--fwd);font-variant-numeric:tabular-nums}
</style>`;

export default {
  id: 'linear-layer',
  title: 'Layers',
  trace: ['linear.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>Layers</h1>
    <p class="lead">A linear layer, <code>nn::Linear</code>, is one matrix product and one addition. This lesson shows the shapes, the gradients, and the start values of its weights.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        '<code>Linear(in, out)</code> computes <code>y = x·Wᵀ + b</code>. Each row of <code>W</code> belongs to one output, called a neuron.',
        'The layer has <code>in·out + out</code> parameters. The batch size does not change this number.',
        'The backward pass uses the same matrices, transposed.',
        'The weights start as random numbers of a size that keeps the signal steady from layer to layer.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('shapes', '1 · Shapes', `
    <p>The input <code>x</code> holds a batch of <code>B</code> samples, one sample in each row. The weight matrix <code>W</code> has one row for each output and one column for each input. The bias <code>b</code> has one value for each output.</p>
    ${eq('\\underset{(B,\\,\\text{in})}{x}\;\\cdot\;\\underset{(\\text{in},\\,\\text{out})}{W^{\\top}}\;+\;\\underset{(\\text{out})}{b}\;=\;\\underset{(B,\\,\\text{out})}{y}')}
    <p>For example, <code>Linear(784, 32)</code> has a weight of shape ${shp(T, `${WEIGHT}.shape`)} with ${T(`${WEIGHT}.numel`)} numbers and a bias with ${T(`${BIAS}.numel`)} numbers. That is ${T(`${P}.param_counts.0.total`)} parameters. If <code>x</code> has the wrong number of columns, TorC throws an error. This is the message for an input of shape <code>(8, 10)</code>.</p>
    <p><code>${escapeHtml(recorded(D, ERR).exception)}: ${escapeHtml(recorded(D, ERR).what)}</code></p>
    <p>The second shape in the message is the shape of <code>Wᵀ</code>, not of <code>W</code>.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('neuron', '2 · One neuron is one row of W', `
    ${eq('y_{ik} = \\sum_{j} x_{ij}\\,W_{kj} + b_k')}
    <p>Each output <code>y[i,k]</code> multiplies sample <code>i</code> by row <code>k</code> of <code>W</code>, element by element, adds the products, and adds the bias <code>b[k]</code>. Click an output in the grid to see the numbers it uses. You can change any number.</p>
    <p>Then copy the first neuron into the second. The two output columns become identical, because two neurons with equal weights always give equal outputs. For this reason the weights start as random numbers.</p>`),
    },
    {
      kind: 'widget', id: 'matmul-grid', className: 'lin-w lin-mg', label: 'Interactive: one output is one neuron looking at one sample',
      mount: (host, D, ctx) => mountMatmulGrid(host, D, ctx),
    },
    {
      kind: 'text',
      html: () => sec('backward', '3 · The backward pass', `
    <p>In TorC, <code>forward()</code> computes <code>add(matmul(x, transpose(weight)), bias)</code>. A bar over a name means the gradient of the loss with respect to that value. The backward pass of the three operations gives three gradients.</p>
    ${eq('\\overline{W} = \\overline{y}^{\\top}x \\qquad \\overline{b}_k = \\sum_i \\overline{y}_{ik} \\qquad \\overline{x} = \\overline{y}\\,W')}
    <p>The layer adds the same bias to every sample. So the gradient of the bias is the sum of the gradients of all samples. The layer sends ${math('\\overline{x}', false)} back to the layer before it.</p>`),
    },
    {
      kind: 'predict', id: 'predict-dx', title: 'Before you read on',
      prompt: (D, T) => `<p>Use the <code>W</code> of the grid, and let the loss be <code>sum(y)</code>. Then every entry of ${math('\\overline{y}', false)} is one. What is each row of ${math('\\overline{x}', false)}?</p>`,
      reveal: (D, T) => `<p>Each row is the column sums of <code>W</code>, which are ${T('linear-dx.x.grad.0')}, ${T('linear-dx.x.grad.1')} and ${T('linear-dx.x.grad.2')}. Both samples get the same row.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('init', '4 · The start values of the weights', `
    <p>The bias starts at zero. Each weight is a random number from a normal distribution with standard deviation ${math('\\sqrt{2/\\text{in}}', false)}. This is the default, and it is called Kaiming initialization. Each output adds <code>in</code> products, and this scale keeps the size of that sum steady from layer to layer.</p>
    <p>The plot sends random inputs through six <code>Linear(256, 256)</code> layers. After each layer, a ReLU sets the negative values to zero. With Kaiming initialization the spread of the values after six layers is between ${T(`${P}.signal.0.across_seeds.5.post_std_min`)} and ${T(`${P}.signal.0.across_seeds.5.post_std_max`)}. With <code>init_std = 0.01</code> it falls to ${sci(D, SIG(1, 6))}. With <code>init_std = 1.0</code> it grows to ${sci(D, SIG(2, 6))}.</p>`),
    },
    {
      kind: 'widget', id: 'init-signal', className: 'lin-w lin-is', label: 'Signal through six layers',
      mount: (host, D, ctx) => mountInitSignal(host, D, ctx),
    },
    {
      kind: 'check', id: 'check-shapes',
      prompt: () => '<p>You apply <code>Linear(784, 32)</code> to a batch of <code>8</code> samples. What are the shapes of <code>W</code>, <code>b</code>, <code>x</code> and <code>y</code>?</p>',
      reveal: () => { const s = linearShapes(784, 32, 8); return `<p><code>W</code> is <code>${fmtShape(s.W)}</code>, <code>b</code> is <code>${fmtShape(s.b)}</code>, <code>x</code> is <code>${fmtShape(s.x)}</code> and <code>y</code> is <code>${fmtShape(s.y)}</code>.</p>`; },
    },
    {
      kind: 'check', id: 'check-bias',
      prompt: () => '<p>In the grid, increase <code>b[2]</code> by one. Which entries of <code>y</code> change, and by how much?</p>',
      reveal: () => '<p>Only the second column changes. Both of its entries increase by exactly one, because the layer adds <code>b[2]</code> to the second output of every sample.</p>',
    },
    {
      kind: 'text',
      html: () => '<section class="ll"><p>Go back to the <a href="../../">course map</a>.</p></section>',
    },
  ],
};
