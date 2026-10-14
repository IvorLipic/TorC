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
  title: 'nn::Linear: a layer as matmul plus bias',
  trace: ['linear.json'],
  sections: [
    {
      kind: 'text',
      html: () => `${STYLE}<section class="hero"><h1>nn::Linear: a layer is a matmul plus a bias</h1>
    <p class="lead">One matrix product, one addition, and weights that start at the right size.</p></section>`,
    },
    {
      kind: 'tldr',
      items: () => [
        '<code>Linear(in, out)</code> computes <code>y = x·Wᵀ + b</code>: <code>W</code> is (out, in), one row per neuron; <code>b</code> is (out).',
        'Parameters: <code>in·out + out</code>, whatever the batch size.',
        `Backward: ${math('\\overline{W} = \\overline{y}^{\\top}x', false)}, ${math('\\overline{b}', false)} = column sums of ${math('\\overline{y}', false)}, ${math('\\overline{x} = \\overline{y}\\,W', false)}.`,
        'Weights start normal with standard deviation <code>√(2/in)</code>, the bias at zero: the signal keeps its size layer after layer.',
      ],
    },
    {
      kind: 'text',
      html: (D, T) => sec('shapes', '1 · Shapes', `
    ${eq('\\underset{(B,\\,\\text{in})}{x}\\;\\cdot\\;\\underset{(\\text{in},\\,\\text{out})}{W^{\\top}}\\;+\\;\\underset{(\\text{out})}{b}\\;=\\;\\underset{(B,\\,\\text{out})}{y}')}
    <p><code>Linear(784, 32)</code> has a weight of shape ${shp(T, `${WEIGHT}.shape`)}, ${T(`${WEIGHT}.numel`)} numbers, plus ${T(`${BIAS}.numel`)} biases: ${T(`${P}.param_counts.0.total`)} parameters. Feed it an <code>x</code> of shape <code>(8, 10)</code> and TorC throws <code>${escapeHtml(recorded(D, ERR).exception)}: ${escapeHtml(recorded(D, ERR).what)}</code>. The second shape is <code>Wᵀ</code>, not <code>W</code>.</p>`),
    },
    {
      kind: 'text',
      html: () => sec('neuron', '2 · One neuron is one row of W', `
    ${eq('y_{ik} = \\sum_{j} x_{ij}\\,W_{kj} + b_k')}
    <p>Pick an output: its row of <code>x</code>, its row of <code>W</code> and its bias light up. Edit any number. Then copy the first neuron into the second: the two output columns become identical. Equal weights make twin neurons, which is why weights start random.</p>`),
    },
    {
      kind: 'widget', id: 'matmul-grid', className: 'lin-w lin-mg', label: 'Interactive: one output is one neuron looking at one sample',
      mount: (host, D, ctx) => mountMatmulGrid(host, D, ctx),
    },
    {
      kind: 'text',
      html: () => sec('backward', '3 · Backward: transposes and column sums', `
    <p><code>forward()</code> is <code>add(matmul(x, transpose(weight)), bias)</code>. Run backwards, the three operations give:</p>
    ${eq('\\overline{W} = \\overline{y}^{\\top}x \\qquad \\overline{b}_k = \\sum_i \\overline{y}_{ik} \\qquad \\overline{x} = \\overline{y}\\,W')}
    <p>The bias was copied to every sample, so its gradient sums over the batch. ${math('\\overline{x}', false)} is what the layer hands to the layer before it.</p>`),
    },
    {
      kind: 'predict', id: 'predict-dx', title: 'Predict: the gradient of the input',
      prompt: (D, T) => `<p>With the grid's <code>W</code> and the loss <code>sum(y)</code>, every entry of ${math('\\overline{y}', false)} is one. What is each row of ${math('\\overline{x}', false)}?</p>`,
      reveal: (D, T) => `<p>The column sums of <code>W</code>: [${T('linear-dx.x.grad.0')}, ${T('linear-dx.x.grad.1')}, ${T('linear-dx.x.grad.2')}], the same for both samples.</p>`,
    },
    {
      kind: 'text',
      html: (D, T) => sec('init', '4 · Start the weights at the right size', `
    <p>The bias starts at zero. Each weight is drawn from a normal distribution with standard deviation ${math('\\sqrt{2/\\text{in}}', false)} (Kaiming, the default). Each output sums <code>in</code> products, so the weights' size sets how much the signal grows per layer; ${math('\\sqrt{2/\\text{in}}', false)} cancels that, and the factor <code>2</code> makes up for ReLU zeroing about half the values. Through six <code>Linear(256, 256)</code> + ReLU layers the spread ends at ${T(SIG(0, 6))} with Kaiming, fades to ${sci(D, SIG(1, 6))} at <code>init_std = 0.01</code> and explodes to ${sci(D, SIG(2, 6))} at <code>init_std = 1.0</code>.</p>`),
    },
    {
      kind: 'widget', id: 'init-signal', className: 'lin-w lin-is', label: 'Signal through six layers',
      mount: (host, D, ctx) => mountInitSignal(host, D, ctx),
    },
    {
      kind: 'check', id: 'check-shapes',
      prompt: () => '<p><code>Linear(784, 32)</code> on a batch of <code>8</code>: what are the shapes of <code>W</code>, <code>b</code>, <code>x</code> and <code>y</code>?</p>',
      reveal: () => { const s = linearShapes(784, 32, 8); return `<p><code>W</code> <code>${fmtShape(s.W)}</code>, <code>b</code> <code>${fmtShape(s.b)}</code>, <code>x</code> <code>${fmtShape(s.x)}</code>, <code>y</code> <code>${fmtShape(s.y)}</code>.</p>`; },
    },
    {
      kind: 'check', id: 'check-bias',
      prompt: () => '<p>In the grid, raise <code>b[2]</code> by one. Which entries of <code>y</code> change, and by how much?</p>',
      reveal: () => '<p>Only the second column, in both rows, by exactly one: the bias is added to every sample\'s answer of the second neuron. Try it.</p>',
    },
    {
      kind: 'text',
      html: () => '<section class="ll"><p>Back to the <a href="../../">course map</a>.</p></section>',
    },
  ],
};
