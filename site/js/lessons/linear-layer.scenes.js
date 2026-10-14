// linear-layer.scenes.js — layout and teaching text of the two graph-lab scenes of the `linear-layer` lesson, plus the few
// DERIVED numbers the prose quotes (plain arithmetic on recorded numbers). Values, gradients and edges come from data/linear.json
// (real TorC run) and from engine.js; no result number is typed here. Strings here never contain numerals the trace could change.

import { resolvePath } from '../core/trace.js';
import { fmtSci } from '../widgets/linear-signal-data.js';

export const SCENES = {
  linear: {
    tab: '1 · Linear = three ops',
    view: [1120, 290], minWidth: 940,
    above: ['x', 'b', 't'],
    pos: { W: [145, 215], Wt: [430, 215], xWt: [680, 215], y: [895, 215], L: [1030, 215], x: [680, 75], b: [895, 75], t: [1030, 75] },
    goal: 'What does nn::Linear compute, and what gradients do its parameters get?',
    intro: '<code>nn::Linear</code> hides three operations: <code>transpose</code> turns the (out, in) weight into (in, out), <code>matmul</code> multiplies the batch by it, and <code>add</code> adds the bias row to every sample. Here they are as separate nodes, followed by an <code>mse</code> loss so the backward pass has a starting point. <code>W</code> and <code>b</code> are the learnable parameters (solid outline); <code>x</code> and <code>t</code> are data (dashed outline).',
    forwardHint: 'Forward: transpose, then matmul, then add, then the loss.',
    backwardHint: 'Backward: the loss gives <code>y</code> its gradient; <code>add</code> splits it between <code>xWt</code> and <code>b</code> (adding up the samples for <code>b</code>); <code>matmul</code> and <code>transpose</code> carry it on to <code>W</code>.',
    takeaway: '<code>b</code> was copied to every sample, so its gradient is the <strong>sum over the samples</strong> of the gradient at <code>y</code>. <code>W</code> receives <code>dYᵀ x</code>: matmul sends <code>xᵀ dY</code> back to <code>Wt</code>, and transpose turns it round into <code>W</code>’s (out, in) shape. <code>x</code> is data, so nothing is computed for it.',
  },
  'linear-dx': {
    tab: '2 · Gradient to the input',
    view: [1120, 290], minWidth: 940,
    above: ['x', 'b'],
    pos: { W: [145, 215], Wt: [430, 215], xWt: [680, 215], y: [895, 215], S: [1030, 215], x: [680, 75], b: [895, 75] },
    goal: 'If the input needs a gradient, what is it?',
    intro: 'The same three operations with a different loss, <code>sum(y)</code>, and with <code>x</code> marked trainable, as if it were the output of an earlier layer. With a sum every output has a gradient of exactly one, so the arithmetic is easy to follow.',
    forwardHint: 'Forward: transpose, matmul, add, then the sum.',
    backwardHint: 'Backward: a unit gradient enters at <code>y</code>; <code>matmul</code> sends <code>dY · W</code> to <code>x</code>.',
    takeaway: 'Every row of <code>x</code>’s gradient equals the <strong>column sums of <code>W</code></strong>: <code>dX = dY · W</code> with <code>dY</code> all ones. This is the gradient a layer hands to the layer before it.',
  },
};

// ---- derived numbers quoted in prose -------------------------------------------------------------------------------------
// Each entry computes a number from recorded trace values. site/test/linear-layer.test.mjs recomputes every one independently.
const sig = (D, name) => D.files.linear.signal.find((s) => s.name === name);
const ratios = (D, name) => {
  const L = sig(D, name).layers.filter((l) => l.layer >= 1);
  return L.slice(1).map((l, i) => l.post_std / L[i].post_std);
};

export const CALCS = {
  // b[2] raised by 1: the second column of y moves by exactly 1 (y[1,2] and y[2,2] of the printed output, plus 1)
  'edit-y12': (D) => D.scenes.linear.nodes.find((n) => n.name === 'y').value.values[1] + 1,
  'edit-y22': (D) => D.scenes.linear.nodes.find((n) => n.name === 'y').value.values[3] + 1,
  // layer-to-layer growth factor of the activations' std (layers 1 -> 2 ... 5 -> 6) for init_std 1.0 and 0.01
  'grow-lo': (D) => Math.min(...ratios(D, 'std_1.0')),
  'grow-hi': (D) => Math.max(...ratios(D, 'std_1.0')),
  'shrink-lo': (D) => Math.min(...ratios(D, 'std_0.01')),
  'shrink-hi': (D) => Math.max(...ratios(D, 'std_0.01')),
  // expected factor sqrt(width/2) * init_std  (Expected mathematics, see the derivation in docs/lessons/linear-layer.md)
  'grow-pred': (D) => Math.sqrt(sig(D, 'std_1.0').width / 2) * sig(D, 'std_1.0').init_std,
  // N = number of outputs of y (batch x out) = the divisor of the mean squared error
  'n-outputs': (D) => D.scenes.linear.nodes.find((n) => n.name === 'y').value.shape.reduce((a, b) => a * b, 1),
  // expected std of ReLU(z) for z ~ N(0, 2): sqrt(1 - 1/pi)
  'relu-std': () => Math.sqrt(1 - 1 / Math.PI),
  // sqrt(2/width): the Kaiming std of Linear(width, width)
  'kaiming-scale': (D) => Math.sqrt(2 / sig(D, 'kaiming').width),
  'shrink-pred': (D) => Math.sqrt(sig(D, 'std_0.01').width / 2) * sig(D, 'std_0.01').init_std,
};

// Very small and very large recorded numbers in scientific form (the shared T() would print 0.000002173 and 2173000).
export function makeSci(D) {
  return (path) => {
    const v = resolvePath(D, path);
    if (typeof v !== 'number') throw new Error(`sci("${path}"): not a number`);
    return `<span class="rec" data-trace="${path}" data-fmt="sci" style="color:var(--fwd);font-variant-numeric:tabular-nums">${fmtSci(v)}</span>`;
  };
}

export const calcText = (v) => String(Number(v.toPrecision(3)));
export function makeCalc(D) {
  return (key) => {
    if (!Object.hasOwn(CALCS, key)) throw new Error(`unknown derived number "${key}"`);
    return `<span class="calc" data-calc="${key}" style="color:var(--warn);font-variant-numeric:tabular-nums" title="Derived: plain arithmetic on numbers recorded from TorC">${calcText(CALCS[key](D))}</span>`;
  };
}
