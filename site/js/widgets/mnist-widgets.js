// widgets/mnist-widgets.js — the two new widgets of the lesson mnist-mlp. They show only what TorC recorded
// (data/mnist-mlp.json from tools/trace/trace_mnist_mlp.cpp, data/mnist-mistakes.json from tools/trace/trace_mnist_mistakes.cpp):
//
//   mountTrainValidation(host, D)  mistakes per 100 after each pass (epoch) of the validation run (C2): the same network trained on
//                                  55,000 training images, measured on 5,000 of them and on the 5,000 held-out validation images.
//                                  Two curves, labelled on the plot; a slider picks the pass.
//   mountConfusion(host, D)        the 10 x 10 confusion matrix of the final network on the 10,000 test images as a heat map
//                                  (row = true digit, column = the network's answer), and the grid of the first 24 mistakes.
//                                  Select a square (click, or the arrow keys) to see its count and the mistakes of that pair.
//
// Pure helpers are exported for the tests; the module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable } from './plot.js';
import { groupThousands } from '../core/numbers.js';

// mistakes per 100 images for an accuracy (a fraction), as a number
export const per100 = (accuracy) => 100 * (1 - accuracy);
// 2 significant figures for a readout (1.6, 2.8, 13)
export const sig2 = (v) => (Math.abs(v) >= 10 ? String(Math.round(v)) : String(Number(v.toPrecision(2))));
// the curve points of the validation run, passes 1 to 20 (pass 0 is the untrained start, off the scale)
export function trainValidationCurves(M) {
  const E = M.validation_run.epochs.filter((e) => e.epoch >= 1);
  return {
    passes: E.map((e) => e.epoch),
    train: E.map((e) => per100(e.train_subset.accuracy)),
    validation: E.map((e) => per100(e.validation.accuracy)),
  };
}
// the off-diagonal squares of a confusion matrix, largest first: [{t, p, n}]
export function confusions(matrix) {
  const out = [];
  matrix.forEach((row, t) => row.forEach((n, p) => { if (t !== p && n > 0) out.push({ t, p, n }); }));
  return out.sort((a, b) => b.n - a.n || a.t - b.t || a.p - b.p);
}
// the words of a digit, for readouts and labels
export const DIGIT = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const article = (d) => (d === 8 ? 'an' : 'a');
// the readout of one square, in words
export function cellText(matrix, t, p) {
  const n = groupThousands(String(matrix[t][p])), total = groupThousands(String(matrix[t].reduce((a, b) => a + b, 0)));
  if (t === p) return `The network read ${n} of the ${total} test images of ${article(t)} ${DIGIT[t]} correctly.`;
  if (matrix[t][p] === 0) return `The network never read ${article(t)} ${DIGIT[t]} as ${article(p)} ${DIGIT[p]}.`;
  return `The network read ${n} of the ${total} test images of ${article(t)} ${DIGIT[t]} as ${article(p)} ${DIGIT[p]}.`;
}

const CSS = `
.mw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.mw-field{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;margin:.3rem 0}
.mw-field input[type=range]{width:min(14rem,60vw);accent-color:var(--fwd)}
.mw-lbl{color:var(--muted);font-size:.9rem}
.mw-read{font:.92rem ui-monospace,monospace;margin:.4rem 0 0;max-width:40rem}
.mw-train{stroke:var(--warn);stroke-width:2.5;fill:none}
.mw-val{stroke:var(--fwd);stroke-width:2.5;fill:none}
.mw-dtrain{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.mw-dval{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.mw-ltrain{fill:var(--warn);font-size:12px}
.mw-lval{fill:var(--fwd);font-size:12px}
.mw-cm{display:block;width:100%;max-width:400px;height:auto;margin:.2rem 0}
.mw-cm:focus-visible{outline:2px solid var(--fwd);outline-offset:2px}
.mw-cell{stroke:var(--bg);stroke-width:1}
.mw-sel{fill:none;stroke:var(--fg);stroke-width:2.5}
.mw-ax{fill:var(--muted);font-size:12px}
.mw-axt{fill:var(--muted);font-size:12px;font-weight:600}
.mw-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.3rem 0 .4rem}
.mw-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(64px,1fr));gap:6px;max-width:44rem;margin:.6rem 0 0}
.mw-tile{display:flex;flex-direction:column;align-items:center;gap:2px;font:.75rem ui-monospace,monospace;color:var(--muted);background:none;border:2px solid transparent;border-radius:6px;padding:2px;cursor:pointer}
.mw-tile canvas{width:100%;max-width:64px;aspect-ratio:1;image-rendering:pixelated;border-radius:4px;background:#000}
.mw-tile[aria-pressed=true]{border-color:var(--warn);color:var(--fg)}
.mw-grid.dim .mw-tile[aria-pressed=false]{opacity:.35}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('mw-styles')) return;
  document.head.append(el('style', { id: 'mw-styles' }, CSS));
}

// ======================================================================================================================
export function mountTrainValidation(host, D) {
  injectStyles();
  const M = D.files['mnist-mlp'], C = trainValidationCurves(M), last = C.passes.length;
  const top = Math.ceil(Math.max(...C.train, ...C.validation) / 2) * 2;
  let pass = 1;
  const frame = el('div', { class: 'mw' });
  const plotHost = el('div');
  const slider = el('input', { type: 'range', min: 1, max: last, step: 1, value: pass, 'aria-label': 'Epoch' });
  const passLbl = el('span', { class: 'mw-lbl' });
  const read = el('p', { class: 'mw-read', 'aria-live': 'polite' });
  frame.append(el('div', { class: 'mw-field' }, el('label', { class: 'mw-lbl' }, 'Epoch '), slider, passLbl), plotHost, read);
  host.replaceChildren(frame);
  const draw = (p) => {
    p.axes({ xTicks: [1, 5, 10, 15, 20], yTicks: Array.from({ length: top / 2 + 1 }, (_, i) => 2 * i) });
    const pts = (ys) => C.passes.map((x, i) => [x, ys[i]]);
    p.curve(pts(C.train), { class: 'mw-train', layer: 'curves' });
    p.curve(pts(C.validation), { class: 'mw-val', layer: 'curves' });
    // labels on the plot, at the right end of each curve (the training curve ends lower)
    p.text(last, C.train[last - 1], 'training images', { class: 'mw-ltrain', anchor: 'end', dy: 16, layer: 'curves' });
    p.text(last, C.validation[last - 1], 'validation images', { class: 'mw-lval', anchor: 'end', dy: -10, layer: 'curves' });
    dots(p);
  };
  const dots = (p) => {
    const i = pass - 1;
    p.clear('dots');
    p.line(pass, 0, pass, top, { class: 'plot-line', layer: 'dots' });
    p.dot(pass, C.train[i], { class: 'mw-dtrain', layer: 'dots' });
    p.dot(pass, C.validation[i], { class: 'mw-dval', layer: 'dots' });
  };
  const plot = createPlot(plotHost, { width: 640, height: 300, x: [1, last], y: [0, top], xLabel: 'epoch (one pass through the training images)', yLabel: 'mistakes per 100 images', ariaLabel: 'Mistakes per 100 images after each epoch, on training images and on validation images', margin: { r: 20 }, onResize: (p) => draw(p) });
  srTable(frame, { caption: 'Mistakes per 100 images after each epoch', columns: ['epoch', 'training images', 'validation images'], rows: C.passes.map((x, i) => [x, sig2(C.train[i]), sig2(C.validation[i])]), fmt: String });
  const update = (announce) => {
    const i = pass - 1;
    passLbl.textContent = `${pass} of ${last}`;
    slider.value = String(pass);
    dots(plot);
    const text = `After epoch ${pass}, the network makes ${sig2(C.train[i])} mistakes per 100 training images and ${sig2(C.validation[i])} per 100 validation images.`;
    read.textContent = text;
    if (!announce) read.removeAttribute('aria-live'); else read.setAttribute('aria-live', 'polite');
  };
  draw(plot);
  slider.addEventListener('input', () => { pass = Number(slider.value); update(true); });
  update(false);
  return { destroy: () => plot.destroy(), setPass: (k) => { pass = k; update(true); } };
}

// ======================================================================================================================
function paintGray(canvas, pixels) {
  const ctx = canvas.getContext('2d'), img = ctx.createImageData(28, 28);
  pixels.forEach((p, k) => { img.data[4 * k] = img.data[4 * k + 1] = img.data[4 * k + 2] = p; img.data[4 * k + 3] = 255; });
  ctx.putImageData(img, 0, 0);
}

export function mountConfusion(host, D) {
  injectStyles();
  const M = D.files['mnist-mlp'], X = D.files['mnist-mistakes'], mat = M.confusion.matrix;
  const maxOff = Math.max(...confusions(mat).map((c) => c.n));
  let sel = { t: 0, p: 0 };
  // geometry in viewBox units: labels on the left and at the top, 10 x 10 squares
  const L = 46, Tp = 40, S = 34, W = L + 10 * S + 4, H = Tp + 10 * S + 4;
  const cells = [];
  const root = svg('svg', { class: 'mw-cm', viewBox: `0 0 ${W} ${H}`, role: 'group', tabindex: '0', 'aria-label': 'Confusion matrix of the test images. Rows are the true digit, columns are the network’s answer. Use the arrow keys to move between squares.' });
  root.append(svg('text', { class: 'mw-axt', x: L + 5 * S, y: 12, 'text-anchor': 'middle' }, 'the network’s answer'));
  root.append(svg('text', { class: 'mw-axt', transform: `translate(12 ${Tp + 5 * S}) rotate(-90)`, 'text-anchor': 'middle' }, 'true digit'));
  for (let d = 0; d < 10; d++) {
    root.append(svg('text', { class: 'mw-ax', x: L + d * S + S / 2, y: Tp - 6, 'text-anchor': 'middle' }, String(d)));
    root.append(svg('text', { class: 'mw-ax', x: L - 8, y: Tp + d * S + S / 2 + 4, 'text-anchor': 'end' }, String(d)));
  }
  for (let t = 0; t < 10; t++) for (let p = 0; p < 10; p++) {
    const n = mat[t][p];
    // right answers in one muted green; mistakes in amber, brighter for more images (a linear scale, so the largest count stands out)
    const fill = t === p ? 'rgba(131,193,103,.45)' : n === 0 ? 'var(--panel2)' : `rgba(227,179,65,${(0.1 + 0.9 * (n / maxOff)).toFixed(3)})`;
    const r = svg('rect', { class: 'mw-cell', x: L + p * S, y: Tp + t * S, width: S, height: S, fill });
    r.addEventListener('click', () => select(t, p, true));
    cells.push(r);
    root.append(r);
  }
  const ring = svg('rect', { class: 'mw-sel', width: S, height: S });
  root.append(ring);
  const read = el('p', { class: 'mw-read' });
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  const tiles = X.items.map((it) => {
    const c = el('canvas', { width: 28, height: 28, 'aria-hidden': 'true' });
    paintGray(c, it.pixels);
    const b = el('button', { type: 'button', class: 'mw-tile', 'aria-pressed': 'false', 'aria-label': `Test image ${it.index}, ${article(it.label)} ${DIGIT[it.label]} read as ${article(it.predicted)} ${DIGIT[it.predicted]}` }, c, el('span', {}, `${it.label} → ${it.predicted}`));
    b.addEventListener('click', () => select(it.label, it.predicted, true));
    return b;
  });
  const grid = el('div', { class: 'mw-grid', role: 'group', 'aria-label': 'The first mistakes in the test images' }, ...tiles);
  const frame = el('div', { class: 'mw' }, root,
    el('div', { class: 'mw-key', 'aria-hidden': 'true' }, el('span', {}, 'green: right answers'), el('span', {}, 'amber: mistakes, brighter for more images'), el('span', {}, 'dark: no image')),
    read, say,
    el('p', { class: 'mw-lbl' }, 'The first mistakes in the test images. Under each image, the true digit comes first, then the network’s answer.'), grid);
  host.replaceChildren(frame);
  srTable(frame, { caption: 'Confusion matrix: rows are the true digit, columns the network’s answer', columns: ['true digit', ...DIGIT.map((_, d) => `read as ${d}`)], rows: mat.map((row, t) => [t, ...row]), fmt: String });

  function select(t, p, announce) {
    sel = { t, p };
    ring.setAttribute('x', L + p * S);
    ring.setAttribute('y', Tp + t * S);
    const text = cellText(mat, t, p);
    const shown = X.items.filter((it) => it.label === t && it.predicted === p).length;
    read.textContent = t === p ? text : `${text} ${shown ? `The grid below marks ${shown === 1 ? 'one of them' : `${shown} of them`}.` : 'None of them is in the grid below.'}`;
    tiles.forEach((b, i) => b.setAttribute('aria-pressed', String(X.items[i].label === t && X.items[i].predicted === p)));
    grid.classList.toggle('dim', t !== p);
    if (announce) say.textContent = read.textContent;
  }
  root.addEventListener('keydown', (e) => {
    const k = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (!k) return;
    e.preventDefault();
    select(Math.min(9, Math.max(0, sel.t + k[0])), Math.min(9, Math.max(0, sel.p + k[1])), true);
  });
  select(0, 0, false);
  return { select };
}
