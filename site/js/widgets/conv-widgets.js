// widgets/conv-widgets.js — the two widgets of the lesson convolution. Both show what TorC recorded (data/conv.json,
// tools/trace/trace_conv.cpp):
//
//   mountSlidingWindow(host, D)   a 5 x 5 input with zero padding, a 3 x 3 kernel and the output for three settings of stride and
//                                 padding; pick an output to see its window and its nine products
//   mountFeatureMaps(host, D)     an MNIST digit and the feature maps of two edge kernels, before or after relu
//
// The products in the first widget are recomputed here from the recorded input and kernel; the tests check that their sum is
// the output TorC recorded. Pure helpers are exported for the tests; the module touches the DOM only in mount*.
import { el } from '../core/dom.js';

// ---- pure helpers ---------------------------------------------------------------------------------------------------------
// the n x n input with p rows and columns of zeros around it: (n + 2p) x (n + 2p), row-major
export function padded(values, n, p) {
  const m = n + 2 * p, out = new Array(m * m).fill(0);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[(y + p) * m + x + p] = values[y * n + x];
  return out;
}
export const outSize = (n, k, stride, p) => Math.floor((n + 2 * p - k) / stride) + 1;
// the window of output (i, j): the cells of the padded grid it covers, and the products with the kernel (k x k, row-major)
export function windowAt(values, n, kernel, k, stride, p, i, j) {
  const m = n + 2 * p, grid = padded(values, n, p), cells = [], products = [];
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) {
    const y = i * stride + a, x = j * stride + b;
    cells.push(y * m + x);
    products.push([grid[y * m + x], kernel[a * k + b]]);
  }
  return { cells, products, sum: products.reduce((s, [v, w]) => s + v * w, 0) };
}

const CSS = `
.cw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.cw-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 .7rem}
.cw-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.cw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.cw-row{display:flex;flex-wrap:wrap;gap:.8rem 1.2rem;align-items:flex-start}
.cw-mat{display:flex;flex-direction:column;gap:.25rem}
.cw-name{font:600 .8rem ui-monospace,monospace;color:var(--muted)}
.cw-grid{display:grid;gap:3px;width:max-content}
.cw-cell{min-width:2.1rem;padding:.25rem .2rem;text-align:center;font:.85rem ui-monospace,monospace;background:var(--panel2);border:1px solid var(--border);border-radius:5px;color:var(--fg)}
.cw-cell.pad{color:var(--muted);border-style:dashed;background:transparent}
.cw-cell.cw-win{border-color:var(--warn);background:rgba(227,179,65,.16)}
button.cw-cell{cursor:pointer}
.cw-cell.sel{border-color:var(--fwd);background:rgba(88,196,221,.22);font-weight:700}
.cw-read{font:.88rem ui-monospace,monospace;color:#c4ccd5;margin:.7rem 0 0;overflow-wrap:anywhere}
.cw-maps{display:flex;flex-wrap:wrap;gap:1rem 1.4rem;align-items:flex-start}
.cw-maps canvas{width:168px;height:168px;image-rendering:pixelated;border:1px solid var(--border);border-radius:6px;background:#000}
.cw-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.5rem 0 0}
@media(max-width:560px){.cw-cell{min-width:1.7rem;font-size:.78rem}.cw-maps canvas{width:140px;height:140px}}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('cw-styles')) return;
  document.head.append(el('style', { id: 'cw-styles' }, CSS));
}
function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'cw-presets', role: 'group', 'aria-label': label }, el('span', { class: 'cw-lbl', 'aria-hidden': 'true' }, label), ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}
const fmt = (v) => String(Math.round(v * 1000) / 1000);

// ======================================================================================================================
export function mountSlidingWindow(host, D) {
  injectStyles();
  const E = D.files.conv.example, n = E.input.shape[2], k = E.kernel.shape[2];
  let run = 0, sel = [0, 0];
  const pr = presets('Setting', E.runs.map((r, i) => [i, `stride ${r.stride}, padding ${r.padding}`]), (i) => { run = i; sel = [0, 0]; draw(); });
  const row = el('div', { class: 'cw-row' });
  const read = el('p', { class: 'cw-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'cw' }, pr.root, row, read));

  function draw() {
    pr.set(run);
    const R = E.runs[run], p = R.padding, m = n + 2 * p, out = R.output.shape[2];
    const grid = padded(E.input.values, n, p);
    const inCells = grid.map((v, idx) => {
      const y = Math.floor(idx / m), x = idx % m, isPad = y < p || y >= m - p || x < p || x >= m - p;
      return el('div', { class: `cw-cell${isPad ? ' pad' : ''}`, 'aria-hidden': 'true' }, String(v));
    });
    const kCells = E.kernel.values.map((v) => el('div', { class: 'cw-cell', 'aria-hidden': 'true' }, String(v)));
    const outCells = R.output.values.map((v, idx) => {
      const i = Math.floor(idx / out), j = idx % out;
      const b = el('button', { type: 'button', class: 'cw-cell', 'aria-label': `output row ${i}, column ${j}: ${fmt(v)}` }, fmt(v));
      b.addEventListener('click', () => { sel = [i, j]; mark(); });
      b.addEventListener('focus', () => { sel = [i, j]; mark(); });
      return b;
    });
    const mat = (name, cells, cols, label) => el('div', { class: 'cw-mat' }, el('div', { class: 'cw-name' }, name), el('div', { class: 'cw-grid', style: `grid-template-columns:repeat(${cols}, auto)`, role: 'group', 'aria-label': label }, ...cells));
    row.replaceChildren(
      mat(p ? `input with padding ${p}` : 'input', inCells, m, 'The input'),
      mat('kernel', kCells, k, 'The kernel'),
      mat(`output ${out} × ${out}`, outCells, out, 'The output. Select a value to see its window.'));
    function mark() {
      inCells.forEach((c) => c.classList.remove('cw-win'));
      outCells.forEach((c) => c.classList.remove('sel'));
      const [i, j] = sel, w = windowAt(E.input.values, n, E.kernel.values, k, R.stride, p, i, j);
      w.cells.forEach((c) => inCells[c].classList.add('cw-win'));
      outCells[i * out + j].classList.add('sel');
      read.textContent = `Output (${i}, ${j}) = ${w.products.map(([v, wt]) => `${v}·${wt < 0 ? `(${wt})` : wt}`).join(' + ')} = ${fmt(w.sum)}.`;
    }
    mark();
  }
  draw();
  return { draw };
}

// ======================================================================================================================
export function mountFeatureMaps(host, D) {
  injectStyles();
  const G = D.files.conv.digit, side = G.shape[2];
  let id = G.maps[0].id, useRelu = false;
  const kp = presets('Kernel', G.maps.map((m) => [m.id, `${m.id} edges`]), (v) => { id = v; draw(); });
  const rp = presets('Output', [[false, 'conv2d'], [true, 'relu(conv2d)']], (v) => { useRelu = v; draw(); });
  const img = el('canvas', { width: side, height: side, role: 'img', 'aria-label': `The input, an MNIST test image of the digit ${G.label}` });
  const map = el('canvas', { width: side, height: side, role: 'img' });
  const kgrid = el('div', { class: 'cw-grid', style: 'grid-template-columns:repeat(3, auto)', 'aria-hidden': 'true' });
  const key = el('div', { class: 'cw-key', 'aria-hidden': 'true' }, el('span', {}, 'amber: a positive output'), el('span', {}, 'blue: a negative output'), el('span', {}, 'dark: near zero'));
  host.replaceChildren(el('div', { class: 'cw' }, el('div', { class: 'cw-row' }, kp.root, rp.root),
    el('div', { class: 'cw-maps' }, el('div', { class: 'cw-mat' }, el('div', { class: 'cw-name' }, 'input'), img),
      el('div', { class: 'cw-mat' }, el('div', { class: 'cw-name' }, 'kernel'), kgrid),
      el('div', { class: 'cw-mat' }, el('div', { class: 'cw-name' }, 'feature map'), map)), key));
  const paint = (canvas, values, signed) => {
    const ctx = canvas.getContext('2d'), im = ctx.createImageData(side, side), m = Math.max(...values.map(Math.abs)) || 1;
    values.forEach((v, q) => {
      let rgb;
      if (!signed) { const g = Math.round(255 * v); rgb = [g, g, g]; } else {
        const t = Math.min(1, Math.abs(v) / m), [R, Gc, B] = v >= 0 ? [227, 179, 65] : [88, 196, 221];
        rgb = [14 + t * (R - 14), 17 + t * (Gc - 17), 22 + t * (B - 22)].map(Math.round);
      }
      im.data.set([...rgb, 255], 4 * q);
    });
    ctx.putImageData(im, 0, 0);
  };
  paint(img, G.pixels, false);
  function draw() {
    kp.set(id); rp.set(useRelu);
    const M = G.maps.find((x) => x.id === id), values = useRelu ? M.relu : M.output;
    kgrid.replaceChildren(...M.kernel.map((v) => el('div', { class: 'cw-cell' }, String(v))));
    paint(map, values, true);
    map.setAttribute('aria-label', `The feature map of the ${id}-edge kernel${useRelu ? ' after relu' : ''}, ${side} by ${side}. It is strongest where the digit has ${id} edges.`);
  }
  draw();
  return { draw };
}
