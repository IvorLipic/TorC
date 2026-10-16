// widgets/tensor-widgets.js — the two widgets of the lesson tensors. Both show only what TorC recorded (data/tensors.json):
//
//   mountBroadcastGrid(host, D)   a (2, 3) plus b of a chosen shape: the shapes lined up from the right, the grids, the result
//                                 or TorC's ShapeError. Selecting a result cell marks the cell of a and the cell of b it used.
//   mountAxisExplorer(host, D)    sum, mean or max of a along axis 0 or 1: which cells each result combines.
//
// Pure helpers (alignShapes, sourceIndex, reducedCells) are exported for the tests. The module touches the DOM only in mount*.
import { el } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';

// Shapes lined up from the right, as TorC compares them. Returns rows of equal length (null = missing dimension) and, per
// column, whether the two sizes fit (equal, or one of them is 1, or one is missing).
export function alignShapes(x, y) {
  const n = Math.max(x.length, y.length);
  const pad = (s) => [...Array(n - s.length).fill(null), ...s];
  const a = pad(x), b = pad(y);
  const fits = a.map((d, i) => d === null || b[i] === null || d === b[i] || d === 1 || b[i] === 1);
  const out = a.map((d, i) => (fits[i] ? Math.max(d ?? 1, b[i] ?? 1) : null));
  return { a, b, fits, out, ok: fits.every(Boolean) };
}

// The flat index in a tensor of shape `shape` that feeds output position `idx` (an index list of the output shape) under
// broadcasting: dimensions are matched from the right, a size-1 dimension always reads index 0.
export function sourceIndex(shape, idx) {
  const off = idx.length - shape.length;
  let flat = 0;
  shape.forEach((d, k) => { flat = flat * d + (d === 1 ? 0 : idx[off + k]); });
  return flat;
}

// The cells (as [row, col]) of a 2-D tensor that result `r` of a reduction along `axis` combines.
export function reducedCells(rows, cols, axis, r) {
  return axis === 0 ? Array.from({ length: rows }, (_, i) => [i, r]) : Array.from({ length: cols }, (_, j) => [r, j]);
}

const shapeText = (s) => `(${s.join(', ')})`;
const LABELS = { same: 'b (2, 3)', row: 'b (3)', row2d: 'b (1, 3)', column: 'b (2, 1)', scalar: 'b (1)', 'wrong-row': 'b (2)', 'wrong-column': 'b (3, 1)' };

function grid(name, rows, cols, values, cls = '') {
  const cells = [];
  const g = el('div', { class: `tw-grid ${cls}`, style: `grid-template-columns:repeat(${cols}, auto)`, role: 'group', 'aria-label': `${name}, shape ${shapeText([rows, cols])}` });
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const c = el('span', { class: 'tw-cell' }, fmt(values[i * cols + j]));
    cells.push(c); g.append(c);
  }
  return { root: el('div', { class: 'tw-mat' }, el('div', { class: 'tw-name' }, name), g), cells };
}
// a 1-D or 2-D shape drawn as rows x cols (a vector of n is one row of n)
const as2d = (s) => (s.length === 1 ? [1, s[0]] : s);

function presetGroup(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'tw-presets', role: 'group', 'aria-label': label }, ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}

const CSS = `
.tw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.tw-presets{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 .7rem}
.tw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.tw-row{display:flex;flex-wrap:wrap;gap:.8rem 1rem;align-items:center}
.tw-op{font-size:1.3rem;color:var(--muted)}
.tw-mat{display:flex;flex-direction:column;gap:.25rem}
.tw-name{font:600 .8rem ui-monospace,monospace;color:var(--muted)}
.tw-grid{display:grid;gap:4px;width:max-content}
.tw-cell{min-width:2.6rem;padding:.3rem .45rem;text-align:center;font:.95rem ui-monospace,monospace;background:var(--panel2);border:1px solid var(--border);border-radius:6px}
button.tw-cell{color:var(--fg);cursor:pointer}
.tw-cell.hl{border-color:var(--fwd);background:rgba(88,196,221,.18)}
.tw-cell.src{border-color:var(--warn);background:rgba(227,179,65,.16)}
.tw-shapes{font:.92rem ui-monospace,monospace;border-collapse:collapse;margin:0 0 .8rem}
.tw-shapes th{color:var(--muted);font-weight:600;text-align:left;padding:.1rem .8rem .1rem 0}
.tw-shapes td{text-align:center;min-width:2rem;padding:.1rem .3rem}
.tw-shapes td.bad{color:var(--bwd);font-weight:700}
.tw-err{border:1px solid var(--bwd);border-radius:8px;padding:.5rem .8rem;font:.9rem ui-monospace,monospace;color:#ffb3ab;background:#2b1513;max-width:100%;overflow-wrap:anywhere}
.tw-read{margin:.7rem 0 0;font-size:.92rem;color:#c4ccd5;min-height:1.4em}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('tw-styles')) return;
  document.head.append(el('style', { id: 'tw-styles' }, CSS));
}

// ======================================================================================================================
export function mountBroadcastGrid(host, D) {
  injectStyles();
  const T = D.files.tensors, A = T.example;
  let pick = 'row', sel = null;
  const presets = presetGroup('Shape of b', T.broadcast.map((c) => [c.id, LABELS[c.id] ?? shapeText(c.b.shape)]), (k) => { pick = k; sel = null; draw(); });
  const shapes = el('table', { class: 'tw-shapes' });
  const row = el('div', { class: 'tw-row' });
  // The visible readout follows the mouse; the screen-reader message changes only on focus or click, so moving the mouse
  // across the grid does not make a screen reader speak again and again.
  const read = el('p', { class: 'tw-read' });
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  const root = el('div', { class: 'tw' }, presets.root, shapes, row, read, say);
  host.replaceChildren(root);

  function draw() {
    const c = T.broadcast.find((x) => x.id === pick);
    presets.set(pick);
    const al = alignShapes(A.shape, c.b.shape);
    const tr = (name, cells, bad = []) => el('tr', {}, el('th', { scope: 'row' }, name), ...cells.map((d, i) => el('td', { class: bad[i] ? 'bad' : '' }, d === null ? '' : String(d))));
    shapes.replaceChildren(el('caption', { class: 'sr' }, 'The two shapes lined up from the right'),
      el('tbody', {}, tr('a', al.a), tr('b', al.b, al.fits.map((f) => !f)), tr('a + b', al.out.map((d, i) => (al.fits[i] ? d : '✗')), al.fits.map((f) => !f))));
    const ga = grid('a', ...as2d(A.shape), A.values);
    const gb = grid('b', ...as2d(c.b.shape), c.b.values);
    const parts = [ga.root, el('span', { class: 'tw-op', 'aria-hidden': 'true' }, '+'), gb.root, el('span', { class: 'tw-op', 'aria-hidden': 'true' }, '=')];
    if (c.sum.exception !== 'none') {
      row.replaceChildren(...parts, el('div', { class: 'tw-err', role: 'note' }, `${c.sum.exception}: ${c.sum.what}`));
      read.textContent = 'TorC cannot add these shapes.';
      say.textContent = read.textContent;
      return;
    }
    const [R, C] = c.sum.shape;
    const gy = el('div', { class: 'tw-grid', style: `grid-template-columns:repeat(${C}, auto)`, role: 'group', 'aria-label': `a + b, shape ${shapeText(c.sum.shape)}` });
    const ys = [];
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      const b = el('button', { type: 'button', class: 'tw-cell', 'aria-label': `a + b, row ${i}, column ${j}: ${fmt(c.sum.values[i * C + j])}` }, fmt(c.sum.values[i * C + j]));
      const choose = (announce) => () => { sel = [i, j]; mark(announce); };
      b.addEventListener('mouseenter', choose(false)); b.addEventListener('focus', choose(true)); b.addEventListener('click', choose(true));
      ys.push(b); gy.append(b);
    }
    row.replaceChildren(...parts, el('div', { class: 'tw-mat' }, el('div', { class: 'tw-name' }, 'a + b'), gy));
    function mark(announce = true) {
      [...ga.cells, ...gb.cells, ...ys].forEach((x) => x.classList.remove('hl', 'src'));
      if (!sel) read.textContent = 'Select a result to see which numbers of a and b it adds. Rows and columns count from 0.';
      else {
        const [i, j] = sel, ia = sourceIndex(A.shape, [i, j]), ib = sourceIndex(c.b.shape, [i, j]);
        ys[i * C + j].classList.add('hl'); ga.cells[ia].classList.add('src'); gb.cells[ib].classList.add('src');
        read.textContent = `Row ${i}, column ${j} adds ${fmt(A.values[ia])} from a and ${fmt(c.b.values[ib])} from b.`;
      }
      if (announce) say.textContent = read.textContent;
    }
    mark();
  }
  draw();
  return { draw };
}

// ======================================================================================================================
export function mountAxisExplorer(host, D) {
  injectStyles();
  const T = D.files.tensors, A = T.example, [R, C] = A.shape;
  let op = 'sum', axis = 0, sel = 0;
  const ops = presetGroup('Reduction', [['sum', 'sum'], ['mean', 'mean'], ['max', 'max']], (k) => { op = k; draw(); });
  const axes = presetGroup('Axis', [[0, 'axis 0 (down the rows)'], [1, 'axis 1 (across the columns)']], (k) => { axis = k; sel = 0; draw(); });
  const row = el('div', { class: 'tw-row' });
  const read = el('p', { class: 'tw-read' });
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'tw' }, ops.root, axes.root, row, read, say));

  function draw() {
    ops.set(op); axes.set(axis);
    const res = T.reductions.find((x) => x.op === op && x.axis === axis).result;
    const ga = grid('a', R, C, A.values);
    const n = res.shape[0];
    const gr = el('div', { class: 'tw-grid', style: `grid-template-columns:repeat(${axis === 0 ? n : 1}, auto)`, role: 'group', 'aria-label': `a.${op}(${axis}), shape ${shapeText(res.shape)}` });
    const rs = res.values.map((v, r) => {
      const b = el('button', { type: 'button', class: 'tw-cell', 'aria-label': `a.${op}(${axis}), entry ${r}: ${fmt(v)}` }, fmt(v));
      const choose = (announce) => () => { sel = r; mark(announce); };
      b.addEventListener('mouseenter', choose(false)); b.addEventListener('focus', choose(true)); b.addEventListener('click', choose(true));
      gr.append(b);
      return b;
    });
    row.replaceChildren(ga.root, el('span', { class: 'tw-op', 'aria-hidden': 'true' }, '→'), el('div', { class: 'tw-mat' }, el('div', { class: 'tw-name' }, `a.${op}(${axis})  shape ${shapeText(res.shape)}`), gr));
    function mark(announce = true) {
      [...ga.cells, ...rs].forEach((x) => x.classList.remove('hl', 'src'));
      rs[sel].classList.add('hl');
      const cells = reducedCells(R, C, axis, sel);
      cells.forEach(([i, j]) => ga.cells[i * C + j].classList.add('src'));
      read.textContent = `Entry ${sel} is the ${op} of ${cells.map(([i, j]) => fmt(A.values[i * C + j])).join(', ')}, which is ${fmt(res.values[sel])}.`;
      if (announce) say.textContent = read.textContent;
    }
    mark();
  }
  draw();
  return { draw };
}
