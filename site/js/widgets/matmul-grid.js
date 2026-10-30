// widgets/matmul-grid.js — "One neuron = one row of W". The numbers of a small layer (by default the taxi example of linear.json,
// `taxi`: x 2×3, W 2×3, b 2; or the scene `linear` when D has no taxi record) as editable grids. Hover, focus or click an output
// y_ik: row i of x, row k of W and b_k light up and the paper work shows y_ik = Σ_j x_ij W_kj + b_k with the numbers.
//
//   mountMatmulGrid(host, D, ctx) -> { destroy(), state }
//   ctx.example = {x, W, b} (arrays of rows) overrides the data; ctx.rowName / ctx.neuronNames / ctx.inputNames name the rows,
//   the neurons and the inputs in the captions and the screen-reader labels.
//
// Every number is plain arithmetic in linear-math.js; site/test/linear-layer.test.mjs checks the preset against the y of the
// real nn::Linear for the same inputs (taxi.y). Positions count from 0 in the tags and the paper work, as in TorC.
import { el } from '../core/dom.js';
import { math } from '../core/katex.js';
import { fmt } from '../ops/texutil.js';
import { useBaseStyle, injectStyle, matrixEl, clearMarks, markRow, markCell } from './linear-ui.js';
import { exampleFromScene, exampleFromTensors, linearForward, neuronTex, clone2 } from './linear-math.js';

const CSS = `
.lin-mg .mg-cap{margin:.3rem 0 0;font-size:.85rem;color:var(--muted)}
.lin-mg .mg-mats{display:flex;flex-wrap:wrap;gap:14px 10px;align-items:flex-start;margin:.4rem 0}
.lin-mg .lin-op{margin-top:2.6rem}
.lin-mg button.lin-y{background:var(--panel2)}
@media(max-width:640px){.lin-mg .lin-op{margin-top:0;width:100%;text-align:center;font-size:1.2rem}}
`;

export function mountMatmulGrid(host, D, ctx = {}) {
  useBaseStyle();
  injectStyle('lin-matmul-grid-style', CSS);
  host.classList.add('lin-mg');
  const base = ctx.example ?? (D.files?.linear?.taxi ? exampleFromTensors(D.files.linear.taxi) : exampleFromScene(D.scenes.linear));
  const rowName = ctx.rowName ?? 'sample', neuronName = (k) => ctx.neuronNames?.[k] ?? `neuron ${k}`, inputName = (j) => ctx.inputNames?.[j] ?? `input ${j}`;
  const B = base.x.length, nIn = base.x[0].length, nOut = base.W.length;
  const st = { x: clone2(base.x), W: clone2(base.W), b: base.b.slice(), pin: { i: 0, k: 1 }, hover: null };
  const bad = new Set();
  const badNote = el('div', { class: 'lin-err', id: 'mg-err', 'aria-live': 'polite' });
  const MSG = 'Enter a number between −1,000,000 and 1,000,000. The grid keeps the last valid value.';

  const inputCell = (get, set, label) => {
    const inp = el('input', { type: 'number', step: 'any', class: 'lin-cell', value: String(get()), 'aria-label': label, 'aria-describedby': 'mg-err' });
    inp.addEventListener('input', () => {
      const v = parseFloat(inp.value);
      if (Number.isFinite(v) && Math.abs(v) <= 1e6) { bad.delete(inp); inp.removeAttribute('aria-invalid'); inp.removeAttribute('title'); if (!bad.size) badNote.textContent = ''; set(v); update(); }
      else { bad.add(inp); inp.setAttribute('aria-invalid', 'true'); inp.title = MSG; badNote.textContent = MSG; }
    });
    inp.addEventListener('change', () => { if (bad.has(inp)) { bad.delete(inp); inp.removeAttribute('aria-invalid'); inp.value = String(get()); if (!bad.size) badNote.textContent = ''; } });
    return inp;
  };

  const mx = matrixEl({ caption: `<code>x</code> · one row per ${rowName}`, rows: B, cols: nIn, rowTag: (i) => `i=${i}`, colTag: (j) => `j=${j}`,
    cell: (i, j) => inputCell(() => st.x[i][j], (v) => { st.x[i][j] = v; }, `x, row ${i} (${rowName} ${i}), column ${j} (${inputName(j)})`) });
  const mw = matrixEl({ caption: '<code>W</code> · one row per <b>neuron</b>', rows: nOut, cols: nIn, rowTag: (k) => `k=${k}`, colTag: (j) => `j=${j}`,
    cell: (k, j) => inputCell(() => st.W[k][j], (v) => { st.W[k][j] = v; }, `W, row ${k} (${neuronName(k)}), weight of ${inputName(j)}`) });
  const mb = matrixEl({ caption: '<code>b</code> · one bias per neuron', rows: 1, cols: nOut, colTag: (k) => `k=${k}`,
    cell: (_, k) => inputCell(() => st.b[k], (v) => { st.b[k] = v; }, `b, position ${k} (${neuronName(k)})`) });
  const my = matrixEl({ caption: '<code>y</code> · outputs. Choose one.', rows: B, cols: nOut, rowTag: (i) => `i=${i}`, colTag: (k) => `k=${k}`,
    cell: (i, k) => {
      const b = el('button', { type: 'button', class: 'lin-cell lin-y', 'data-key': `mg-y-${i}-${k}` });
      b.addEventListener('mouseenter', () => { st.hover = { i, k }; update(); });
      b.addEventListener('mouseleave', () => { st.hover = null; update(); });
      b.addEventListener('focus', () => { st.pin = { i, k }; update(); });
      b.addEventListener('click', () => { st.pin = { i, k }; update(); });
      return b;
    } });

  const paper = el('div', { class: 'lin-paper' });
  const status = el('p', { class: 'lin-sr', 'aria-live': 'polite' });

  const refresh = () => {                                    // after buttons set many values at once
    for (let i = 0; i < B; i++) for (let j = 0; j < nIn; j++) mx.cells[i][j].value = String(st.x[i][j]);
    for (let k = 0; k < nOut; k++) { for (let j = 0; j < nIn; j++) mw.cells[k][j].value = String(st.W[k][j]); mb.cells[0][k].value = String(st.b[k]); }
    for (const inp of bad) inp.removeAttribute('aria-invalid');
    bad.clear(); badNote.textContent = '';
    update();
  };
  const buttons = el('div', { class: 'lin-btns' },
    el('button', { type: 'button', class: 'btn small', onclick: () => { st.x = clone2(base.x); st.W = clone2(base.W); st.b = base.b.slice(); refresh(); status.textContent = 'The grid shows the start values again.'; } }, 'Reset'));

  host.replaceChildren(
    el('div', { class: 'mg-mats' }, mx.root, el('span', { class: 'lin-op', 'aria-hidden': 'true' }, '·'), mw.root, el('span', { class: 'lin-op', 'aria-hidden': 'true' }, '+'), mb.root, el('span', { class: 'lin-op', 'aria-hidden': 'true' }, '='), my.root),
    el('p', { class: 'mg-cap' }, 'Each output yᵢₖ uses row i of x and row k of W, plus bₖ. Positions count from 0.'),
    badNote, paper, buttons, status);

  function update() {
    const y = linearForward(st.x, st.W, st.b);
    const act = st.hover ?? st.pin;
    for (let i = 0; i < B; i++) for (let k = 0; k < nOut; k++) {
      const c = my.cells[i][k];
      c.textContent = fmt(y[i][k]);
      c.setAttribute('aria-label', `y, row ${i} (${rowName} ${i}), ${neuronName(k)}: ${fmt(y[i][k])}`);
    }
    for (const m of [mx, mw, mb, my]) clearMarks(m);
    markRow(mx, act.i); markRow(mw, act.k); markCell(mb, 0, act.k); mb.colTags[act.k]?.classList.add('hl');
    markCell(my, act.i, act.k); my.rowTags[act.i]?.classList.add('hl'); my.colTags[act.k]?.classList.add('hl');
    paper.innerHTML = math(neuronTex(st.x, st.W, st.b, act.i, act.k), true);
  }

  update();
  return { destroy() {}, state: st };
}
