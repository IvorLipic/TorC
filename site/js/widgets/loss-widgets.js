// widgets/loss-widgets.js — the two widgets of the lesson losses ("Softmax and cross-entropy"). Both show only what TorC
// recorded (data/losses.json, tools/trace/trace_losses.cpp):
//
//   mountLossExplorer(host, D)    one photo, three fruit classes (apple, banana, cherry = classes 0, 1, 2): a row of scores and the
//                                 right class; the softmax shares and the cross-entropy (the surprise at the right class). Rows:
//                                 the `ce` rows (any right class) and the two `spread` rows (the apple is right; the same apple
//                                 share, the rest spread evenly or unevenly)
//   mountGradientCompare(host, D) two classes, scores [z, 0], the first class is right: the gradient for z of cross-entropy and of
//                                 the squared miss between the shares and the target shares, for z from -6 to 6
//
// Readouts use 2-3 significant figures (sig3) and whole percents (pct). The module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable, ticks } from './plot.js';

export const FRUITS = ['apple', 'banana', 'cherry'];
const MINUS = '−';
// 3 significant figures, the minus sign U+2212, no trailing zeros
export const sig3 = (v) => { const r = Number(v.toPrecision(3)); return (Object.is(r, -0) || r === 0 ? '0' : String(r)).replace('-', MINUS); };
// a share as a percent: whole percents from 1 % up, two significant figures below (0.24 %), never a false 0 % or 100 %
export const pct = (p) => {
  const x = 100 * p;
  if (x >= 99.5 && x < 100) return `${Number(x.toPrecision(3))}%`;
  return x >= 1 ? `${Math.round(x)}%` : `${Number(x.toPrecision(2))}%`;
};

// The rows the explorer offers: {key, label, logits, cases: [{target, loss, softmax}] (one per right class it allows)}.
export function explorerRows(L) {
  const fromCe = L.ce.map((r) => ({ key: r.id, label: r.logits.map(sig3).join(', '), logits: r.logits, cases: r.cases }));
  const fromSpread = L.spread.map((r) => ({ key: r.id, label: `${r.logits.map(sig3).join(', ')} (same apple share)`, logits: r.logits, cases: [{ target: r.target, loss: r.loss, softmax: r.softmax }] }));
  return [...fromCe, ...fromSpread];
}

const CSS = `
.lw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.lw-ctl{display:flex;flex-direction:column;gap:.45rem;margin:0 0 .6rem}
.lw-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.lw-lbl{color:var(--muted);font-size:.9rem;min-width:6.5rem}
.lw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.lw .btn[disabled]{opacity:.4}
.lw-cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;max-width:34rem}
.lw-col{min-width:0;background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.5rem .6rem;font:.85rem ui-monospace,monospace}
.lw-col.target{border-color:var(--ok)}
.lw-h{margin:0 0 .3rem;font:600 .85rem system-ui,sans-serif;color:var(--fg)}
.lw-row{display:flex;flex-wrap:wrap;justify-content:space-between;gap:0 .5rem}
.lw-row b{color:var(--muted);font-weight:600;font-family:system-ui,sans-serif}
.lw-bar{height:.5rem;border-radius:4px;background:var(--fwd);margin:.15rem 0 .1rem;min-width:2px}
.lw-read{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font:.92rem ui-monospace,monospace;margin:.6rem 0 0}
.lw-read b{color:var(--muted);font-weight:600;margin-right:.3rem;font-family:system-ui,sans-serif}
.lw-field{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}
.lw-field input[type=range]{width:min(16rem,100%);accent-color:var(--fwd)}
.lw-ce{stroke:var(--fwd);stroke-width:2.4;fill:none}
.lw-sq{stroke:var(--warn);stroke-width:2.4;stroke-dasharray:6 4;fill:none}
.lw-dot2{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.lw-zero{stroke:var(--muted);stroke-width:1}
.lw-lab{font:600 12px system-ui,sans-serif;paint-order:stroke;stroke:var(--bg);stroke-width:3px}
.lw-lab.ce{fill:var(--fwd)}
.lw-lab.sq{fill:var(--warn)}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('lw-styles')) return;
  document.head.append(el('style', { id: 'lw-styles' }, CSS));
}
function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  const root = el('div', { class: 'lw-presets', role: 'group', 'aria-label': label }, el('span', { class: 'lw-lbl', 'aria-hidden': 'true' }, label), ...btns);
  return { root, btns, set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}

// ======================================================================================================================
export function mountLossExplorer(host, D) {
  injectStyles();
  const rows = explorerRows(D.files.losses);
  let row = 'mild', target = 0;
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const rp = presets('scores', rows.map((r) => [r.key, r.label]), (k) => { row = k; if (!caseOf()) target = current().cases[0].target; draw(); announce(); });
  const tp = presets('right class', FRUITS.map((f, i) => [i, f]), (k) => { target = k; draw(); announce(); });
  const cols = el('div', { class: 'lw-cols', role: 'group', 'aria-label': 'The three classes' });
  const read = el('div', { class: 'lw-read' });
  host.replaceChildren(el('div', { class: 'lw' }, el('div', { class: 'lw-ctl' }, rp.root, tp.root), cols, read, status));
  const current = () => rows.find((r) => r.key === row);
  const caseOf = () => current().cases.find((c) => c.target === target);

  function draw() {
    const r = current(), c = caseOf();
    rp.set(row); tp.set(target);
    tp.btns.forEach((b, i) => { b.disabled = !r.cases.some((x) => x.target === i); });
    cols.replaceChildren(...r.logits.map((z, j) => el('div', { class: `lw-col${j === target ? ' target' : ''}` },
      el('div', { class: 'lw-h' }, `${FRUITS[j]}${j === target ? ' (right)' : ''}`),
      el('div', { class: 'lw-row' }, el('b', {}, 'score'), sig3(z)),
      el('div', { class: 'lw-row' }, el('b', {}, 'share'), pct(c.softmax[j])),
      el('div', { class: 'lw-bar', style: `width:${(100 * c.softmax[j]).toFixed(1)}%`, 'aria-hidden': 'true' }))));
    read.innerHTML = `<span><b>share of the right class</b> ${pct(c.softmax[target])}</span><span><b>surprise (the loss), lower is better</b> ${sig3(c.loss)}</span>`;
  }
  function announce() {
    const r = current(), c = caseOf();
    status.textContent = `Scores ${r.label}. Shares ${FRUITS.map((f, j) => `${f} ${pct(c.softmax[j])}`).join(', ')}. The right class is the ${FRUITS[target]}, and the surprise is ${sig3(c.loss)}.`;
  }
  draw();
  return { draw, choose: (k, t) => { row = k; if (t !== undefined) target = t; draw(); } };
}

// ======================================================================================================================
export function mountGradientCompare(host, D) {
  injectStyles();
  const C = D.files.losses.curves, z = C.z;
  let k = 0;
  const id = `lw-z${Math.random().toString(36).slice(2, 7)}`;
  const input = el('input', { type: 'range', id, min: 0, max: z.length - 1, step: 1, value: k });
  input.addEventListener('input', () => { k = Number(input.value); mark(); });
  const box = el('div'), read = el('div', { class: 'lw-read' });   // the slider's aria-valuetext carries the values
  host.replaceChildren(el('div', { class: 'lw' }, el('div', { class: 'lw-ctl' }, el('div', { class: 'lw-field' }, el('label', { for: id, class: 'lw-lbl' }, 'score of the right class'), input)), box, read));
  srTable(host, { columns: ['score of the right class', 'share of the right class', 'cross-entropy gradient', 'squared miss gradient'], rows: z.map((v, i) => [v, C.p[i], C.ce_grad[i], C.sq_grad[i]]), caption: 'The gradient for the score of the right class, two classes' });
  const paint = (pl) => {
    pl.axes({ xTicks: ticks([-6, 6], 6), yTicks: [-1, -0.75, -0.5, -0.25, 0] });
    pl.layer('curves').append(svg('line', { class: 'lw-zero', x1: pl.sx(-6), x2: pl.sx(6), y1: pl.sy(0), y2: pl.sy(0) }));
    pl.curve(z.map((v, i) => [v, C.ce_grad[i]]), { class: 'lw-ce', layer: 'curves' });
    pl.curve(z.map((v, i) => [v, C.sq_grad[i]]), { class: 'lw-sq', layer: 'curves' });
    pl.text(-5.8, -0.86, 'cross-entropy', { class: 'lw-lab ce', layer: 'curves' });
    pl.text(-5.8, -0.1, 'squared miss', { class: 'lw-lab sq', layer: 'curves', dy: 4 });
    pl.layer('live');
  };
  const p = createPlot(box, { width: 640, height: 280, x: [-6, 6], y: [-1.05, 0.08], xLabel: 'score of the right class', yLabel: 'gradient', ariaLabel: '', onResize: (pl) => { paint(pl); mark(); } });
  paint(p);

  function mark() {
    p.clear('live');
    p.dot(z[k], C.ce_grad[k], { r: 6, layer: 'live' });
    p.dot(z[k], C.sq_grad[k], { class: 'lw-dot2', r: 5, layer: 'live' });
    const text = `score ${sig3(z[k])}, share of the right class ${pct(C.p[k])}: cross-entropy gradient ${sig3(C.ce_grad[k])}, squared miss gradient ${sig3(C.sq_grad[k])}`;
    input.setAttribute('aria-valuetext', text);
    p.root.setAttribute('aria-label', `The gradient of two losses for the score of the right class. At the dots, ${text}.`);
    read.innerHTML = `<span><b>share of the right class</b> ${pct(C.p[k])}</span><span><b>gradient, cross-entropy</b> ${sig3(C.ce_grad[k])}</span><span><b>gradient, squared miss</b> ${sig3(C.sq_grad[k])}</span>`;
  }
  mark();
  return { mark, set: (i) => { k = i; input.value = i; mark(); } };
}
