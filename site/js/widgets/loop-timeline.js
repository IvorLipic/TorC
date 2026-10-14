// widgets/loop-timeline.js — the zero_grad trap. A ten-iteration training loop (backward, step, and — if you leave it ticked —
// zero_grad), with a toggle for the last call. For the chosen iteration it shows what W.grad and W.data hold after every call; two charts
// show the gradient step() read in each iteration and the loss, for both versions of the loop.
//
//   mountLoopTimeline(host, D)
// The loop is recomputed in the browser (optim.js trainingLoop, the library's formulas); TorC's own recorded loops (data/optim.json: the
// lr 0.05 run, and no_zero_grad.training_loop) are drawn as open circles / checked by the tests. Pure helpers (narrate, ledger) are exported.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable, ticks } from './plot.js';
import { makeModel, trainingLoop } from '../optim.js';
import { injectStyle, SHARED_CSS, badge, fmt, fitWidth, onResize, enhanceScrollers } from './sgd-ui.js';
import { codePanel } from './sgd-code.js';
import { yRange } from './lr-lab.js';

export const DEFAULT_ITERATION = 3;
const par = (x) => (x < 0 ? `(${fmt(x)})` : fmt(x));

// What happens to W.grad and W.data around each call of iteration `k` (1-based). rows = trainingLoop(...) incl. one extra iteration.
export function ledger(rows, k, zeroGrad, lr) {
  const r = rows[k - 1];
  const leftover = zeroGrad || k === 1 ? null : rows[k - 2].gw;          // null = has_grad() is false
  const wAfter = rows[k].w;
  return {
    leftover, thisPass: r.gw_this_pass, read: r.gw, wBefore: r.w, wAfter, lr,
    cells: {
      start: { grad: leftover, data: r.w },
      afterBackward: { grad: r.gw, data: r.w },
      afterStep: { grad: r.gw, data: wAfter },
      afterZeroGrad: { grad: zeroGrad ? null : r.gw, data: wAfter },
    },
  };
}

export function narrate(L, zeroGrad, k) {
  const parts = [];
  if (L.leftover == null) parts.push(`<code>W.grad</code> is empty when iteration ${k} starts${k === 1 ? '' : ' (the previous <code>zero_grad()</code> cleared it)'}. <code>backward()</code> writes this pass’s gradient, <strong>${fmt(L.thisPass)}</strong>.`);
  else parts.push(`Nothing cleared <code>W.grad</code> after the last <code>step()</code>, so it still holds <strong>${fmt(L.leftover)}</strong>. <code>backward()</code> <em>adds</em> this pass’s ${fmt(L.thisPass)}: ${par(L.leftover)} + ${par(L.thisPass)} = <strong>${fmt(L.read)}</strong>.`);
  parts.push(`<code>step()</code> reads <strong>${fmt(L.read)}</strong> and moves <code>W</code> from ${fmt(L.wBefore)} to ${fmt(L.wAfter)} (${fmt(L.wBefore)} − ${fmt(L.lr)}·${par(L.read)}).`);
  if (Math.sign(L.read) !== Math.sign(L.thisPass)) parts.push(`This pass’s gradient alone says “${L.thisPass < 0 ? 'increase' : 'decrease'} <code>W</code>”, but the leftover outweighs it, so <code>step()</code> moves <code>W</code> the other way.`);
  return parts.join(' ');
}

const CSS = `
.sgd-loop .toggles{display:flex;flex-wrap:wrap;gap:8px 22px;align-items:center;margin:.2rem 0 .6rem}
.sgd-loop .toggles label.chk{display:inline-flex;gap:.5rem;align-items:center;cursor:pointer;font-size:.98rem}
.sgd-loop .toggles input[type=checkbox]{width:1.15rem;height:1.15rem;accent-color:var(--fwd)}
.sgd-loop .pickit{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.sgd-loop .pickit input[type=range]{accent-color:var(--fwd);width:min(220px,50vw)}
.sgd-loop .grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);gap:8px 18px;align-items:start}
.sgd-loop .chart svg{display:block;width:100%;height:auto;max-height:235px}
.sgd-loop .chart text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sgd-loop h4.sub{margin:.5rem 0 .15rem;color:var(--muted);font-size:.85rem}
.sgd-loop .bar-on{fill:var(--bwd);stroke:var(--bwd);stroke-width:1}
.sgd-loop .bar-off{fill:none;stroke:var(--muted);stroke-width:1.4;stroke-dasharray:3 2}
.sgd-loop .band{fill:rgba(255,255,255,.08)}
.sgd-loop .lc-main{fill:none;stroke:var(--fwd);stroke-width:2.6}
.sgd-loop .lc-ghost{fill:none;stroke:var(--muted);stroke-width:1.8;stroke-dasharray:5 4}
.sgd-loop .lc-dot{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.sgd-loop .lc-rec{fill:none;stroke:var(--warn);stroke-width:2}
.sgd-loop .lc-grid{stroke:var(--border);stroke-width:.6}
.sgd-loop .lc-t{fill:var(--muted);font-size:10.5px}
.sgd-loop .lc-l{fill:#c4ccd5;font-size:11px}
.sgd-loop .lc-zero{stroke:var(--fg);stroke-width:1.2}
.sgd-loop .ledger td.kept{color:var(--warn)}
.sgd-loop .ledger th[scope=col]{font-size:.78rem}
.sgd-loop .ledger th[scope=row]{white-space:nowrap}
.sgd-loop .ledger td,.sgd-loop .ledger th[scope=col]{min-width:5.5em}
.sgd-loop .ledger td{white-space:nowrap}
.sgd-loop .narr{font-size:.93rem;margin:.4rem 0 .5rem;color:#d3dae2}
@media(max-width:900px){.sgd-loop .grid2{grid-template-columns:minmax(0,1fr)}}
`;

export function mountLoopTimeline(host, D) {
  injectStyle('sgd-ui-css', SHARED_CSS); injectStyle('sgd-loop-css', CSS);
  const O = D.files.optim;
  const model = makeModel(D.scenes.sgd);
  const lr = O.no_zero_grad.training_loop.lr;
  const N = 10;
  const withZ = trainingLoop(model, { lr, iterations: N + 1, zeroGrad: true });
  const noZ = trainingLoop(model, { lr, iterations: N + 1, zeroGrad: false });
  const recWith = O.runs.find((r) => r.id === 'sgd-lr0.05').steps;                  // TorC: zero_grad() every iteration
  const recNo = O.no_zero_grad.training_loop.iterations;                            // TorC: zero_grad() never
  const state = { zero: true, k: DEFAULT_ITERATION };
  let lastW = 0;

  const root = el('div', { class: 'sgd-w sgd-loop' });
  const chk = el('input', { type: 'checkbox', id: 'sgd-zg', checked: true, 'data-key': 'zg' });
  const range = el('input', { type: 'range', id: 'sgd-it', min: 1, max: N, step: 1, value: state.k, 'data-key': 'it' });
  const itLabel = el('span', { class: 'mono', 'aria-hidden': 'true' });
  const prev = el('button', { type: 'button', class: 'btn small', 'data-key': 'prev', onclick: () => pick(state.k - 1) }, '◀');
  const next = el('button', { type: 'button', class: 'btn small', 'data-key': 'next', 'aria-label': 'Next iteration', onclick: () => pick(state.k + 1) }, '▶');
  prev.setAttribute('aria-label', 'Previous iteration');
  const codeHost = el('div'), narr = el('div', { class: 'narr', 'aria-live': 'polite' }), ledgerHost = el('div', { class: 'sgd-scroll' });
  const barsHost = el('div', { class: 'chart' }), lossHost = el('div', { class: 'chart' }), lossNote = el('p', { class: 'hint', style: 'margin:.1rem 0 .3rem;font-size:.8rem' });
  const left = el('div'), right = el('div');
  left.append(el('h4', { class: 'sub' }, 'The loop body (lines quoted from TorC’s linear_regression example)'), codeHost, narr, ledgerHost);
  right.append(el('h4', { class: 'sub' }, 'The gradient step() read in each iteration'), barsHost, el('p', { class: 'hint', style: 'margin:.1rem 0 .3rem;font-size:.8rem' }, 'Filled bars: the loop as ticked above. Dashed outlines: the other version, for comparison.'));
  const barsTable = srTable(right, { columns: ['iteration', 'W.grad read, with zero_grad()', 'W.grad read, without', 'TorC recorded, with', 'TorC recorded, without'], rows: [], caption: 'Gradient of W that step() reads' });
  right.append(el('h4', { class: 'sub' }, 'Loss at the start of each iteration (log scale)'), lossHost, lossNote);
  const lossTable = srTable(right, { columns: ['iteration', 'loss, with zero_grad()', 'loss, without', 'TorC recorded, with', 'TorC recorded, without'], rows: [], caption: 'Loss at the start of each iteration' });

  const dev = Math.max(...recWith.map((r, i) => Math.abs(r.gw - withZ[i].gw) / (1 + Math.abs(r.gw))), ...recNo.map((r, i) => Math.abs(r.gw - noZ[i].gw) / (1 + Math.abs(r.gw))));
  root.append(
    badge('browser', `Both loops are recomputed live (learning rate ${lr}). TorC’s own recorded loops are the open circles and agree with them (largest relative difference ${dev.toExponential(1)} on the gradient).`),
    el('div', { class: 'toggles' },
      el('label', { class: 'chk', for: 'sgd-zg' }, chk, el('span', {}, 'The loop calls ', el('code', {}, 'optimizer.zero_grad()'), ' at the end of every iteration')),
      el('div', { class: 'pickit' }, el('label', { for: 'sgd-it' }, 'Iteration'), prev, range, next, itLabel)),
    el('div', { class: 'grid2' }, left, right));
  host.append(root);

  const rowsNow = () => (state.zero ? withZ : noZ);

  function drawCode() {
    codeHost.replaceChildren(codePanel({ key: 'loop', ref: O.library.ref, highlight: [], off: state.zero ? [] : [68], notes: state.zero ? {} : { 68: 'removed in this version' } }));
  }

  function drawLedger() {
    const L = ledger(rowsNow(), state.k, state.zero, lr);
    narr.innerHTML = `<strong>Iteration ${state.k}.</strong> ${narrate(L, state.zero, state.k)}`;
    const g = (x) => (x == null ? 'empty' : fmt(x));
    const kept = !state.zero;
    ledgerHost.replaceChildren(el('table', { class: 'sgd-tab ledger' },
      el("caption", {}, `W around each call, iteration ${state.k} (“empty” = has_grad() is false)`),
      el('thead', {}, el('tr', {}, ['', 'iteration starts', 'after backward()', 'after step()', 'after zero_grad()'].map((c) => el('th', { scope: 'col' }, c)))),
      el('tbody', {},
        el('tr', {}, el('th', { scope: 'row' }, el('code', {}, 'W.grad')), el('td', {}, g(L.cells.start.grad)), el('td', {}, g(L.cells.afterBackward.grad)), el('td', {}, g(L.cells.afterStep.grad)), el('td', { class: kept ? 'kept' : '' }, state.zero ? g(null) : `${fmt(L.cells.afterZeroGrad.grad)} (no call: kept)`)),
        el('tr', {}, el('th', { scope: 'row' }, el('code', {}, 'W.data')), ...[L.cells.start, L.cells.afterBackward, L.cells.afterStep, L.cells.afterZeroGrad].map((c) => el('td', {}, fmt(c.data)))))));
  }

  function bars() {
    const W = fitWidth(barsHost, 300, 460, 440), H = Math.round(W * 0.53), m = { l: 52, r: 10, t: 12, b: 38 };
    lastW = W;
    const maxAbs = Math.max(...withZ.slice(0, N).map((r) => Math.abs(r.gw)), ...noZ.slice(0, N).map((r) => Math.abs(r.gw)));
    const R = Math.ceil(maxAbs / 10) * 10;
    barsHost.replaceChildren();
    const p = createPlot(barsHost, { width: W, height: H, x: [0.4, N + 0.6], y: [-R, R], xLabel: 'iteration', yLabel: 'W.grad read by step()', margin: m, ariaLabel: 'Bar chart of the gradient step() read in each iteration' });
    p.axes({ xTicks: Array.from({ length: N }, (_, i) => i + 1), yTicks: ticks([-R, R], 4) });
    const sel = rowsNow(), other = state.zero ? noZ : withZ;
    const bw = 0.62;
    const layer = p.layer('bars');
    layer.append(svg('rect', { class: 'band', x: p.sx(state.k - 0.5), y: m.t, width: p.sx(state.k + 0.5) - p.sx(state.k - 0.5), height: H - m.t - m.b }));
    for (let i = 1; i <= N; i++) {
      const rect = (v, cls) => { const y0 = p.sy(0), y1 = p.sy(v); return svg('rect', { class: cls, x: p.sx(i - bw / 2), y: Math.min(y0, y1), width: p.sx(i + bw / 2) - p.sx(i - bw / 2), height: Math.max(Math.abs(y1 - y0), 0.5) }, svg('title', {}, `iteration ${i}: ${fmt(v)}`)); };
      layer.append(rect(other[i - 1].gw, 'bar-off'), rect(sel[i - 1].gw, 'bar-on'));
    }
    layer.append(svg('line', { class: 'lc-zero', x1: m.l, x2: W - m.r, y1: p.sy(0), y2: p.sy(0) }));
    const r = sel[state.k - 1];
    p.root.append(svg('text', { class: 'lc-l', x: Math.min(Math.max(p.sx(state.k), m.l + 20), W - m.r - 20), y: r.gw < 0 ? p.sy(r.gw) + 14 : p.sy(r.gw) - 5, 'text-anchor': 'middle' }, fmt(r.gw)));
    barsTable.update(Array.from({ length: N }, (_, i) => [i + 1, withZ[i].gw, noZ[i].gw, recWith[i].gw, recNo[i].gw]));
  }

  function loss() {
    const W = fitWidth(lossHost, 300, 460, 440), H = Math.round(W * 0.53), m = { l: 64, r: 12, t: 12, b: 38 };
    const all = [...withZ.slice(0, N), ...noZ.slice(0, N)].map((r) => r.loss);
    const [lo, hi] = yRange(all);
    lossHost.replaceChildren();
    const p = createPlot(lossHost, { width: W, height: H, x: [0.5, N + 0.5], y: [lo, hi], xLabel: 'iteration', yLabel: 'loss', margin: m, ariaLabel: 'Loss at the start of each iteration with and without zero_grad, log scale' });
    p.axes({ xTicks: Array.from({ length: N }, (_, i) => i + 1), yTicks: [] });
    for (let d = lo; d <= hi; d++) p.root.append(svg('line', { class: 'lc-grid', x1: m.l, x2: W - m.r, y1: p.sy(d), y2: p.sy(d) }), svg('text', { class: 'lc-t', x: m.l - 6, y: p.sy(d) + 3.5, 'text-anchor': 'end' }, fmt(10 ** d)));
    const ly = (v) => Math.log10(Math.max(v, 1e-12));
    const series = (rows) => rows.slice(0, N).map((r, i) => [i + 1, ly(r.loss)]);
    const sel = rowsNow(), other = state.zero ? noZ : withZ;
    p.curve(series(other), { class: 'lc-ghost' });
    p.curve(series(sel), { class: 'lc-main' });
    sel.slice(0, N).forEach((r, i) => p.dot(i + 1, ly(r.loss), { class: 'lc-dot', r: 3.4, title: `iteration ${i + 1}: loss ${fmt(r.loss)}` }));
    const rec = state.zero ? recWith : recNo;
    rec.forEach((r, i) => p.dot(i + 1, ly(r.loss), { class: 'lc-rec', r: 6.2, title: `TorC recorded: loss ${fmt(r.loss)}` }));
    lossNote.textContent = state.zero ? 'Solid line: the loop with zero_grad(). Dashed line: the loop without it.' : 'Solid line: the loop without zero_grad(). Dashed line: the loop with it.';
    lossTable.update(Array.from({ length: N }, (_, i) => [i + 1, withZ[i].loss, noZ[i].loss, recWith[i].loss, recNo[i].loss]));
  }

  function pick(k) { state.k = Math.max(1, Math.min(N, k)); range.value = state.k; render(); }
  function render() {
    itLabel.textContent = `${state.k} / ${N}`;
    drawCode(); drawLedger(); bars(); loss();
    enhanceScrollers(root);
    prev.setAttribute('aria-disabled', String(state.k === 1)); next.setAttribute('aria-disabled', String(state.k === N));
  }
  chk.addEventListener('change', () => { state.zero = chk.checked; render(); });
  range.addEventListener('input', () => pick(Number(range.value)));
  const ac = new AbortController();
  onResize(() => { if (Math.abs(fitWidth(barsHost, 300, 460, 440) - lastW) > 8) render(); }, ac.signal);
  render();
  return { destroy() { ac.abort(); root.remove(); }, state };
}
