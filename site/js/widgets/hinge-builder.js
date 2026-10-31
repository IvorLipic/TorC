// widgets/hinge-builder.js — the hinge builder of the lesson nonlinearity (brief §6 Part 2 and §9 "aha" widgets): each hidden ReLU
// neuron of the trained model is a hinge; switch neurons on and off and see their sum change shape. Data: data/nonlinearity.json
// `hinges` (the trained relu model after 400 steps: units[k].contribution = W2_k · relu(W1_k x + b1_k) on data.grid from the real
// ops, derived.kink_x, b2, sum_plus_b2) and `data` (the 21 points of t = x²).
//
//   hingeSum(N, on)        b2 + the contributions of the neurons in `on` (a Set of unit indices), on the grid (pure, tested: with
//                          every neuron on it equals sum_plus_b2, which TorC checked against the model's prediction)
//   typicalMiss(N, ys)     √(mean of the squared misses) of a curve on the grid at the 21 data points (pure, tested: with every
//                          neuron on its square equals the recorded loss of the trained model)
//   mountHingeBuilder(host, D) -> { destroy(), set(on) }
import { el } from '../core/dom.js';
import { createPlot, srTable, ticks } from './plot.js';

const sig2 = (v) => (Math.abs(v) < 0.005 ? '0' : Number(v.toPrecision(2)).toString());

export function hingeSum(N, on) {
  const H = N.hinges;
  return N.data.grid.map((_, i) => H.b2 + H.units.reduce((s, u) => s + (on.has(u.unit) ? u.contribution[i] : 0), 0));
}

// index of each data point on the grid (both are built from integers: data x = k/5 - 2, grid = k/10 - 2.2)
export function dataOnGrid(N) {
  return N.data.x.map((x) => {
    const i = N.data.grid.findIndex((g) => Math.abs(g - x) < 1e-6);
    if (i < 0) throw new Error(`data point x = ${x} is not on the grid`);
    return i;
  });
}

export function typicalMiss(N, ys) {
  const idx = dataOnGrid(N);
  return Math.sqrt(idx.reduce((s, gi, k) => s + (ys[gi] - N.data.t[k]) ** 2, 0) / idx.length);
}

const CSS = `
.hb{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.hb-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 .5rem}
.hb-ctl .hb-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.hb .btn[aria-pressed=true]{border-color:var(--warn);color:var(--warn);background:rgba(227,179,65,.12)}
.hb-read{margin:.4rem 0 0;font-size:.95rem}
.hb-unit{stroke:var(--warn);stroke-width:1.6;stroke-dasharray:5 4;fill:none;opacity:.9}
.hb-sum{stroke:var(--fwd);stroke-width:3;fill:none}
.hb-kink{fill:var(--warn)}
.hb-data{fill:var(--fg);stroke:var(--bg);stroke-width:1.5}
.hb-tsum{fill:var(--fwd);font-size:12px;font-weight:700}
.hb-tunit{fill:var(--warn);font-size:12px;font-weight:700}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('hb-styles')) return;
  document.head.append(el('style', { id: 'hb-styles' }, CSS));
}

export function mountHingeBuilder(host, D) {
  injectStyles();
  const N = D.files.nonlinearity, H = N.hinges, grid = N.data.grid;
  let on = new Set(H.units.map((u) => u.unit));
  const unitBtns = H.units.map((u) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'true', onclick: () => { const s = new Set(on); if (s.has(u.unit)) s.delete(u.unit); else s.add(u.unit); set(s); } }, `neuron ${u.unit}`));
  const status = el('p', { class: 'lin-sr', 'aria-live': 'polite', style: 'position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)' });
  const all = el('button', { type: 'button', class: 'btn small', onclick: () => { set(new Set(H.units.map((u) => u.unit))); status.textContent = 'All eight neurons are on.'; } }, 'All on');
  const none = el('button', { type: 'button', class: 'btn small', onclick: () => { set(new Set()); status.textContent = 'All neurons are off. The sum is a flat line at the bias of the output neuron.'; } }, 'All off');
  const box = el('div');
  const read = el('p', { class: 'hb-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'hb' },
    el('div', { class: 'hb-ctl', role: 'group', 'aria-label': 'Hidden neurons' }, el('span', { class: 'hb-lbl', 'aria-hidden': 'true' }, 'Hidden neurons'), ...unitBtns),
    el('div', { class: 'hb-ctl' }, all, none), box, read, status));
  const table = srTable(host, { columns: ['x', 'sum'], rows: [], caption: 'The sum of the switched-on neurons and the output bias' });

  const paint = (pl) => {
    pl.axes({ xTicks: ticks([-2, 2], 4), yTicks: [0, 1, 2, 3, 4] });
    const ys = hingeSum(N, on);
    for (const u of H.units) {
      if (!on.has(u.unit)) continue;
      pl.curve(grid.map((x, i) => [x, u.contribution[i]]), { class: 'hb-unit', layer: 'units' });
      const kx = u.derived.kink_x;
      if (kx > grid[0] && kx < grid.at(-1)) pl.dot(kx, 0, { class: 'hb-kink', r: 3.5, layer: 'units', title: `kink of neuron ${u.unit}` });
    }
    N.data.x.forEach((x, i) => pl.dot(x, N.data.t[i], { class: 'hb-data', r: 3.5, layer: 'data' }));
    pl.curve(grid.map((x, i) => [x, ys[i]]), { class: 'hb-sum', layer: 'sum' });
    pl.text(grid.at(-1), ys.at(-1), 'sum', { class: 'hb-tsum', anchor: 'end', dx: -4, dy: ys.at(-1) > 4 ? 16 : -8, layer: 'sum' });
    if (on.size === 1) {
      const u = H.units.find((v) => on.has(v.unit));
      const right = u.w1 > 0, i = right ? grid.length - 1 : 0;
      pl.text(grid[i], u.contribution[i], `neuron ${u.unit}`, { class: 'hb-tunit', anchor: right ? 'end' : 'start', dx: right ? -4 : 4, dy: 18, layer: 'units' });
    }
    return ys;
  };
  let plot = null;
  function set(s) {
    on = s;
    unitBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(on.has(H.units[k].unit))));
    plot?.destroy();
    box.replaceChildren();
    plot = createPlot(box, { width: 640, height: 300, x: [-2.2, 2.2], y: [-0.6, 4.6], xLabel: 'x', yLabel: 'value', ariaLabel: '', onResize: paint });
    const ys = paint(plot);
    const miss = typicalMiss(N, ys);
    plot.root.setAttribute('aria-label', `The points of t = x squared. ${on.size} hidden neurons are dashed hinges, and their sum with the output bias is a thick line.`);
    read.textContent = `${on.size} of ${H.units.length} neurons are on. The sum misses the points by about ${sig2(miss)} on average.`;
    table.update(grid.map((x, i) => [x, ys[i]]));
  }
  set(on);
  return { destroy() { plot?.destroy(); }, set };
}
