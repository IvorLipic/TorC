// widgets/two-bowls.js — the steep-bowl refutation of the lesson why-gradients (brief C3, Appendix B): the two-notebook loss
// (2w − 6)² and the four-notebook loss (4w − 12)² on one plot, with the same bottom at w = 3. Choose a weight, and both bowls show
// their gradient there next to the distance to the bottom: the same distance, a gradient four times as large. Every value is
// recorded by TorC (data/why-gradients.json `grid`, `steep.grid`, `steep.compare`); nothing is computed here but the drawing.
// site/test/why-gradients.test.mjs checks the rows. The DOM is touched only in mountTwoBowls.
import { el } from '../core/dom.js';
import { createPlot, ticks } from './plot.js';
import { fmt3 } from './landscape-tex.js';

const X = [-0.5, 6.5], Y = [0, 160];

// The weights the learner can choose, with both gradients, both losses and the distance to the bottom (steep.compare).
export function bowlRows(files) {
  return files.steep.compare.map((c) => ({ w: c.w, distance: c.distance, grad: c.grad, steepGrad: c.steep_grad, L: c.L, steepL: c.steep_L, ratio: c.grad_ratio }));
}

// The loss on a recorded curve at weight w, by straight-line interpolation between grid points (only used to place a label).
export function lossOn(pts, w) {
  const i = Math.max(1, pts.findIndex(([x]) => x >= w));
  const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
  return y0 + ((y1 - y0) * (w - x0)) / (x1 - x0);
}

let seq = 0;

export function mountTwoBowls(host, D) {
  injectStyles();
  const files = D.files['why-gradients'];
  const rows = bowlRows(files);
  const wMin = files.setup.w_min;
  const shallow = files.grid.map((q) => [q.w, q.L]), steep = files.steep.grid.map((q) => [q.w, q.L]);
  const id = `bowls${++seq}`;
  let k = 0;

  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const btns = rows.map((r, i) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { k = i; draw(); announce(); } }, fmt3(r.w)));
  const controls = el('div', { class: 'bowls-ctl', role: 'group', 'aria-labelledby': `${id}-w` }, el('span', { class: 'bowls-lbl', id: `${id}-w` }, 'your weight w'), ...btns);
  const plotBox = el('div', { class: 'bowls-plot' });
  const readout = el('div', { class: 'bowls-read' });
  host.replaceChildren(controls, plotBox, readout, status);
  const p = createPlot(plotBox, { width: 640, height: 300, x: X, y: Y, xLabel: 'w', yLabel: 'loss', ariaLabel: 'Two loss curves with the same bottom.', onResize: () => draw() });

  function draw() {
    const r = rows[k];
    btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === k)));
    for (const n of ['curves', 'live']) p.clear(n);
    p.root.querySelectorAll('.plot-axes').forEach((g) => g.remove());
    p.axes({ xTicks: ticks(X, 7), yTicks: [0, 40, 80, 120, 160] });
    p.curve(shallow, { class: 'plot-curve', layer: 'curves' });
    p.curve(steep, { class: 'plot-curve bowls-steep', layer: 'curves' });
    p.line(wMin, Y[0], wMin, Y[1] * 0.86, { class: 'bowls-bottom', layer: 'curves' });
    p.text(wMin, Y[1] * 0.86, 'bottom of both', { class: 'plot-text bowls-label', anchor: 'middle', dy: -5, layer: 'curves' });
    p.text(-0.3, lossOn(shallow, -0.3), 'two notebooks', { class: 'plot-text bowls-label', dx: 2, dy: -12, layer: 'curves' });
    p.text(0.75, lossOn(steep, 0.75), 'four notebooks', { class: 'plot-text bowls-label bowls-label-steep', dx: 10, dy: 4, layer: 'curves' });
    // the chosen weight: a dot and the tangent on each bowl, and the distance to the bottom along the axis
    p.tangent(r.w, r.L, r.grad, { half: 0.7, layer: 'live' });
    p.tangent(r.w, r.steepL, r.steepGrad, { half: 0.35, layer: 'live', class: 'plot-tangent bowls-steep-t' });
    p.dot(r.w, r.L, { r: 6, layer: 'live' });
    p.dot(r.w, r.steepL, { r: 6, layer: 'live', class: 'plot-dot bowls-steep-dot' });
    p.line(r.w, 4, wMin, 4, { class: 'bowls-dist', layer: 'live' });
    p.root.setAttribute('aria-label', `Two loss curves with the same bottom at w = ${fmt3(wMin)}. At w = ${fmt3(r.w)} the distance to the bottom is ${fmt3(r.distance)}. The gradient is ${fmt3(r.grad)} for two notebooks and ${fmt3(r.steepGrad)} for four notebooks.`);
    readout.innerHTML = `<span><b>distance to the bottom</b> ${fmt3(r.distance)}</span>`
      + `<span><b>gradient, two notebooks</b> ${fmt3(r.grad)}</span>`
      + `<span class="bowls-steep-txt"><b>gradient, four notebooks</b> ${fmt3(r.steepGrad)}</span>`;
  }
  function announce() {
    const r = rows[k];
    status.textContent = `w is ${fmt3(r.w)}. The distance to the bottom is ${fmt3(r.distance)}. The gradient is ${fmt3(r.grad)} for two notebooks and ${fmt3(r.steepGrad)} for four notebooks.`;
  }
  draw();
  return { choose: (i) => btns[i].click() };
}

const CSS = `
.bowls-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:.2rem 0 .4rem}
.bowls-lbl{color:var(--muted);font-size:.9rem;margin-right:.3rem}
.bowls-read{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;align-items:baseline;font-family:ui-monospace,monospace;font-size:.92rem;margin:.3rem 0 0}
.bowls-read b{color:var(--muted);font-weight:600;margin-right:.3rem;font-family:system-ui,sans-serif}
.bowls-steep{stroke:var(--warn)}
.bowls-steep-t{stroke:var(--warn)}
.bowls-steep-dot{fill:var(--warn)}
.bowls-steep-txt{color:var(--warn)}
.bowls-bottom{stroke:var(--muted);stroke-dasharray:4 4}
.bowls-dist{stroke:var(--fg);stroke-width:2.5}
.bowls-label{fill:var(--muted);font-size:12px}
.bowls-label-steep{fill:var(--warn)}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('bowls-styles')) return;
  document.head.append(el('style', { id: 'bowls-styles' }, CSS));
}
