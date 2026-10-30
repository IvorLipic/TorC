// widgets/nudge-lens.js — the nudge lens of the lesson why-gradients (brief §9 "Nudge lens", Appendix B): choose the size of a
// nudge of w from w = 0, and compare the change of the loss that the straight line (gradient × nudge) predicts with the change
// that TorC measured. The view zooms with the nudge, so for a small nudge the loss curve and the straight line look like one line.
// The measured values are the recorded secants of data/why-gradients.json (`secants[].h, L0, L_h, dL, slope, backward_slope`);
// the curve between them is drawn by landscape-model.js, which site/test/why-gradients.test.mjs checks against the same file.
// Pure helpers (nudgeRows, lensView) are exported for the tests; the DOM is touched only in mountNudgeLens.
import { el } from '../core/dom.js';
import { createPlot, ticks } from './plot.js';
import { makeModel } from './landscape-model.js';
import { fmt3 } from './landscape-tex.js';

// One row per recorded nudge size, largest first (the order of the trace).
export function nudgeRows(files) {
  return files.secants.map((s) => ({
    h: s.h, w0: s.w0, before: s.L0, after: s.L_h, actual: s.dL, ratio: s.slope, grad: s.backward_slope,
    predicted: s.backward_slope * s.h,                 // what the straight line promises: gradient × nudge
    gap: s.dL - s.backward_slope * s.h,                // what the promise misses (4h² for this bowl)
  }));
}

// The window for nudge h from w0: w from w0 − 0.3h to w0 + 1.3h, the loss from the lowest to the highest value of the curve and
// the straight line in that window, with a margin. `lossAt` is the loss of the model.
export function lensView(row, lossAt) {
  const x = [row.w0 - 0.3 * row.h, row.w0 + 1.3 * row.h];
  const line = (w) => row.before + row.grad * (w - row.w0);
  const ys = [lossAt(x[0]), lossAt(x[1]), line(x[0]), line(x[1]), row.after];
  const lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.12;
  return { x, y: [lo - pad, hi + pad], line };
}

let seq = 0;

export function mountNudgeLens(host, D) {
  injectStyles();
  const files = D.files['why-gradients'];
  const m = makeModel(D.scenes.why);
  const rows = nudgeRows(files);
  const id = `lens${++seq}`;
  let k = 0;

  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const btns = rows.map((r, i) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { k = i; draw(); announce(); } }, String(r.h)));
  const controls = el('div', { class: 'lens-ctl', role: 'group', 'aria-labelledby': `${id}-h` },
    el('span', { class: 'lens-lbl', id: `${id}-h` }, 'size of the nudge'), ...btns);
  const plotBox = el('div', { class: 'lens-plot' });
  const readout = el('div', { class: 'lens-read' });
  const hint = el('p', { class: 'hint' }, 'The dashed straight line changes by the gradient for each unit of w. The view zooms in when you choose a smaller nudge.');
  host.replaceChildren(controls, plotBox, readout, hint, status);

  let p = null, lastW = 0;
  // the plot is rebuilt for every view (its axes change with the nudge), so a width change (rotating a phone) redraws it too
  const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => {
    if (host.isConnected && Math.abs(plotBox.clientWidth - lastW) > 1) draw();
  });
  ro?.observe(plotBox);
  function draw() {
    const r = rows[k], v = lensView(r, m.lossAt);
    btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === k)));
    p?.destroy();
    plotBox.replaceChildren();
    p = createPlot(plotBox, { width: 640, height: 300, x: v.x, y: v.y, xLabel: 'w', yLabel: 'loss', margin: { l: 64 },
      ariaLabel: `The loss curve near w = ${fmt3(r.w0)} and the straight line with the gradient ${fmt3(r.grad)}. For a nudge of ${r.h}, the line predicts a change of ${fmt3(r.predicted)}, and the actual change is ${fmt3(r.actual)}.` });
    const dec = (t) => String(Number(t.toPrecision(4)));   // 0.0005, not 5e-4: the reader knows decimals
    p.axes({ xTicks: ticks(v.x, 5), yTicks: ticks(v.y, 5), fmt: dec });
    const n = 120, pts = Array.from({ length: n + 1 }, (_, i) => { const w = v.x[0] + ((v.x[1] - v.x[0]) * i) / n; return [w, m.lossAt(w)]; });
    p.curve(pts, { class: 'plot-curve lens-curve' });
    p.line(v.x[0], v.line(v.x[0]), v.x[1], v.line(v.x[1]), { class: 'lens-line' });
    p.line(r.w0 + r.h, r.after, r.w0 + r.h, r.before + r.predicted, { class: 'lens-gap' });
    p.line(r.w0, v.y[0], r.w0, r.before, { class: 'lens-mark' });
    p.line(r.w0 + r.h, v.y[0], r.w0 + r.h, r.after, { class: 'lens-mark' });
    p.dot(r.w0, r.before, { r: 6 });
    p.dot(r.w0 + r.h, r.after, { class: 'plot-dot lens-actual', r: 5 });
    p.dot(r.w0 + r.h, r.before + r.predicted, { class: 'lens-pred', r: 4 });
    // labels on the plot: the curve is above the straight line (the bowl bends upward), so its label goes above, the line's below
    const lx = r.w0 + 0.5 * r.h;
    p.text(lx, m.lossAt(lx), 'loss curve', { class: 'plot-text lens-label', anchor: 'start', dx: 4, dy: -12 });
    p.text(lx, v.line(lx), 'straight line', { class: 'plot-text lens-label lens-label-line', anchor: 'end', dx: -4, dy: 20 });
    p.text(r.w0, v.y[0], 'start', { class: 'plot-text lens-label', dx: 4, dy: -6 });
    readout.innerHTML = `<span><b>nudge</b> ${r.h}</span>`
      + `<span><b>predicted change, gradient × nudge</b> ${fmt3(r.predicted)}</span>`
      + `<span><b>actual change of the loss</b> ${fmt3(r.actual)}</span>`
      + `<span><b>actual change ÷ nudge</b> ${fmt3(r.ratio)}</span>`;
    lastW = plotBox.clientWidth;
  }
  function announce() {
    const r = rows[k];
    status.textContent = `Nudge ${r.h}. The predicted change is ${fmt3(r.predicted)}. The actual change is ${fmt3(r.actual)}. The actual change divided by the nudge is ${fmt3(r.ratio)}.`;
  }
  draw();
  return { choose: (i) => btns[i].click(), destroy: () => { ro?.disconnect(); p?.destroy(); } };
}

const CSS = `
.lens-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:.2rem 0 .4rem}
.lens-lbl{color:var(--muted);font-size:.9rem;margin-right:.3rem}
.lens-read{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;align-items:baseline;font-family:ui-monospace,monospace;font-size:.92rem;margin:.3rem 0 0}
.lens-read b{color:var(--muted);font-weight:600;margin-right:.3rem;font-family:system-ui,sans-serif}
.lens-line{stroke:var(--fwd);stroke-width:1.6;stroke-dasharray:6 4}
.lens-gap{stroke:var(--warn);stroke-width:3}
.lens-mark{stroke:var(--muted);stroke-dasharray:2 4}
.lens-pred{fill:var(--fwd)}
.lens-label{fill:var(--muted);font-size:12px}
.lens-label-line{fill:var(--fwd)}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('lens-styles')) return;
  document.head.append(el('style', { id: 'lens-styles' }, CSS));
}
