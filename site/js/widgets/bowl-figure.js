// widgets/bowl-figure.js — a small static picture of the whole loss curve of the two-notebook model, for the setup text of the
// lesson why-gradients (review H1: the reader sees the curve, its bottom and the start before any step keeps a share of the
// distance). Pure: it returns an SVG string built from the recorded grid (data/why-gradients.json `grid`), so Node renders it
// in the tests exactly as the browser does. Labels sit on the plot; the classes are the shared plot classes of css/widgets.css.
import { escapeHtml } from '../core/dom.js';

const W = 420, H = 210, M = { l: 40, r: 14, t: 14, b: 34 };

// grid: [{w, L}], start: {w, L}, wMin: the bottom; returns '<figure>…<svg>…</svg><figcaption>…</figcaption></figure>'
export function bowlFigure({ grid, start, wMin, caption }) {
  const xs = grid.map((q) => q.w), ys = grid.map((q) => q.L);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys) * 1.08;
  const sx = (v) => M.l + ((v - x0) / (x1 - x0)) * (W - M.l - M.r);
  const sy = (v) => M.t + (1 - v / y1) * (H - M.t - M.b);
  const f = (v) => v.toFixed(1);
  const d = grid.map((q, i) => `${i ? 'L' : 'M'}${f(sx(q.w))},${f(sy(q.L))}`).join(' ');
  const base = sy(0);
  const aria = `The loss of the two-notebook model against w. The curve is a bowl. It starts high at w = ${start.w}, falls to zero at the bottom, w = ${wMin}, and rises again.`;
  return `<figure class="wg-fig"><svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(aria)}" focusable="false">`
    + `<rect class="plot-frame" x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" fill="none"/>`
    + `<line class="plot-grid" x1="${f(sx(wMin))}" y1="${M.t}" x2="${f(sx(wMin))}" y2="${f(base)}"/>`
    + `<path class="plot-curve" d="${d}" fill="none"/>`
    + `<circle class="plot-dot" cx="${f(sx(start.w))}" cy="${f(sy(start.L))}" r="5"/>`
    + `<circle class="plot-dot" cx="${f(sx(wMin))}" cy="${f(base)}" r="4"/>`
    + `<text class="plot-text" x="${f(sx(start.w) + 9)}" y="${f(sy(start.L) + 4)}">start</text>`
    + `<text class="plot-text" x="${f(sx(wMin))}" y="${f(base - 10)}" text-anchor="middle">bottom</text>`
    + `<text class="plot-tick" x="${f(sx(start.w))}" y="${f(base + 15)}" text-anchor="middle">${start.w}</text>`
    + `<text class="plot-tick" x="${f(sx(wMin))}" y="${f(base + 15)}" text-anchor="middle">${wMin}</text>`
    + `<text class="plot-label" x="${f((M.l + W - M.r) / 2)}" y="${H - 4}" text-anchor="middle">w</text>`
    + `<text class="plot-label" transform="translate(14 ${f((M.t + base) / 2)}) rotate(-90)" text-anchor="middle">loss</text>`
    + `</svg><figcaption>${caption}</figcaption></figure>`;
}
