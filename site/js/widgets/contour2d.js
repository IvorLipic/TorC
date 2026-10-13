// widgets/contour2d.js — the loss landscape L(w, b) as a heat map with labelled contour lines, a trail of parameter values and an arrow
// for the move the next step() would make. Everything is computed in the browser from the engine's model (optim.js makeModel), so the
// picture and the numbers cannot disagree; the tests compare it with the 25 loss values TorC printed (data/optim.json surface).
//
//   const c = mountContour2d(host, { model, base: {w: [-1, 4.5], b: [-1.5, 3]}, optimum: {w: 2, b: 1}, samples });
//   c.draw({ trail: [{w, b, loss, gw, gb}, …], next: {w, b} | null, caption });
//
// Pure helpers are exported for the tests (gridFor, contourSegments, niceLevels, fitView, chooseView, heatColor); the module touches
// the DOM only when mountContour2d runs. Accessibility: role="img" with a live description, every plotted number is also in the
// "Show as table" views, markers differ by shape and label (not only by colour).
import { svg, el } from '../core/dom.js';
import { createPlot, tableToggle, ticks, fmtTick } from './plot.js';
import { injectStyle, uid, fmt, scrollable, fitWidth, onResize } from './sgd-ui.js';

// ---- pure helpers --------------------------------------------------------------------------------------------
// Loss on an (nw x nb) grid whose nodes include both ends of each range. z[j][i] is the loss at (xs[i], ys[j]).
export function gridFor(model, wr, br, nw = 64, nb = 56) {
  const xs = Array.from({ length: nw }, (_, i) => wr[0] + ((wr[1] - wr[0]) * i) / (nw - 1));
  const ys = Array.from({ length: nb }, (_, j) => br[0] + ((br[1] - br[0]) * j) / (nb - 1));
  let zmin = Infinity, zmax = -Infinity;
  const z = ys.map((b) => xs.map((w) => { const v = model.loss(w, b); if (v < zmin) zmin = v; if (v > zmax) zmax = v; return v; }));
  return { xs, ys, z, zmin, zmax };
}

// Marching squares: the segments [[x1, y1], [x2, y2]] where the grid is crossed by z = level (data coordinates).
export function contourSegments({ xs, ys, z }, level) {
  const segs = [];
  const lerp = (a, b, va, vb) => a + (b - a) * ((level - va) / (vb - va));
  for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
    const v00 = z[j][i], v10 = z[j][i + 1], v11 = z[j + 1][i + 1], v01 = z[j + 1][i];
    const idx = (v00 >= level ? 1 : 0) | (v10 >= level ? 2 : 0) | (v11 >= level ? 4 : 0) | (v01 >= level ? 8 : 0);
    if (idx === 0 || idx === 15) continue;
    const x0 = xs[i], x1 = xs[i + 1], y0 = ys[j], y1 = ys[j + 1];
    const B = () => [lerp(x0, x1, v00, v10), y0];
    const R = () => [x1, lerp(y0, y1, v10, v11)];
    const T = () => [lerp(x0, x1, v01, v11), y1];
    const L = () => [x0, lerp(y0, y1, v00, v01)];
    switch (idx) {
      case 1: case 14: segs.push([L(), B()]); break;
      case 2: case 13: segs.push([B(), R()]); break;
      case 3: case 12: segs.push([L(), R()]); break;
      case 4: case 11: segs.push([R(), T()]); break;
      case 5: segs.push([L(), T()], [B(), R()]); break;
      case 6: case 9: segs.push([B(), T()]); break;
      case 7: case 8: segs.push([L(), T()]); break;
      case 10: segs.push([L(), B()], [R(), T()]); break;
      default: break;
    }
  }
  return segs;
}

// A readable set of contour levels strictly between lo and hi: 1-2-5 per decade, coarsened until there are at most `max`.
export function niceLevels(lo, hi, max = 9) {
  const floor = Math.max(lo, 0.5);
  for (const family of [[1, 2, 5], [1, 3], [1]]) {
    const out = [];
    for (let d = -1; d <= 8; d++) for (const m of family) { const v = m * 10 ** d; if (v > floor && v < hi) out.push(v); }
    if (out.length <= max) return out;
    if (family.length === 1) return out.slice(0, max);
  }
  return [];
}

// Choose pixel-equal scales: the (w, b) ranges are widened so one unit of w is as long as one unit of b on screen.
export function fitView(wr, br, iw, ih) {
  const s = Math.min(iw / (wr[1] - wr[0]), ih / (br[1] - br[0]));
  const cw = (wr[0] + wr[1]) / 2, cb = (br[0] + br[1]) / 2;
  return { w: [cw - iw / (2 * s), cw + iw / (2 * s)], b: [cb - ih / (2 * s), cb + ih / (2 * s)], scale: s };
}

// The default range if everything fits, otherwise the smallest range that contains the base range and every point (plus 8 % margin).
export function chooseView(base, points) {
  const inside = points.every((p) => p.w >= base.w[0] && p.w <= base.w[1] && p.b >= base.b[0] && p.b <= base.b[1]);
  if (inside) return { w: base.w.slice(), b: base.b.slice(), zoomedOut: false };
  const ext = (lo, hi, vals) => { const a = Math.min(lo, ...vals), b = Math.max(hi, ...vals), m = 0.08 * (b - a); return [a - m, b + m]; };
  return { w: ext(base.w[0], base.w[1], points.map((p) => p.w)), b: ext(base.b[0], base.b[1], points.map((p) => p.b)), zoomedOut: true };
}

const STOPS = [[0, [20, 33, 80]], [0.3, [31, 84, 150]], [0.55, [42, 157, 143]], [0.8, [233, 196, 106]], [1, [244, 162, 97]]];
// loss -> colour on a log-like scale so the valley floor and the walls are both visible
export function heatColor(loss, lossMax) {
  const t = Math.min(1, Math.max(0, Math.log1p(Math.max(loss, 0)) / Math.log1p(Math.max(lossMax, 1e-9))));
  let k = 1;
  while (k < STOPS.length - 1 && t > STOPS[k][0]) k++;
  const [t0, c0] = STOPS[k - 1], [t1, c1] = STOPS[k];
  const f = (t - t0) / (t1 - t0 || 1);
  return `rgb(${c0.map((v, i) => Math.round(v + (c1[i] - v) * f)).join(',')})`;
}

const STAR = (cx, cy, r) => Array.from({ length: 10 }, (_, i) => { const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r; return `${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`; }).join(' ');

const CSS = `
.c2d{min-width:0}
.c2d svg.plot{max-height:min(54vh,430px);margin:0 auto}
.c2d .c2d-heat-img{image-rendering:auto}
.c2d .c2d-contour{fill:none;stroke:rgba(255,255,255,.62);stroke-width:1}
.c2d .c2d-clabel{fill:#fff;font-size:10.5px;stroke:#0b0f14;stroke-width:3px;paint-order:stroke;stroke-linejoin:round}
.c2d .c2d-halo{fill:none;stroke:#0b0f14;stroke-width:5.5px;stroke-linejoin:round;stroke-linecap:round}
.c2d .c2d-path{fill:none;stroke:#fff;stroke-width:2.4px;stroke-linejoin:round;stroke-linecap:round}
.c2d .c2d-next{fill:none;stroke:#fff;stroke-width:2.4px;stroke-dasharray:6 5}
.c2d .c2d-pt{fill:#fff;stroke:#0b0f14;stroke-width:2px}
.c2d .c2d-cur{fill:var(--bwd);stroke:#fff;stroke-width:2.5px}
.c2d .c2d-start{fill:#0b0f14;stroke:#fff;stroke-width:2px}
.c2d .c2d-opt{fill:#fff;stroke:#0b0f14;stroke-width:1.5px}
.c2d .c2d-lbl{fill:#fff;font-size:12px;font-weight:700;stroke:#0b0f14;stroke-width:3.5px;paint-order:stroke;stroke-linejoin:round}
.c2d .c2d-frame{fill:none;stroke:var(--muted);stroke-width:1}
.c2d .c2d-axis-note{margin:.2rem 0 0;color:var(--muted);font-size:.8rem}
.c2d-legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:.3rem 0 0;padding:0;list-style:none;font-size:.8rem;color:#c4ccd5}
.c2d-legend svg{width:18px;height:14px;vertical-align:-2px;margin-right:4px}
`;

// ---- the widget -----------------------------------------------------------------------------------------------------
export function mountContour2d(host, { model, base = { w: [-1, 4.5], b: [-1.5, 3] }, optimum = { w: 2, b: 1 }, samples = null, maxWidth = 560 }) {
  injectStyle('sgd-contour2d-css', CSS);
  const wrap = el('div', { class: 'c2d' });
  const plotHost = el('div');
  const legend = el('ul', { class: 'c2d-legend', 'aria-label': 'Legend' });
  const note = el('p', { class: 'c2d-axis-note', 'aria-live': 'polite' });
  wrap.append(plotHost, legend, note);
  host.append(wrap);
  const margin = { l: 44, r: 10, t: 10, b: 36 };
  let width = maxWidth, height = Math.round(maxWidth * 0.77), iw = 0, ih = 0;
  const setSize = () => { width = fitWidth(wrap, 300, maxWidth, maxWidth); height = Math.round(width * 0.77); iw = width - margin.l - margin.r; ih = height - margin.t - margin.b; };
  setSize();
  const ac = new AbortController();
  const cache = new Map();                     // view key -> {grid, levels, segs}
  let built = null;                            // {key, plot, layers}
  let last = null;
  const trailTable = scrollable(tableToggle(wrap, {
    columns: ['update', 'w', 'b', 'loss', 'dL/dw', 'dL/db'], rows: [], caption: 'Positions on the trail (before each update; the last row is where the trail stands now)',
    fmt: (v) => (typeof v === 'number' ? fmtTick(v) : String(v)),
  }));
  let sampleTable = null;
  if (samples) {
    const ws = [...new Set(samples.map((p) => p.w))], bs = [...new Set(samples.map((p) => p.b))];
    const rows = ws.map((w) => [`w = ${fmtTick(w)}`, ...bs.map((b) => samples.find((p) => p.w === w && p.b === b).loss)]);
    sampleTable = scrollable(tableToggle(wrap, { columns: ['loss printed by TorC', ...bs.map((b) => `b = ${fmtTick(b)}`)], rows, caption: 'The 25 loss values the real library computed (the heat map is recomputed in your browser and agrees with them)' }));
    sampleTable.wrap.querySelector('button').textContent = 'Show TorC’s 25 loss samples';
    sampleTable.wrap.querySelector('button').addEventListener('click', function () { this.textContent = sampleTable.table.hidden ? 'Show TorC’s 25 loss samples' : 'Hide the samples'; });
  }

  function legendItems(start) {
    const item = (icon, text) => el('li', {}, icon, text);
    const sv = (...k) => svg('svg', { viewBox: '0 0 18 14', 'aria-hidden': 'true' }, ...k);
    legend.replaceChildren(
      item(sv(svg('rect', { x: 3, y: 3, width: 9, height: 9, fill: '#0b0f14', stroke: '#fff', 'stroke-width': 2 })), `start (${fmtTick(start.w)}, ${fmtTick(start.b)})`),
      item(sv(svg('circle', { cx: 9, cy: 7, r: 5, fill: '#fc6255', stroke: '#fff', 'stroke-width': 2 })), 'where the parameters are now'),
      item(sv(svg('line', { x1: 1, y1: 7, x2: 17, y2: 7, stroke: '#fff', 'stroke-width': 2, 'stroke-dasharray': '4 3' })), 'the next step() would move here'),
      item(sv(svg('polygon', { points: STAR(9, 7.5, 7), fill: '#fff', stroke: '#0b0f14' })), `minimum (${fmtTick(optimum.w)}, ${fmtTick(optimum.b)})`),
      item(sv(svg('line', { x1: 1, y1: 7, x2: 17, y2: 7, stroke: '#cfd8e3', 'stroke-width': 1.3 })), 'contour line: equal loss (labelled)'));
  }

  function buildBase(view) {
    const key = [width, ...view.w, ...view.b].map((v) => Number(v).toFixed(3)).join('|');
    const fit = fitView(view.w, view.b, iw, ih);
    plotHost.replaceChildren();
    const plot = createPlot(plotHost, { width, height, x: fit.w, y: fit.b, xLabel: 'w  (weight)', yLabel: 'b  (bias)', ariaLabel: 'Loss landscape over w and b', margin });
    let entry = cache.get(key);
    if (!entry) {
      const grid = gridFor(model, fit.w, fit.b);
      const levels = niceLevels(grid.zmin, grid.zmax);
      entry = { grid, levels, segs: levels.map((l) => contourSegments(grid, l)), url: null };
      cache.set(key, entry);
    }
    const { grid } = entry;
    plot.axes({ xTicks: ticks(fit.w, 6), yTicks: ticks(fit.b, 6) });
    // heat map: one small canvas (one pixel per grid node) stretched over the plot, so the browser interpolates smoothly
    const heat = plot.layer('heat');
    const dw = (grid.xs[1] - grid.xs[0]) / 2, db = (grid.ys[1] - grid.ys[0]) / 2;
    try {
      if (!entry.url) {
        const cv = document.createElement('canvas');
        cv.width = grid.xs.length; cv.height = grid.ys.length;
        const g = cv.getContext('2d');
        const img = g.createImageData(cv.width, cv.height);
        grid.z.forEach((row, j) => row.forEach((v, i) => {
          const m = /(\d+),(\d+),(\d+)/.exec(heatColor(v, grid.zmax));
          const o = ((cv.height - 1 - j) * cv.width + i) * 4;
          img.data[o] = +m[1]; img.data[o + 1] = +m[2]; img.data[o + 2] = +m[3]; img.data[o + 3] = 255;
        }));
        g.putImageData(img, 0, 0);
        entry.url = cv.toDataURL('image/png');
      }
      heat.append(svg('image', { class: 'c2d-heat-img', href: entry.url, x: plot.sx(grid.xs[0] - dw), y: plot.sy(grid.ys.at(-1) + db), width: plot.sx(grid.xs.at(-1) + dw) - plot.sx(grid.xs[0] - dw), height: plot.sy(grid.ys[0] - db) - plot.sy(grid.ys.at(-1) + db), preserveAspectRatio: 'none' }));
    } catch {
      grid.z.forEach((row, j) => row.forEach((v, i) => heat.append(svg('rect', { x: plot.sx(grid.xs[i] - dw), y: plot.sy(grid.ys[j] + db), width: 2 * dw * fit.scale + 0.6, height: 2 * db * fit.scale + 0.6, fill: heatColor(v, grid.zmax) }))));
    }
    // contour lines + one label per level, placed where the line is highest on screen but inside the plot
    const cl = plot.layer('contour');
    entry.levels.forEach((lev, k) => {
      const segs = entry.segs[k];
      if (!segs.length) return;
      cl.append(svg('path', { class: 'c2d-contour', d: segs.map(([[x1, y1], [x2, y2]]) => `M${plot.sx(x1).toFixed(1)},${plot.sy(y1).toFixed(1)}L${plot.sx(x2).toFixed(1)},${plot.sy(y2).toFixed(1)}`).join('') }));
      let best = null;
      for (const [[x1, y1], [x2, y2]] of segs) {
        const px = plot.sx((x1 + x2) / 2), py = plot.sy((y1 + y2) / 2);
        if (px < margin.l + 16 || px > width - margin.r - 16 || py < margin.t + 10 || py > margin.t + ih - 8) continue;
        if (!best || py < best.py) best = { px, py };
      }
      if (best) cl.append(svg('text', { class: 'c2d-clabel', x: best.px, y: best.py + 3.5, 'text-anchor': 'middle' }, fmt(lev)));
    });
    plot.root.append(svg('rect', { class: 'c2d-frame', x: margin.l, y: margin.t, width: iw, height: ih }));
    plot.layer('trail');
    // the minimum, drawn once with the base
    const mk = plot.layer('marks');
    mk.append(svg('polygon', { class: 'c2d-opt', points: STAR(plot.sx(optimum.w), plot.sy(optimum.b), 9) }),
      svg('text', { class: 'c2d-lbl', x: plot.sx(optimum.w) - 12, y: plot.sy(optimum.b) - 9, 'text-anchor': 'end' }, 'minimum'));
    built = { key, plot, view };
    return built;
  }

  function draw({ trail, next = null, caption = '' }) {
    const pts = [...trail, ...(next ? [next] : [])];
    last = { trail, next, caption };
    const w0 = width; setSize();
    if (width !== w0) built = null;
    const view = chooseView(base, [...pts, optimum]);
    const key = [width, ...view.w, ...view.b].map((v) => Number(v).toFixed(3)).join('|');
    const b = built && built.key === key ? built : buildBase(view);
    const { plot } = b;
    plot.clear('trail');
    const layer = plot.layer('trail');
    const sx = plot.sx, sy = plot.sy;
    const path = (arr) => arr.map((p, i) => `${i ? 'L' : 'M'}${sx(p.w).toFixed(1)},${sy(p.b).toFixed(1)}`).join('');
    if (next) {
      const last = trail.at(-1);
      const d = `M${sx(last.w).toFixed(1)},${sy(last.b).toFixed(1)}L${sx(next.w).toFixed(1)},${sy(next.b).toFixed(1)}`;
      const mid = uid('c2dm');
      layer.append(svg('defs', {}, svg('marker', { id: mid, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }, svg('path', { d: 'M0,0L10,5L0,10z', fill: '#fff', stroke: '#0b0f14', 'stroke-width': 1.2 }))),
        svg('path', { class: 'c2d-halo', d }), svg('path', { class: 'c2d-next', d, 'marker-end': `url(#${mid})` }));
    }
    if (trail.length > 1) layer.append(svg('path', { class: 'c2d-halo', d: path(trail) }), svg('path', { class: 'c2d-path', d: path(trail) }));
    const lastI = trail.length - 1;
    const labelled = [];                       // pixel positions of the numbers already written, so they never pile up
    const wantLabel = (cx, cy, force) => {
      if (!force && labelled.some(([x, y]) => Math.hypot(x - cx, y - cy) < 22)) return false;
      labelled.push([cx, cy]);
      return true;
    };
    if (lastI > 0 && trail.length <= 13) wantLabel(sx(trail[lastI].w), sy(trail[lastI].b), true);
    trail.forEach((p, i) => {
      const cx = sx(p.w), cy = sy(p.b);
      const tip = `after ${i} update${i === 1 ? '' : 's'}: w = ${fmt(p.w)}, b = ${fmt(p.b)}, loss = ${fmt(p.loss)}`;
      if (i === 0) layer.append(svg('rect', { class: 'c2d-start', x: cx - 5, y: cy - 5, width: 10, height: 10 }, svg('title', {}, tip)));
      else if (i !== lastI) layer.append(svg('circle', { class: 'c2d-pt', cx, cy, r: 4 }, svg('title', {}, tip)));
      if (i > 0 && i !== lastI && trail.length <= 13 && wantLabel(cx, cy, false)) layer.append(svg('text', { class: 'c2d-lbl', x: cx + 7, y: cy - 6 }, String(i)));
    });
    if (lastI > 0) {
      const p = trail[lastI], cx = sx(p.w), cy = sy(p.b);
      layer.append(svg('circle', { class: 'c2d-cur', cx, cy, r: 7 }, svg('title', {}, `after ${lastI} updates: w = ${fmt(p.w)}, b = ${fmt(p.b)}, loss = ${fmt(p.loss)}`)));
      if (trail.length <= 13) layer.append(svg('text', { class: 'c2d-lbl', x: cx + 10, y: cy - 8 }, String(lastI)));
    } else {
      const p = trail[0], cx = sx(p.w), cy = sy(p.b);
      layer.append(svg('circle', { class: 'c2d-cur', cx, cy, r: 7 }, svg('title', {}, `start: w = ${fmt(p.w)}, b = ${fmt(p.b)}, loss = ${fmt(p.loss)}`)));
    }
    const cur = trail.at(-1);
    plot.root.setAttribute('aria-label', `Loss landscape over w and b. ${caption} The parameters are at w = ${fmt(cur.w)}, b = ${fmt(cur.b)} where the loss is ${fmt(cur.loss)}. The minimum is at w = ${fmt(optimum.w)}, b = ${fmt(optimum.b)}.`);
    note.textContent = view.zoomedOut ? 'The view zoomed out to keep the whole trail in sight; the bowl you saw before is the small region around the star.' : '';
    trailTable.update(trail.map((p, i) => [i, p.w, p.b, p.loss, p.gw ?? '–', p.gb ?? '–']));
    legendItems(trail[0]);
    return { view };
  }

  onResize(() => { if (last && fitWidth(wrap, 300, maxWidth, maxWidth) !== width) draw(last); }, ac.signal);
  return { draw, el: wrap, destroy() { ac.abort(); wrap.remove(); } };
}
