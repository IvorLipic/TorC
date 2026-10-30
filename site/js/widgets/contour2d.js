// widgets/contour2d.js — the loss landscape L(w, b) as a heat map with labelled contour lines, a trail of parameter values, an arrow
// for the move the next step() would make, an optional gradient arrow (uphill) and an optional fog that hides the map. Everything is computed in the browser from the engine's model (optim.js makeModel), so the
// picture and the numbers cannot disagree; the tests compare it with the 25 loss values TorC printed (data/optim.json surface).
//
//   const c = mountContour2d(host, { model, base: {w: [-1, 4.5], b: [-1.5, 3]}, optimum: {w: 2, b: 1} });
//   c.draw({ trail: [{w, b, loss, gw, gb}, …], next: {w, b} | null, uphill: {w, b} | null, fog: false, caption });
//   `uphill` is the tip of the gradient arrow drawn from the last trail point (the sgd-step lesson passes point + lr·gradient, the
//   mirror image of the step). `fog: true` hides the heat map, the contour lines and the lowest point, so only the trail and the arrows show.
//
// Pure helpers are exported for the tests (gridFor, contourSegments, niceLevels, fitView, chooseView, heatColor); the module touches
// the DOM only when mountContour2d runs. Accessibility: role="img" with a live description, the trail is also in a screen-reader-only
// table, markers differ by shape and label (not only by colour).
import { svg, el } from '../core/dom.js';
import { createPlot, srTable, ticks, fmtTick } from './plot.js';
import { injectStyle, uid, fmt, fitWidth, onResize } from './sgd-ui.js';

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

// The points the view must contain: the trail, the next point, the tip of the gradient arrow and the lowest point (M1: an arrow
// whose tip leaves the view would lose its head to the clip path).
export const viewFor = (base, { trail, next = null, uphill = null, optimum }) => chooseView(base, [...trail, ...(next ? [next] : []), ...(uphill ? [uphill] : []), optimum]);

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
.c2d .c2d-fogbg{fill:#39424d}
.c2d .c2d-grad{fill:none;stroke:#ffb454;stroke-width:3.5px;stroke-linecap:round}
.c2d .c2d-gradlbl{fill:#ffb454;font-size:12px;font-weight:700;stroke:#0b0f14;stroke-width:3.5px;paint-order:stroke;stroke-linejoin:round}
.c2d.fog .layer-heat,.c2d.fog .layer-contour,.c2d.fog .layer-marks,.c2d-legend .c2d-hide{display:none}
`;

// ---- the widget -----------------------------------------------------------------------------------------------------
export function mountContour2d(host, { model, base = { w: [-1, 4.5], b: [-1.5, 3] }, optimum = { w: 2, b: 1 }, maxWidth = 560 }) {
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
  const trailTable = srTable(wrap, {
    columns: ['step', 'w', 'b', 'loss', 'gradient for w', 'gradient for b'], rows: [], caption: 'Positions on the trail. Each row is the position before a step. The last row is where the trail stands now.',
    fmt: (v) => (typeof v === 'number' ? fmtTick(v) : String(v)),
  });

  function legendItems(start, { uphill, fog }) {
    const item = (icon, text, show = true) => el('li', show ? {} : { class: 'c2d-hide' }, icon, text);
    const sv = (...k) => svg('svg', { viewBox: '0 0 18 14', 'aria-hidden': 'true' }, ...k);
    legend.replaceChildren(
      item(sv(svg('rect', { x: 3, y: 3, width: 9, height: 9, fill: '#0b0f14', stroke: '#fff', 'stroke-width': 2 })), `start (${fmtTick(start.w)}, ${fmtTick(start.b)})`),
      item(sv(svg('circle', { cx: 9, cy: 7, r: 5, fill: '#fc6255', stroke: '#fff', 'stroke-width': 2 })), 'current point'),
      item(sv(svg('line', { x1: 1, y1: 7, x2: 17, y2: 7, stroke: '#fff', 'stroke-width': 2, 'stroke-dasharray': '4 3' })), 'next step'),
      item(sv(svg('line', { x1: 1, y1: 7, x2: 17, y2: 7, stroke: '#ffb454', 'stroke-width': 3 })), 'gradient × lr, drawn from the current point', Boolean(uphill)),
      item(sv(svg('polygon', { points: STAR(9, 7.5, 7), fill: '#fff', stroke: '#0b0f14' })), `lowest point (${fmtTick(optimum.w)}, ${fmtTick(optimum.b)})`, !fog));
  }

  function buildBase(view) {
    const key = [width, ...view.w, ...view.b].map((v) => Number(v).toFixed(3)).join('|');
    const fit = fitView(view.w, view.b, iw, ih);
    plotHost.replaceChildren();
    const plot = createPlot(plotHost, { width, height, x: fit.w, y: fit.b, xLabel: 'w', yLabel: 'b', ariaLabel: 'Loss map over w and b', margin });
    let entry = cache.get(key);
    if (!entry) {
      const grid = gridFor(model, fit.w, fit.b);
      const levels = niceLevels(grid.zmin, grid.zmax);
      entry = { grid, levels, segs: levels.map((l) => contourSegments(grid, l)), url: null };
      cache.set(key, entry);
    }
    const { grid } = entry;
    plot.axes({ xTicks: ticks(fit.w, 6), yTicks: ticks(fit.b, 6) });
    // the fog: a plain field under the heat map, seen only when the fog hides the heat map
    plot.layer('fog').append(svg('rect', { class: 'c2d-fogbg', x: margin.l, y: margin.t, width: iw, height: ih }));
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
    const placed = [];                         // label boxes already written: [x, y, half width]
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
      const text = fmt(lev), half = 3.2 * text.length + 4;
      if (best && !placed.some(([x, y, h]) => Math.abs(x - best.px) < h + half && Math.abs(y - best.py) < 14)) {
        placed.push([best.px, best.py, half]);
        cl.append(svg('text', { class: 'c2d-clabel', x: best.px, y: best.py + 3.5, 'text-anchor': 'middle' }, svg('title', {}, `Loss ${text} on this contour line.`), text));
      }
    });
    plot.root.append(svg('rect', { class: 'c2d-frame', x: margin.l, y: margin.t, width: iw, height: ih }));
    plot.layer('trail');
    // the minimum, drawn once with the base
    const mk = plot.layer('marks');
    mk.append(svg('polygon', { class: 'c2d-opt', points: STAR(plot.sx(optimum.w), plot.sy(optimum.b), 9) }),
      svg('text', { class: 'c2d-lbl', x: plot.sx(optimum.w) + 12, y: plot.sy(optimum.b) + 16, 'text-anchor': 'start' }, 'lowest point'));
    built = { key, plot, view };
    return built;
  }

  function draw({ trail, next = null, uphill = null, fog = false, caption = '' }) {
    last = { trail, next, uphill, fog, caption };
    wrap.classList.toggle('fog', fog);
    const w0 = width; setSize();
    if (width !== w0) built = null;
    const view = viewFor(base, { trail, next, uphill, optimum });
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
    // the gradient arrow goes on top of the trail: from the second step on it points back along the path already walked
    if (uphill) {
      const from = trail.at(-1), x1 = sx(from.w), y1 = sy(from.b), x2 = sx(uphill.w), y2 = sy(uphill.b);
      const len = Math.hypot(x2 - x1, y2 - y1);
      if (len >= 2) {
        const gid = uid('c2dg');
        const d = `M${x1.toFixed(1)},${y1.toFixed(1)}L${x2.toFixed(1)},${y2.toFixed(1)}`;
        layer.append(svg('defs', {}, svg('marker', { id: gid, viewBox: '0 0 10 10', refX: 7, refY: 5, markerWidth: 4.5, markerHeight: 4.5, orient: 'auto' }, svg('path', { d: 'M0,0L10,5L0,10z', fill: '#ffb454' }))),
          svg('path', { class: 'c2d-halo', d }), svg('path', { class: 'c2d-grad', d, 'marker-end': `url(#${gid})` }));
        if (len >= 28) {
          // the label sits past the tip, but turns back inside the plot when the tip is near an edge (about 90 px of text)
          const room = 92, right = x2 >= x1 ? x2 + 6 + room <= margin.l + iw : x2 - 6 - room < margin.l;
          const y = Math.min(Math.max(y2 + (y2 > y1 ? 14 : -6), margin.t + 12), margin.t + ih - 4);
          layer.append(svg('text', { class: 'c2d-gradlbl', x: right ? x2 + 6 : x2 - 6, y, 'text-anchor': right ? 'start' : 'end' }, 'gradient × lr'));
        }
      }
    }
    const lastI = trail.length - 1;
    const labelled = [];                       // pixel positions of the numbers already written, so they never pile up
    const wantLabel = (cx, cy, force) => {
      if (!force && (labelled.some(([x, y]) => Math.hypot(x - cx, y - cy) < 30) || Math.hypot(sx(optimum.w) - cx, sy(optimum.b) - cy) < 30)) return false;
      labelled.push([cx, cy]);
      return true;
    };
    if (lastI > 0 && trail.length <= 13) wantLabel(sx(trail[lastI].w), sy(trail[lastI].b), true);
    trail.forEach((p, i) => {
      const cx = sx(p.w), cy = sy(p.b);
      const tip = `After ${i} step${i === 1 ? '' : 's'}, w = ${fmt(p.w)}, b = ${fmt(p.b)} and the loss is ${fmt(p.loss)}.`;
      if (i === 0) layer.append(svg('rect', { class: 'c2d-start', x: cx - 5, y: cy - 5, width: 10, height: 10 }, svg('title', {}, tip)));
      else if (i !== lastI) layer.append(svg('circle', { class: 'c2d-pt', cx, cy, r: 4 }, svg('title', {}, tip)));
      if (i > 0 && i !== lastI && trail.length <= 13 && wantLabel(cx, cy, false)) layer.append(svg('text', { class: 'c2d-lbl', x: cx + 7, y: cy - 6 }, String(i)));
    });
    if (lastI > 0) {
      const p = trail[lastI], cx = sx(p.w), cy = sy(p.b);
      layer.append(svg('circle', { class: 'c2d-cur', cx, cy, r: 7 }, svg('title', {}, `After ${lastI} steps, w = ${fmt(p.w)}, b = ${fmt(p.b)} and the loss is ${fmt(p.loss)}.`)));
      if (trail.length <= 13) layer.append(svg('text', { class: 'c2d-lbl', x: cx + 10, y: cy - 8 }, String(lastI)));
    } else {
      const p = trail[0], cx = sx(p.w), cy = sy(p.b);
      layer.append(svg('circle', { class: 'c2d-cur', cx, cy, r: 7 }, svg('title', {}, `At the start, w = ${fmt(p.w)}, b = ${fmt(p.b)} and the loss is ${fmt(p.loss)}.`)));
    }
    const cur = trail.at(-1);
    plot.root.setAttribute('aria-label', (fog
      ? `Loss map over w and b, hidden by fog. ${caption} The current point is at w = ${fmt(cur.w)} and b = ${fmt(cur.b)}, and the loss there is ${fmt(cur.loss)}.`
      : `Loss map over w and b. ${caption} The current point is at w = ${fmt(cur.w)} and b = ${fmt(cur.b)}, and the loss there is ${fmt(cur.loss)}. The lowest point is at w = ${fmt(optimum.w)} and b = ${fmt(optimum.b)}.`)
      + (uphill ? ` An orange arrow shows the gradient times lr. It points uphill, toward w = ${fmt(uphill.w)} and b = ${fmt(uphill.b)}.` : ''));
    note.textContent = view.zoomedOut ? 'The view zoomed out to keep the whole trail in sight. The map you saw before is the region around the star.' : '';
    trailTable.update(trail.map((p, i) => [i, p.w, p.b, p.loss, p.gw ?? '–', p.gb ?? '–']));
    legendItems(trail[0], { uphill, fog });
    return { view };
  }

  onResize(() => { if (last && fitWidth(wrap, 300, maxWidth, maxWidth) !== width) draw(last); }, ac.signal);
  return { draw, el: wrap, destroy() { ac.abort(); wrap.remove(); } };
}
