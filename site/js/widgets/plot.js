// widgets/plot.js — a tiny SVG plotting helper (axes, curves, points, tangent lines, heat grids, legend-free) used by lessons.
// No charting library. Every plot should offer a table view (`tableToggle`) for accessibility and traceability.
//
//   const p = createPlot(host, { width: 680, height: 340, x: [-1, 7], y: [0, 80], xLabel: 'w', yLabel: 'L(w)', ariaLabel: '…' });
//   p.axes({ xTicks: [0, 2, 4, 6], yTicks: [0, 20, 40, 60] });
//   p.curve(pts, { class: 'plot-curve' });          // pts = [[x, y], …]
//   p.dot(x, y, { class: 'plot-dot', r: 6 });
//   p.tangent(x0, y0, slope, { half: 1.5 });        // line through (x0, y0) with the given slope, clipped to the plot
//   p.heat(grid, { color });                        // grid = { xs: [...], ys: [...], z: z[iy][ix] }; cells tile the plot exactly; colour via color(z) -> css colour
//   p.layer('name') -> <g> you can clear with p.clear('name'); every draw call accepts { layer: 'name' }.
// Coordinates are DATA coordinates; p.sx/p.sy map them to pixels.
//
// Size. `width`/`height` are the plot's DESIGN size in px (the viewBox, so one unit is one CSS pixel and text keeps its 11-12px size).
// A host wider than that scales the SVG up to fill it, but never taller than 78% of the viewport (the SVG's max-width is its aspect
// times that height, so there is no letterboxing). A host NARROWER than `width` at mount (a phone) gets the plot built at the host's
// width instead (never below `minWidth`, default 260; the height shrinks too but keeps a readable aspect), so text is never scaled
// down. A host with no width yet (detached, hidden) gets the design size; create the plot inside the element it will live in.
// Pass `responsive: false` to keep the design size whatever the host is (the viewBox then scales down, and the text with it).
// Resizing. Give `onResize(plot, {width, height})` to be told when the host width changes the size (a ResizeObserver; rotating
// a phone, resizing a window): the plot has then already cleared its axes and every layer and updated sx/sy/width/height/plotArea
// IN PLACE (so references you hold stay valid), and the callback must draw its content again - the same code that drew it first.
// Without `onResize` the size is decided once, at mount. `plot.destroy()` stops observing.
// Ids. Every plot gets its own id prefix for its clip path and (lazily, when `line(..., {arrow: true})` is first used) its arrow
// marker, so any number of plots can share a page.
import { svg, el } from '../core/dom.js';

let plotSeq = 0;
const MAX_VH_SHARE = 0.78;   // a plot never grows taller than this share of the viewport height

export function createPlot(host, opts) {
  const { width: W0 = 680, height: H0 = 340, x, y, xLabel = '', yLabel = '', ariaLabel = 'Plot', margin = {}, responsive = true, minWidth = 260, onResize } = opts;
  const m = { l: 52, r: 16, t: 14, b: 40, ...margin };
  const id = `plot${++plotSeq}`;
  let width = W0, height = H0, iw = 0, ih = 0;
  // design size -> actual size for a host of `hw` px
  const sizeFor = (hw) => {
    if (!responsive || !(hw > 0) || hw >= W0 - 1) return [W0, H0];
    const w = Math.round(Math.max(Math.min(minWidth, W0), hw));
    return [w, Math.round(Math.min(H0, Math.max((H0 * w) / W0, w * 0.7)))];
  };
  const sx = (v) => m.l + ((v - x[0]) / (x[1] - x[0])) * iw;
  const sy = (v) => m.t + ih - ((v - y[0]) / (y[1] - y[0])) * ih;
  const root = svg('svg', { class: 'plot', role: 'img', 'aria-label': ariaLabel });
  const clipId = `${id}-clip`;
  const clipRect = svg('rect', {});
  root.append(svg('defs', {}, svg('clipPath', { id: clipId }, clipRect)));
  const layers = {};
  const axesGroups = [];
  const layer = (name = 'main') => {
    if (!layers[name]) { layers[name] = svg('g', { class: `layer-${name}`, 'clip-path': `url(#${clipId})` }); root.append(layers[name]); }
    return layers[name];
  };
  const target = (o) => layer(o?.layer ?? 'main');
  let arrowId = null;
  const arrowRef = () => {   // the arrow marker exists only in plots that draw an arrow, under an id that is unique on the page
    if (!arrowId) {
      arrowId = `${id}-arrow`;
      root.prepend(svg('defs', {}, svg('marker', { id: arrowId, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, svg('path', { d: 'M0,0 L10,5 L0,10 z', class: 'plot-arrow' }))));
    }
    return `url(#${arrowId})`;
  };
  // sets width/height/iw/ih, the viewBox, the clip rectangle and the max-width cap (aspect x viewport height: the plot is never taller
  // than MAX_VH_SHARE of the viewport, and since it shrinks in width with its aspect there is no letterboxing)
  function setSize(w, h) {
    width = w; height = h; iw = w - m.l - m.r; ih = h - m.t - m.b;
    root.setAttribute('viewBox', `0 0 ${w} ${h}`);
    for (const [k, v] of Object.entries({ x: m.l, y: m.t, width: iw, height: ih })) clipRect.setAttribute(k, v);
    root.style.maxWidth = `calc(${(w / h).toFixed(4)} * ${MAX_VH_SHARE * 100}vh)`;
    api.width = w; api.height = h; api.plotArea = { l: m.l, t: m.t, w: iw, h: ih };
  }
  const api = {
    root, sx, sy, width, height, plotArea: null,
    layer,
    clear(name) { layers[name]?.replaceChildren(); },
    axes({ xTicks = ticks(x), yTicks = ticks(y), fmt = fmtTick } = {}) {
      const g = svg('g', { class: 'plot-axes' });
      for (const t of xTicks) g.append(svg('line', { class: 'plot-grid', x1: sx(t), x2: sx(t), y1: m.t, y2: m.t + ih }), svg('text', { class: 'plot-tick', x: sx(t), y: m.t + ih + 16, 'text-anchor': 'middle' }, fmt(t)));
      for (const t of yTicks) g.append(svg('line', { class: 'plot-grid', x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t) }), svg('text', { class: 'plot-tick', x: m.l - 6, y: sy(t) + 4, 'text-anchor': 'end' }, fmt(t)));
      g.append(svg('rect', { class: 'plot-frame', x: m.l, y: m.t, width: iw, height: ih, fill: 'none' }));
      if (xLabel) g.append(svg('text', { class: 'plot-label', x: m.l + iw / 2, y: height - 4, 'text-anchor': 'middle' }, xLabel));
      if (yLabel) g.append(svg('text', { class: 'plot-label', transform: `translate(13 ${m.t + ih / 2}) rotate(-90)`, 'text-anchor': 'middle' }, yLabel));
      root.insertBefore(g, root.children[1]);   // axes sit below every layer
      axesGroups.push(g);
      return g;
    },
    curve(pts, o = {}) {
      const d = pts.map(([px, py], i) => `${i ? 'L' : 'M'}${sx(px).toFixed(2)},${sy(py).toFixed(2)}`).join(' ');
      return target(o).appendChild(svg('path', { class: o.class ?? 'plot-curve', d, fill: 'none' }));
    },
    polyline(pts, o = {}) { return api.curve(pts, o); },
    dot(px, py, o = {}) {
      const c = target(o).appendChild(svg('circle', { class: o.class ?? 'plot-dot', cx: sx(px), cy: sy(py), r: o.r ?? 5 }));
      if (o.title) c.append(svg('title', {}, o.title));
      return c;
    },
    line(x1, y1, x2, y2, o = {}) {
      return target(o).appendChild(svg('line', { class: o.class ?? 'plot-line', x1: sx(x1), y1: sy(y1), x2: sx(x2), y2: sy(y2), 'marker-end': o.arrow ? arrowRef() : null }));
    },
    // line through (x0, y0) with slope `slope` (data units), drawn over x0 ± half
    tangent(x0, y0, slope, o = {}) {
      const h = o.half ?? (x[1] - x[0]) / 6;
      return api.line(x0 - h, y0 - slope * h, x0 + h, y0 + slope * h, { class: o.class ?? 'plot-tangent', layer: o.layer });
    },
    text(px, py, s, o = {}) {
      return target(o).appendChild(svg('text', { class: o.class ?? 'plot-text', x: sx(px) + (o.dx ?? 0), y: sy(py) + (o.dy ?? 0), 'text-anchor': o.anchor ?? 'start' }, s));
    },
    // One rectangle per grid point. A cell reaches halfway to its neighbours, and the first/last cell reaches as far outward as it
    // does inward, so the cells tile the plot with no gap and no overlap (the layer is clipped to the plot area).
    heat(grid, { color, layer: ln } = {}) {
      const g = target({ layer: ln ?? 'heat' });
      const { xs, ys, z } = grid;
      const ex = cellEdges(xs.map(sx)), ey = cellEdges(ys.map(sy));
      for (let iy = 0; iy < ys.length; iy++) for (let ix = 0; ix < xs.length; ix++) {
        g.append(svg('rect', { class: 'plot-cell', x: Math.min(ex[ix], ex[ix + 1]), y: Math.min(ey[iy], ey[iy + 1]), width: Math.abs(ex[ix + 1] - ex[ix]), height: Math.abs(ey[iy + 1] - ey[iy]), 'shape-rendering': 'crispEdges', fill: color ? color(z[iy][ix]) : '#456' }));
      }
      return g;
    },
    destroy() { ro?.disconnect(); },
  };
  setSize(...sizeFor(host?.clientWidth));
  host.append(root);

  let ro = null;
  if (onResize && typeof ResizeObserver !== 'undefined') {
    ro = new ResizeObserver(() => {
      if (!host.isConnected) return;
      if (!root.isConnected) { ro.disconnect(); return; }
      const [w, h] = sizeFor(host.clientWidth);
      if (Math.abs(w - width) < 2) return;
      axesGroups.splice(0).forEach((g) => g.remove());
      Object.values(layers).forEach((g) => g.replaceChildren());
      setSize(w, h);
      onResize(api, { width: w, height: h });
    });
    ro.observe(host);
  }
  return api;
}

// Cell boundaries for grid points at pixel positions `p` (n points -> n+1 edges): halfway between neighbours, and the outer edges as
// far outside the first/last point as the neighbouring edge is inside, so n cells tile [e0, en] with no gap (heat()).
export function cellEdges(p) {
  const n = p.length;
  if (n === 1) return [p[0] - 0.5, p[0] + 0.5];
  const e = [p[0] - (p[1] - p[0]) / 2];
  for (let i = 1; i < n; i++) e.push((p[i - 1] + p[i]) / 2);
  e.push(p[n - 1] + (p[n - 1] - p[n - 2]) / 2);
  return e;
}

export function ticks([a, b], n = 5) {
  const span = b - a, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(Number(t.toPrecision(12)));
  return out;
}
// Axis/table number: 4 significant digits, exponent form (1.5e6, 2e-4) beyond 1e6 and below 1e-3 instead of a long string of digits.
export function fmtTick(t) {
  if (!Number.isFinite(t) || t === 0) return String(t === 0 ? 0 : t);
  const a = Math.abs(t);
  if (Number(a.toPrecision(4)) >= 1e6 || a < 1e-3) return Number(t.toPrecision(3)).toExponential().replace('e+', 'e');
  return String(Number(t.toPrecision(4)));
}

// "Show as table" toggle: a button + a <table> built from the same rows the plot used (columns: [header,…], rows: [[…],…]).
export function tableToggle(host, { columns, rows, caption = 'Plotted values', fmt = (v) => (typeof v === 'number' ? fmtTick(v) : String(v)) }) {
  const wrap = el('div', { class: 'plot-table' });
  const table = el('table', {}, el('caption', {}, caption),
    el('thead', {}, el('tr', {}, columns.map((c) => el('th', { scope: 'col' }, c)))),
    el('tbody', {}, rows.map((r) => el('tr', {}, r.map((v) => el('td', {}, fmt(v)))))));
  table.hidden = true;
  const btn = el('button', { type: 'button', class: 'btn small', 'aria-expanded': 'false', onclick: () => { table.hidden = !table.hidden; btn.setAttribute('aria-expanded', String(!table.hidden)); btn.textContent = table.hidden ? 'Show as table' : 'Hide table'; } }, 'Show as table');
  const scroller = el('div', { class: 'plot-table-scroll' }, table);   // a wide table scrolls sideways inside this box, never the page
  wrap.append(btn, scroller);
  host.append(wrap);
  return { wrap, table, scroller, update(newRows) { table.querySelector('tbody').replaceChildren(...newRows.map((r) => el('tr', {}, r.map((v) => el('td', {}, fmt(v)))))); } };
}
