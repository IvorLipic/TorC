// widgets/plot.js — a tiny SVG plotting helper (axes, curves, points, tangent lines, heat grids, legend-free) used by lessons.
// No charting library. Every plot should offer a table view (`tableToggle`) for accessibility and traceability.
//
//   const p = createPlot(host, { width: 680, height: 340, x: [-1, 7], y: [0, 80], xLabel: 'w', yLabel: 'L(w)', ariaLabel: '…' });
//   p.axes({ xTicks: [0, 2, 4, 6], yTicks: [0, 20, 40, 60] });
//   p.curve(pts, { class: 'plot-curve' });          // pts = [[x, y], …]
//   p.dot(x, y, { class: 'plot-dot', r: 6 });
//   p.tangent(x0, y0, slope, { half: 1.5 });        // line through (x0, y0) with the given slope, clipped to the plot
//   p.heat(grid, { color });                        // grid = { xs: [...], ys: [...], z: z[iy][ix] }; colour via color(z) -> css colour
//   p.layer('name') -> <g> you can clear with p.clear('name'); every draw call accepts { layer: 'name' }.
// Coordinates are DATA coordinates; p.sx/p.sy map them to pixels.
import { svg, el } from '../core/dom.js';

export function createPlot(host, opts) {
  const { width = 680, height = 340, x, y, xLabel = '', yLabel = '', ariaLabel = 'Plot', margin = {} } = opts;
  const m = { l: 52, r: 16, t: 14, b: 40, ...margin };
  const iw = width - m.l - m.r, ih = height - m.t - m.b;
  const sx = (v) => m.l + ((v - x[0]) / (x[1] - x[0])) * iw;
  const sy = (v) => m.t + ih - ((v - y[0]) / (y[1] - y[0])) * ih;
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, class: 'plot', role: 'img', 'aria-label': ariaLabel });
  const clip = `plotclip-${Math.random().toString(36).slice(2, 8)}`;
  root.append(svg('defs', {}, svg('clipPath', { id: clip }, svg('rect', { x: m.l, y: m.t, width: iw, height: ih }))));
  const layers = {};
  const layer = (name = 'main') => {
    if (!layers[name]) { layers[name] = svg('g', { class: `layer-${name}`, 'clip-path': `url(#${clip})` }); root.append(layers[name]); }
    return layers[name];
  };
  const target = (o) => layer(o?.layer ?? 'main');
  const api = {
    root, sx, sy, width, height, plotArea: { l: m.l, t: m.t, w: iw, h: ih },
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
      return target(o).appendChild(svg('line', { class: o.class ?? 'plot-line', x1: sx(x1), y1: sy(y1), x2: sx(x2), y2: sy(y2), 'marker-end': o.arrow ? 'url(#plot-arrow)' : null }));
    },
    // line through (x0, y0) with slope `slope` (data units), drawn over x0 ± half
    tangent(x0, y0, slope, o = {}) {
      const h = o.half ?? (x[1] - x[0]) / 6;
      return api.line(x0 - h, y0 - slope * h, x0 + h, y0 + slope * h, { class: o.class ?? 'plot-tangent', layer: o.layer });
    },
    text(px, py, s, o = {}) {
      return target(o).appendChild(svg('text', { class: o.class ?? 'plot-text', x: sx(px) + (o.dx ?? 0), y: sy(py) + (o.dy ?? 0), 'text-anchor': o.anchor ?? 'start' }, s));
    },
    heat(grid, { color, layer: ln } = {}) {
      const g = target({ layer: ln ?? 'heat' });
      const { xs, ys, z } = grid;
      const cw = (i) => (sx(xs[Math.min(i + 1, xs.length - 1)]) - sx(xs[Math.max(i - 1, 0)])) / (i === 0 || i === xs.length - 1 ? 2 : 2);
      for (let iy = 0; iy < ys.length; iy++) for (let ix = 0; ix < xs.length; ix++) {
        const w = Math.abs(cw(ix)), h = Math.abs((sy(ys[Math.min(iy + 1, ys.length - 1)]) - sy(ys[Math.max(iy - 1, 0)])) / 2);
        g.append(svg('rect', { class: 'plot-cell', x: sx(xs[ix]) - w / 2, y: sy(ys[iy]) - h / 2, width: w + 0.5, height: h + 0.5, fill: color ? color(z[iy][ix]) : '#456' }));
      }
      return g;
    },
  };
  root.prepend(svg('defs', {}, svg('marker', { id: 'plot-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, svg('path', { d: 'M0,0 L10,5 L0,10 z', class: 'plot-arrow' }))));
  host.append(root);
  return api;
}

export function ticks([a, b], n = 5) {
  const span = b - a, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(Number(t.toPrecision(12)));
  return out;
}
export const fmtTick = (t) => String(Number(t.toPrecision(4)));

// "Show as table" toggle: a button + a <table> built from the same rows the plot used (columns: [header,…], rows: [[…],…]).
export function tableToggle(host, { columns, rows, caption = 'Plotted values', fmt = (v) => (typeof v === 'number' ? fmtTick(v) : String(v)) }) {
  const wrap = el('div', { class: 'plot-table' });
  const table = el('table', {}, el('caption', {}, caption),
    el('thead', {}, el('tr', {}, columns.map((c) => el('th', { scope: 'col' }, c)))),
    el('tbody', {}, rows.map((r) => el('tr', {}, r.map((v) => el('td', {}, fmt(v)))))));
  table.hidden = true;
  const btn = el('button', { type: 'button', class: 'btn small', 'aria-expanded': 'false', onclick: () => { table.hidden = !table.hidden; btn.setAttribute('aria-expanded', String(!table.hidden)); btn.textContent = table.hidden ? 'Show as table' : 'Hide table'; } }, 'Show as table');
  wrap.append(btn, table);
  host.append(wrap);
  return { wrap, table, update(newRows) { table.querySelector('tbody').replaceChildren(...newRows.map((r) => el('tr', {}, r.map((v) => el('td', {}, fmt(v)))))); } };
}
