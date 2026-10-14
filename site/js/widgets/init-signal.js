// widgets/init-signal.js — "why initialise like that?": six layers of Linear(256,256)+ReLU, the std of the activations per layer for
// three initialisations (+ a band over other seeds). Precomputed by the real TorC run (site/data/linear.json: signal): a browser cannot
// reproduce C++'s random number generator, so the learner picks among the recorded cases. Values also go to a screen-reader table.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable } from './plot.js';
import { useBaseStyle, injectStyle, segmented } from './linear-ui.js';
import { fmtSci, signalSeries, signalTable, log10, decadeTicks, expLabel } from './linear-signal-data.js';

const CSS = `
.lin-is .is-key{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.84rem;color:#c4ccd5;margin:.3rem 0}
.lin-is .is-key span.sw{display:inline-block;vertical-align:middle;margin-right:.3rem;font-size:.95rem}
.lin-is .sw.band{width:18px;height:12px;background:color-mix(in srgb,currentColor 25%,transparent);border:1px solid currentColor}
.lin-is .is-line{fill:none;stroke-width:2.4}
.lin-is .is-band{stroke:none;fill-opacity:.2}
.lin-is .is-line.s-kaiming{stroke:var(--fwd)}
.lin-is .is-line.s-std_0\\.01{stroke:var(--warn);stroke-dasharray:8 4}
.lin-is .is-line.s-std_1\\.0{stroke:var(--bwd);stroke-dasharray:2 4;stroke-linecap:round}
.lin-is .is-band.s-kaiming,.lin-is .is-marker.s-kaiming{fill:var(--fwd)}
.lin-is .is-band.s-std_0\\.01,.lin-is .is-marker.s-std_0\\.01{fill:var(--warn)}
.lin-is .is-band.s-std_1\\.0,.lin-is .is-marker.s-std_1\\.0{fill:var(--bwd)}
.lin-is .is-marker{stroke:var(--bg);stroke-width:1.5}
.lin-is .is-mean{fill:none;stroke:var(--fg);stroke-width:1.6}
.lin-is .sw.mean{width:22px;border-top:2px solid var(--fg);height:0}
.lin-is .is-ylab{fill:var(--muted);font-size:11px;text-anchor:end}
`;

const plotWidth = (host) => Math.max(300, Math.min(860, Math.round((host.clientWidth || 640) - 28)));
const watchWidth = (host, redraw) => {
  if (typeof ResizeObserver === 'undefined') return () => {};
  let last = plotWidth(host);
  const ro = new ResizeObserver(() => { const w = plotWidth(host); if (Math.abs(w - last) > 8) { last = w; redraw(); } });
  ro.observe(host);
  return () => ro.disconnect();
};

const cssVar = (name) => (name === 'kaiming' ? 'fwd' : name === 'std_0.01' ? 'warn' : 'bwd');
const MARK = { kaiming: '●', 'std_0.01': '■', 'std_1.0': '▲' };

export function mountInitSignal(host, D) {
  useBaseStyle();
  injectStyle('lin-init-style', CSS);
  host.classList.add('lin-is');
  const sig = D.files.linear.signal;
  const series = signalSeries(sig);
  const views = [
    { value: 'kaiming', label: 'Kaiming (the default)' },
    { value: 'std_0.01', label: 'init_std 0.01' },
    { value: 'std_1.0', label: 'init_std 1.0' },
    { value: 'all', label: 'All three (log scale)' },
  ];
  let view = 'kaiming';
  const seg = segmented({ label: 'Show', name: 'is-view', value: view, options: views, onChange: (v) => { view = v; draw(); } });
  const plotBox = el('div'), keyBox = el('div', { class: 'is-key' });
  const first = sig[0];
  host.replaceChildren(seg.root, plotBox, keyBox);
  const t = signalTable(series);
  const tbl = srTable(host, { columns: t.columns, rows: t.rows, caption: 'Standard deviation of the activations after each layer (layer 0 is the input itself)', fmt: (v) => String(v) });

  function shown() { return view === 'all' ? series : series.filter((s) => s.name === view); }

  function draw() {
    const list = shown();
    const logAxis = view !== 'kaiming';
    const allVals = list.flatMap((s) => [...s.points.map((q) => q.std), ...s.band.flatMap((b) => [b.min, b.max])]);
    const w = plotWidth(host), h = Math.round(Math.max(w * 0.4 + 24, 260));
    let yDom, yTicks;
    if (logAxis) {
      const lo = Math.floor(log10(Math.min(...allVals))) - 0.4, hi = Math.ceil(log10(Math.max(...allVals))) + 0.4;
      yDom = [lo, hi]; yTicks = decadeTicks(lo, hi, hi - lo > 8 ? 2 : 1);
    } else { yDom = [0, 1.2]; yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]; }
    const f = logAxis ? log10 : (v) => v;
    const p = createPlot(plotBox, { width: w, height: h, x: [-0.3, 6.3], y: yDom, xLabel: 'layer (0 = the input)', yLabel: w < 480 ? (logAxis ? 'std (log)' : 'std') : (logAxis ? 'std of the activations (log scale)' : 'std of the activations'), margin: { l: logAxis ? 62 : 56 },
      ariaLabel: `Standard deviation of the activations over six layers for ${list.map((s) => s.label).join(', ')}: ${list.map((s) => s.points.map((q) => fmtSci(q.std)).join(', ')).join('; ')}.` });
    plotBox.replaceChildren(p.root);
    p.axes({ xTicks: [0, 1, 2, 3, 4, 5, 6], yTicks: logAxis ? [] : yTicks, fmt: (v) => String(Number(v.toPrecision(3))) });
    if (logAxis) {
      const g = svg('g', { class: 'plot-axes' });
      for (const e of yTicks) g.append(svg('line', { class: 'plot-grid', x1: p.plotArea.l, x2: p.plotArea.l + p.plotArea.w, y1: p.sy(e), y2: p.sy(e) }), svg('text', { class: 'is-ylab', x: p.plotArea.l - 6, y: p.sy(e) + 4 }, expLabel(e)));
      p.root.insertBefore(g, p.root.children[2]);
    }
    for (const s of list) {
      const cls = `s-${s.name}`;
      if (s.band.length) {
        const top = s.band.map((b) => `${p.sx(b.layer).toFixed(1)},${p.sy(f(b.max)).toFixed(1)}`);
        const bot = s.band.slice().reverse().map((b) => `${p.sx(b.layer).toFixed(1)},${p.sy(f(b.min)).toFixed(1)}`);
        p.layer('band').append(svg('path', { class: `is-band ${cls}`, d: `M${[...top, ...bot].join(' L')} Z` }));
      }
      if (list.length === 1 && s.band.length) p.curve(s.band.map((b) => [b.layer, f(b.mean)]), { layer: 'lines', class: 'is-mean' });
      p.curve(s.points.map((q) => [q.layer, f(q.std)]), { layer: 'lines', class: `is-line ${cls}` });
      for (const q of s.points) {
        const x = p.sx(q.layer), y = p.sy(f(q.std));
        const m = marker(s.name, x, y, cls);
        m.append(svg('title', {}, `${s.label}, layer ${q.layer}: ${fmtSci(q.std)}`));
        p.layer('lines').append(m);
      }
    }
    keyBox.replaceChildren(...[
      ...list.map((s) => el('span', {}, el('span', { class: 'sw', style: `color:var(--${cssVar(s.name)})` }, MARK[s.name]), s.label)),
      list.length === 1 && list[0].band.length ? el('span', {}, el('span', { class: 'sw mean' }), `average of ${first.n_extra_seeds} other draws`) : null,
      list.some((s) => s.band.length) ? el('span', { style: list.length === 1 ? `color:var(--${cssVar(list[0].name)})` : 'color:var(--muted)' }, el('span', { class: 'sw band' }), `range over ${first.n_extra_seeds} other draws`) : null].filter(Boolean));
  }
  draw();
  const stop = watchWidth(host, draw);
  return { destroy: stop, tbl };
}

function marker(name, x, y, cls) {
  const r = 5;
  if (name === 'std_0.01') return svg('rect', { class: `is-marker ${cls}`, x: x - r, y: y - r, width: 2 * r, height: 2 * r });
  if (name === 'std_1.0') return svg('path', { class: `is-marker ${cls}`, d: `M${x},${y - r - 1} L${x + r + 1},${y + r} L${x - r - 1},${y + r} Z` });
  return svg('circle', { class: `is-marker ${cls}`, cx: x, cy: y, r: r });
}
