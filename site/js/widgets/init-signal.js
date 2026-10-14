// widgets/init-signal.js — the two "why initialise like that?" widgets.
//
//   mountWeightHistogram(host, D, ctx)  weights TorC drew for a few discrete layer choices: histogram vs. the expected bell curve
//   mountInitSignal(host, D, ctx)       six layers of Linear(256,256)+ReLU: std of the activations per layer for three inits (+ seed band)
//
// BOTH ARE PRECOMPUTED by the real TorC run (site/data/linear.json: init_stats, signal). A browser cannot reproduce C++'s random number
// generator, so nothing is redrawn here; the learner chooses among the recorded cases. The only arithmetic done in the browser is drawing
// the expected normal curve (a formula) next to the recorded counts. Every plot has a "show as table" toggle.
import { el, svg } from '../core/dom.js';
import { createPlot, ticks, tableToggle } from './plot.js';
import { useBaseStyle, injectStyle, sourceBadge, segmented, uid } from './linear-ui.js';
import { fmtShape } from './linear-shapes.js';
import { histBins, statsLabel, fmtSci, signalSeries, signalTable, log10, decadeTicks, expLabel } from './linear-signal-data.js';

const CSS = `
.lin-is .is-key{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.84rem;color:#c4ccd5;margin:.3rem 0}
.lin-is .is-key span.sw{display:inline-block;vertical-align:middle;margin-right:.3rem;font-size:.95rem}
.lin-is .sw.bar{width:14px;height:12px;background:rgba(88,196,221,.55);border:1px solid var(--fwd)}
.lin-is .sw.exp{width:22px;border-top:3px dashed var(--warn);height:0}
.lin-is .sw.band{width:18px;height:12px;background:color-mix(in srgb,currentColor 25%,transparent);border:1px solid currentColor}
.lin-is .ih-bar{fill:rgba(88,196,221,.55);stroke:var(--fwd);stroke-width:1}
.lin-is .ih-exp{stroke:var(--warn);stroke-width:2.2;stroke-dasharray:6 4;fill:none}
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
.lin-is dl.is-facts{display:grid;grid-template-columns:auto 1fr;gap:.15rem .9rem;margin:.4rem 0;font-size:.9rem}
@media(min-width:900px){.lin-is dl.is-facts{grid-template-columns:auto 1fr auto 1fr;gap:.15rem 1.2rem}}
.lin-is dl.is-facts dt{color:var(--muted)}
.lin-is dl.is-facts dd{margin:0;font-family:ui-monospace,monospace;color:var(--fg)}
.lin-is .is-vals{margin:.4rem 0;font-size:.9rem}
.lin-is .is-vals div{margin:.25rem 0;font-family:ui-monospace,monospace;font-size:.85rem;color:#c4ccd5;overflow-wrap:anywhere}
.lin-is .is-vals b{font-family:system-ui;color:var(--fg)}
.lin-is select{font:inherit;font-size:.95rem;color:var(--fg);background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:.3rem .5rem;max-width:100%}
.lin-is .plot-table button{margin-top:.3rem}
.lin-is .plot-table th,.lin-is .plot-table td{white-space:nowrap}
`;

const plotWidth = (host) => Math.max(300, Math.min(860, Math.round((host.clientWidth || 640) - 28)));
const watchWidth = (host, redraw) => {
  if (typeof ResizeObserver === 'undefined') return () => {};
  let last = plotWidth(host);
  const ro = new ResizeObserver(() => { const w = plotWidth(host); if (Math.abs(w - last) > 8) { last = w; redraw(); } });
  ro.observe(host);
  return () => ro.disconnect();
};

// =========================================================================================================
export function mountWeightHistogram(host, D) {
  useBaseStyle();
  injectStyle('lin-init-style', CSS);
  host.classList.add('lin-is');
  const stats = D.files.linear.init_stats;
  let cur = Math.max(0, stats.findIndex((s) => s.in === 784 && s.kaiming_default));
  const selId = uid('ih-sel');
  const select = el('select', { id: selId, 'data-key': 'ih-select', onchange: () => { cur = Number(select.value); draw(); } },
    stats.map((s, i) => el('option', { value: String(i), selected: i === cur }, statsLabel(s))));
  const plotBox = el('div'), facts = el('div'), extra = el('div');
  const key = el('div', { class: 'is-key' },
    el('span', {}, el('span', { class: 'sw bar' }), 'observed: weights per bin'),
    el('span', {}, el('span', { class: 'sw exp' }), 'expected for a normal distribution with the expected σ'));
  host.replaceChildren(
    el('div', { class: 'lin-head' }, el('h3', {}, 'The weights TorC drew'), sourceBadge('pre', 'Precomputed from the real TorC run · pick a recorded case')),
    el('div', { class: 'lin-row' }, el('label', { for: selId, class: 'lin-note dim', style: 'margin:0' }, 'Layer:'), select),
    plotBox, key, facts, extra);
  const tbl = tableToggle(host, { columns: ['bin from', 'bin to', 'observed weights', 'expected weights'], rows: [], caption: 'Histogram bins (observed counts from TorC; expected counts from the normal curve)', fmt: (v) => (typeof v === 'number' ? fmtSci(v) : String(v)) });

  function draw() {
    const s = stats[cur];
    const sigma = s.expected_std;
    const bins = histBins(s.histogram, s.numel, sigma);
    const ymaxData = Math.max(...bins.map((b) => Math.max(b.count, b.expected)));
    const yt = ticks([0, ymaxData * 1.12], 4);
    const ymax = Math.max(yt[yt.length - 1], ymaxData * 1.08);
    const w = plotWidth(host);
    const p = createPlot(plotBox, { width: w, height: Math.round(Math.max(w * 0.33 + 20, 250)), x: [s.histogram.lo, s.histogram.hi], y: [0, ymax], xLabel: 'weight value', yLabel: w < 480 ? 'count' : 'weights per bin',
      ariaLabel: `Histogram of the ${s.numel} weights of ${statsLabel(s)}; ${s.numel > 100 ? 'bell shaped, centred on zero' : 'a handful of values'}. A table view follows.`, margin: { l: 56 } });
    plotBox.replaceChildren(p.root);
    p.axes({ xTicks: [-4, -2, 0, 2, 4].map((k) => k * sigma), yTicks: yt, fmt: fmtSci });
    const bars = p.layer('bars');
    for (const b of bins) if (b.count > 0) bars.append(svg('rect', { class: 'ih-bar', x: p.sx(b.lo) + 0.5, y: p.sy(b.count), width: Math.max(1, p.sx(b.hi) - p.sx(b.lo) - 1), height: p.sy(0) - p.sy(b.count) }));
    p.curve(bins.map((b) => [b.mid, b.expected]), { layer: 'curve', class: 'ih-exp' });

    const expl = s.kaiming_default ? `√(2 / ${s.in}) = ${fmtSci(sigma)}` : `init_std = ${fmtSci(sigma)}`;
    facts.replaceChildren(el('dl', { class: 'is-facts' },
      el('dt', {}, 'weights drawn'), el('dd', {}, `${s.numel} (seed ${s.seed})`),
      el('dt', {}, 'expected σ'), el('dd', {}, expl),
      el('dt', {}, 'observed σ'), el('dd', {}, `${fmtSci(s.std)}  (${fmtSci(s.std_over_expected)} × expected)`),
      el('dt', {}, 'observed mean'), el('dd', {}, fmtSci(s.mean)),
      el('dt', {}, 'smallest / largest'), el('dd', {}, `${fmtSci(s.min)} / ${fmtSci(s.max)}`),
      el('dt', {}, 'bias'), el('dd', {}, s.bias_all_zero ? `all ${s.out} values exactly 0` : 'not all zero')));
    const lines = [];
    if (s.histogram.under + s.histogram.over > 0) lines.push(`${s.histogram.under} weights lie below −4σ and ${s.histogram.over} above +4σ, outside the plot.`);
    if (s.numel < 100) lines.push(`With only ${s.numel} weights the histogram is lumpy: a bell shape needs many draws (try the larger layers).`);
    if (s.weights) lines.push(`All ${s.numel} weights are shipped; ${s.numel <= 12 ? `here they are (row by row, shape ${fmtShape(s.weight_shape)}): ${s.weights.values.map((v) => fmtSci(v)).join(', ')}.` : 'the histogram counts them.'}`);
    extra.replaceChildren(...lines.map((t) => el('p', { class: 'lin-note dim', style: 'margin:.2rem 0' }, t)));
    tbl.table.querySelector('caption').textContent = `Histogram bins for ${statsLabel(s)} (observed counts from TorC; expected counts from the normal curve)`;
    tbl.update(bins.map((b) => [b.lo, b.hi, b.count, b.expected]));
  }
  draw();
  const stop = watchWidth(host, draw);
  return { destroy: stop };
}

// =========================================================================================================
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
    { value: 'all', label: 'Compare all three (log scale)' },
  ];
  let view = 'kaiming';
  const seg = segmented({ label: 'Show', name: 'is-view', value: view, options: views, onChange: (v) => { view = v; draw(); } });
  const plotBox = el('div'), keyBox = el('div', { class: 'is-key' }), vals = el('div', { class: 'is-vals' });
  const first = sig[0];
  const setup = `Setup: a batch of ${first.batch} samples with ${first.width} random numbers each (drawn from a normal distribution with mean 0 and standard deviation 1) goes through six copies of Linear(${first.width}, ${first.width}) + ReLU, where ReLU replaces every negative number by 0. The plotted number is the standard deviation (the spread) of the ReLU output over all of its values; layer 0 is the input itself. The log scale puts equal multiplication factors at equal distances. ` +
    `Band: lowest to highest value over ${first.n_extra_seeds} other random draws of the weights.`;
  host.replaceChildren(
    el('div', { class: 'lin-head' }, el('h3', {}, 'The signal through six layers'), sourceBadge('pre', 'Precomputed from the real TorC run · choose a recorded case')),
    seg.root, plotBox, keyBox, vals, el('p', { class: 'lin-note dim' }, setup));
  const t = signalTable(series);
  const tbl = tableToggle(host, { columns: t.columns, rows: t.rows, caption: 'Standard deviation of the activations after each layer (layer 0 is the input itself)', fmt: (v) => String(v) });

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
      ariaLabel: `Standard deviation of the activations over six layers for ${list.map((s) => s.label).join(', ')}. A table view follows.` });
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
      ...list.map((s) => el('span', {}, el('span', { class: 'sw', style: `color:var(--${cssVar(s.name)})` }, MARK[s.name]), `${s.label}, first run${s.name === 'kaiming' ? '' : s.name === 'std_0.01' ? ' (dashed line)' : ' (dotted line)'}`)),
      list.length === 1 && list[0].band.length ? el('span', {}, el('span', { class: 'sw mean' }), `white line: average of the ${first.n_extra_seeds} other draws`) : null,
      list.some((s) => s.band.length) ? el('span', { style: list.length === 1 ? `color:var(--${cssVar(list[0].name)})` : 'color:var(--muted)' }, el('span', { class: 'sw band' }), `shaded band${list.length > 1 ? 's' : ''}: lowest to highest over ${first.n_extra_seeds} other random draws`) : null].filter(Boolean));
    vals.replaceChildren(...list.flatMap((s) => [
      el('div', {}, el('b', {}, `${s.label}, first run: `), s.points.filter((q) => q.layer >= 1).map((q) => fmtSci(q.std)).join(' → ')),
      list.length === 1 && s.band.length ? el('div', {}, el('b', {}, `${first.n_extra_seeds}-draw average: `), s.band.map((b) => fmtSci(b.mean)).join(' → ')) : null].filter(Boolean)));
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
