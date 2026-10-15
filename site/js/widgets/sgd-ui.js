// widgets/sgd-ui.js — small shared pieces for the sgd-step lesson widgets: one-time <style> injection, number format, sizing,
// preset buttons and a small log-scale loss chart.
// (The graph lab only reacts to the arrow keys when focus is in the lab or nowhere, so widgets need no arrow-key isolation.)
// Nothing here touches the DOM at import time (the lesson tests import every module in Node).
import { el, svg } from '../core/dom.js';
import { createPlot } from './plot.js';
import { fmt as baseFmt } from '../ops/texutil.js';

// Numbers shown to learners: texutil's fmt, but anything of size 1e7 or more (or NaN / infinity) in exponent form, never 19 digits.
export function fmt(x) {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'not a number' : (x > 0 ? 'infinity' : '-infinity');
  return Math.abs(x) >= 1e7 ? Number(x.toPrecision(3)).toExponential().replace('e+', 'e') : baseFmt(x);
}

export function injectStyle(id, css) {
  if (typeof document === 'undefined' || document.getElementById(id)) return;
  const s = document.createElement('style');
  s.id = id;
  s.textContent = css;
  document.head.append(s);
}

// width of a host in px, clamped (viewBox width for plots, so text keeps its size on a phone)
export const fitWidth = (host, min, max, dflt) => { const w = host?.clientWidth; return w ? Math.max(min, Math.min(max, Math.round(w))) : dflt; };
export function onResize(fn, signal) {
  if (typeof window === 'undefined') return;
  let raf = 0;
  window.addEventListener('resize', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(fn); }, { signal });
}

export const uid = (p) => `${p}-${Math.random().toString(36).slice(2, 8)}`;

// A labelled group of preset buttons; pick(v) is called with the preset's value. Returns {root, set(v)} to mark the pressed one.
export function presets(label, values, pick) {
  const btns = values.map((v) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(v) }, String(v)));
  return {
    root: el('div', { class: 'sgd-presets', role: 'group', 'aria-label': label }, el('span', { class: 'sgd-plabel', 'aria-hidden': 'true', html: label }), ...btns),
    set(v) { btns.forEach((b, i) => b.setAttribute('aria-pressed', String(values[i] === v))); },
  };
}

// decade range [lo, hi] (powers of ten) covering every positive finite value, at least two decades tall
export function yRange(values) {
  const v = values.filter((x) => x > 0 && Number.isFinite(x));
  let lo = Math.floor(Math.log10(Math.min(...v))), hi = Math.ceil(Math.log10(Math.max(...v)));
  if (hi - lo < 2) hi = lo + 2;
  return [lo, hi];
}
const fmtDecade = (d) => (d >= -3 && d <= 4 ? fmt(10 ** d) : `1e${d}`);

// Loss on a log scale against the update/iteration number. series = [{values, cls, dots?}] with values[i] at x = x0 + i; `range` is
// the values the y axis must cover (so the axis does not jump while a run is stepped through). Redraws from scratch into host.
export function logChart(host, { series, range, x0 = 0, n, xLabel, ariaLabel }) {
  host.replaceChildren();
  const W = fitWidth(host, 280, 460, 420), H = Math.round(W * 0.55), m = { l: 56, r: 12, t: 12, b: 38 };
  const [lo, hi] = yRange(range);
  const p = createPlot(host, { width: W, height: H, x: [x0 - 0.3, x0 + n + 0.3], y: [lo, hi], xLabel, yLabel: 'loss', margin: m, ariaLabel, responsive: false });
  p.axes({ xTicks: Array.from({ length: Math.floor(n / 2) + 1 }, (_, i) => x0 + 2 * i), yTicks: [] });
  const every = Math.max(1, Math.ceil((hi - lo) / 6));
  for (let d = lo; d <= hi; d += every) p.root.append(svg('line', { class: 'lc-grid', x1: m.l, x2: W - m.r, y1: p.sy(d), y2: p.sy(d) }), svg('text', { class: 'lc-t', x: m.l - 6, y: p.sy(d) + 3.5, 'text-anchor': 'end' }, fmtDecade(d)));
  for (const s of series) {
    const pts = s.values.map((v, i) => [x0 + i, Math.log10(Math.max(v, 1e-12))]).filter(([, y]) => Number.isFinite(y) && y <= hi);
    if (pts.length > 1) p.curve(pts, { class: s.cls });
    if (s.dots) pts.forEach(([x, y]) => p.dot(x, y, { class: 'lc-dot', r: 3.4 }));
  }
  return p;
}

export const SHARED_CSS = `
.sgd-w{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.sgd-ctl{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;margin:.1rem 0 .5rem}
.sgd-ctl .row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.sgd-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.sgd-plabel{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.sgd-w .btn[aria-disabled=true]{opacity:.4;cursor:default}
.sgd-w .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.sgd-w .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sgd-w .sub{margin:.3rem 0 .15rem;color:var(--muted);font-size:.85rem}
.sgd-w .chart svg{display:block;width:100%;height:auto;max-height:240px}
.sgd-w .chart text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sgd-w .lc-main{fill:none;stroke:var(--fwd);stroke-width:2.5}
.sgd-w .lc-ghost{fill:none;stroke:var(--muted);stroke-width:1.8;stroke-dasharray:5 4}
.sgd-w .lc-dot{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.sgd-w .lc-grid{stroke:var(--border);stroke-width:.6}
.sgd-w .lc-t{fill:var(--muted);font-size:10.5px}
.sgd-read{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-family:ui-monospace,monospace;font-size:.92rem;margin:.4rem 0 0}
.sgd-read b{color:var(--muted);font-weight:600;margin-right:.3rem}
.sgd-read .bad{color:var(--bwd)}
.sgd-grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:10px 18px;align-items:start}
@media(max-width:900px){.sgd-grid{grid-template-columns:minmax(0,1fr)}}
@media(max-width:640px){.sgd-w{padding:.7rem .7rem .9rem}}
`;
