// widgets/sgd-ui.js — small shared pieces for the sgd-step lesson widgets: one-time <style> injection, a labelled slider with a
// number box, an honest "where do these numbers come from" badge and a few sizing helpers.
// (The graph lab only reacts to the arrow keys when focus is in the lab or nowhere, so widgets need no arrow-key isolation.)
// Nothing here touches the DOM at import time (the lesson tests import every module in Node).
import { el } from '../core/dom.js';
import { fmt as baseFmt } from '../ops/texutil.js';

// Numbers shown to learners: texutil's fmt, but anything of size 1e7 or more (or NaN / infinity) in exponent form, never 19 digits.
export function fmt(x) {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'not a number' : (x > 0 ? 'infinity' : '-infinity');
  return Math.abs(x) >= 1e7 ? Number(x.toPrecision(3)).toExponential().replace('e+', 'e') : baseFmt(x);
}
// the same as a TeX number (1.23e18 -> 1.23\\times 10^{18})
export const texNum = (x) => { const t = fmt(x); const m = /^(-?[\d.]+)e(-?\d+)$/.exec(t); return m ? `${m[1]}\\times 10^{${m[2]}}` : t; };
export const texPar = (x) => (x < 0 ? `(${texNum(x)})` : texNum(x));

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

// A horizontally scrolling box must be reachable without a pointer and must show that there is more: role=region + a name, tabindex=0
// while the content is wider than the box (kept up to date by observers), and a shadow on the cut edge (CSS below).
const watched = new WeakSet();
export function enhanceScroller(box, label) {
  if (typeof document === 'undefined' || watched.has(box)) return;
  watched.add(box);
  box.setAttribute('role', 'region');
  const name = () => label ?? box.querySelector('caption')?.textContent ?? 'Scrollable content';
  const sync = () => {
    box.setAttribute('aria-label', name() + ' (scrolls sideways)');
    if (box.scrollWidth > box.clientWidth + 1) box.setAttribute('tabindex', '0'); else box.removeAttribute('tabindex');
  };
  if (typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(sync);
    const watch = () => { ro.disconnect(); ro.observe(box); for (const c of box.children) ro.observe(c); };
    new MutationObserver(() => { watch(); sync(); }).observe(box, { childList: true, subtree: true, attributes: true, attributeFilter: ['hidden'] });
    watch();
  }
  sync();
}
export function enhanceScrollers(root) {
  if (typeof document === 'undefined') return;
  root ??= document;
  root.querySelectorAll('.sgd-scroll, .sgd-w .math').forEach((b) => enhanceScroller(b, b.classList.contains('math') ? 'Equation' : undefined));
}

export const approx = (a, b) => Math.abs(a - b) <= 1e-9 * (1 + Math.abs(b));
export const round3 = (v) => Number(v.toPrecision(6));        // trims float noise (0.15000000000000002) without ever turning a small positive value into 0
export const uid = (p) => `${p}-${Math.random().toString(36).slice(2, 8)}`;

// kind: 'browser' (recomputed live by the JS port) | 'torc' (printed by the real library, shown as recorded)
export function badge(kind, text) {
  return el('p', { class: `sgd-badge ${kind}` }, el('span', { class: 'sgd-badge-k' }, kind === 'torc' ? 'Precomputed from TorC' : 'Recomputed in your browser'), ' ', text);
}

// A labelled range slider paired with a number box (both native inputs, both labelled). `validate(v)` may throw an Error whose
// message is shown (used to reproduce TorC's own constructor message for lr <= 0). marker = {value, label} draws a tick under the track.
export function sliderField({ id, label, min, max, step, value, onChange, validate, marker }) {
  const err = el('span', { class: 'sgd-err', role: 'status' });
  const range = el('input', { type: 'range', id: id + '-r', min, max, step, value, 'data-key': id + '-r' });
  const num = el('input', { type: 'number', id: id + '-n', step, value, 'data-key': id + '-n', inputmode: 'decimal', 'aria-label': label + ', type a number' });
  const root = el('div', { class: 'sgd-field' },
    el('label', { for: id + '-r' }, label),
    el('div', { class: 'sgd-slide' }, range,
      marker ? el('span', { class: 'sgd-mark', style: `left:calc(8px + (100% - 16px) * ${(marker.value - min) / (max - min)})`, 'aria-hidden': 'true' }, el('i'), marker.label) : null),
    num, err);
  const apply = (raw) => {
    const v = Number(raw);
    if (raw === '' || Number.isNaN(v)) return false;
    try { validate?.(v); } catch (e) { err.textContent = `${e.name === 'TorcError' ? 'TorcError' : 'Error'}: ${e.message}`; num.classList.add('bad'); return false; }
    err.textContent = ''; num.classList.remove('bad');
    onChange(v);
    return true;
  };
  range.addEventListener('input', () => { num.value = range.value; apply(range.value); });
  num.addEventListener('input', () => { if (apply(num.value)) range.value = num.value; });
  return {
    root, range, num,
    set(v) { range.value = v; num.value = v; err.textContent = ''; num.classList.remove('bad'); },
    describe(text) { range.setAttribute('aria-valuetext', text); },
  };
}

export const SHARED_CSS = `
.sgd-w{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.sgd-w h3{font-size:1.05rem;margin:0 0 .3rem}
.sgd-w .sgd-sub{margin:0 0 .5rem;color:var(--muted);font-size:.92rem}
.sgd-badge{margin:0 0 .6rem;font-size:.8rem;color:var(--muted);line-height:1.4}
.sgd-badge-k{display:inline-block;border:1px solid;border-radius:99px;padding:0 .55rem;margin-right:.3rem;font-weight:600;color:var(--fwd)}
.sgd-badge.torc .sgd-badge-k{color:var(--warn)}
.sgd-ctl{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:flex-end;margin:.2rem 0 .5rem}
.sgd-ctl .row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.sgd-field{display:grid;grid-template-columns:auto 5.6rem;gap:2px 10px;align-items:center;min-width:min(100%,300px)}
.sgd-field label{grid-column:1/-1;font-size:.88rem;color:#c4ccd5}
.sgd-slide{position:relative;padding-bottom:1.35rem;min-width:0}
.sgd-slide input[type=range]{width:100%;margin:0;accent-color:var(--fwd)}
.sgd-mark{position:absolute;top:1.3rem;transform:translateX(-50%);font-size:.7rem;color:var(--warn);white-space:nowrap;line-height:1}
.sgd-mark i{position:absolute;left:50%;top:-.95rem;height:.8rem;border-left:2px solid var(--warn)}
.sgd-field input[type=number]{background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:7px;padding:.2rem .4rem;font:.9rem ui-monospace,monospace;width:5.6rem}
.sgd-field input.bad{border-color:var(--bwd);background:#2b1513}
.sgd-err{grid-column:1/-1;color:#ffb3ab;font-size:.82rem;min-height:0}
.sgd-w .btn[aria-disabled=true]{opacity:.4;cursor:default}
.sgd-w .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.sgd-w .btn small{display:block;font-size:.7rem;color:var(--muted);line-height:1.1}
.sgd-status{min-height:2.7em;margin:.2rem 0 .6rem;padding:.35rem .7rem;border-left:3px solid var(--fwd);background:rgba(88,196,221,.07);border-radius:0 8px 8px 0;font-size:.93rem}
.sgd-status.warn{border-left-color:var(--warn);background:rgba(227,179,65,.08)}
.sgd-status.bad{border-left-color:var(--bwd);background:rgba(252,98,85,.09)}
.sgd-w .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sgd-tab{border-collapse:collapse;font-size:.85rem;width:100%}
.sgd-tab caption{text-align:left;color:var(--muted);padding-bottom:.2rem}
.sgd-tab th,.sgd-tab td{border-bottom:1px solid var(--border);padding:.22rem .5rem;text-align:right;font-family:ui-monospace,monospace}
.sgd-tab th{color:var(--muted);font-weight:600;font-family:inherit}
.sgd-tab th:first-child,.sgd-tab td:first-child{text-align:left}
.sgd-tab .katex{font-size:1em}
.sgd-tab th[scope=row]{white-space:nowrap}
.sgd-tab.text th,.sgd-tab.text td{text-align:left}
.sgd-scroll,.sgd-w .math{--cover:var(--panel);overflow-x:auto;
background:linear-gradient(to right,var(--cover) 30%,transparent) left center/28px 100% no-repeat local,
linear-gradient(to left,var(--cover) 30%,transparent) right center/28px 100% no-repeat local,
radial-gradient(farthest-side at 0 50%,rgba(255,255,255,.28),transparent) left center/12px 100% no-repeat scroll,
radial-gradient(farthest-side at 100% 50%,rgba(255,255,255,.28),transparent) right center/12px 100% no-repeat scroll}
.sgd-scroll.on-bg{--cover:var(--bg)}
.sgd-scroll:focus-visible,.sgd-w .math:focus-visible{outline:2px solid var(--fwd);outline-offset:2px}
.sgd-w details{margin:.5rem 0;color:#c4ccd5;font-size:.93rem}
.sgd-w summary{cursor:pointer;color:var(--fwd)}
@media(max-width:640px){.sgd-w{padding:.7rem .7rem .9rem}.sgd-field{min-width:100%}}
`;
