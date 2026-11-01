// widgets/backprop-cost.js — the nudge-vs-backprop counter of the lesson backprop (brief §6, §9). For each of three networks built
// like the digit network (data/backprop-timing.json, a TorC timing on one computer, see docs/trace-schema.md), nudging needs
// 1 + N forward passes (one at the start, one per nudge) and backpropagation needs one forward and one backward pass, which together
// cost `backprop_over_nudge` nudges in the timing. Both rows are drawn at ONE scale: a blue square (or pixel) is one forward pass of a
// nudge, and the red bar is as long as the measured cost of backpropagation. When a network has more forward passes than fit, one
// pixel stands for several passes, and the head says how many. The DOM is touched only in mountBackpropCost; costModels and gridFor
// are pure and tested in site/test/backprop.test.mjs.
import { el } from '../core/dom.js';

export const thousands = (n) => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

export function costModels(D) {
  const runs = D.files['backprop-timing'].runs;
  const label = (r, i) => (i === 0 ? 'the digit network' : `hidden layers of ${r.hidden}`);
  return runs.map((r, i) => ({ id: `h${r.hidden}`, label: label(r, i), parameters: r.parameters, nudgePasses: 1 + r.parameters, backpropCost: r.backprop_over_nudge }));
}

// The grid for `count` passes in a box `width` px wide and at most `maxH` px high: the cell size (px), the gap, the columns, the
// rows, and `per`, how many passes one cell stands for (1 unless even 1 px cells do not fit).
export function gridFor(count, width, maxH = 200) {
  const per = Math.max(1, Math.ceil(count / (width * maxH)));
  const cells = Math.ceil(count / per);
  let cell = Math.floor(Math.sqrt((width * maxH) / cells));
  cell = Math.max(1, Math.min(22, cell));
  const gap = cell >= 6 ? 2 : 0;
  const cols = Math.max(1, Math.floor((width + gap) / (cell + gap)));
  const rows = Math.ceil(cells / cols);
  return { cell, gap, cols, rows, cells, per, height: rows * (cell + gap) - gap };
}

// one significant figure, as the prose says it
const about = (x) => String(Number(x.toPrecision(1)));

let seq = 0;

export function mountBackpropCost(host, D) {
  injectStyles();
  const models = costModels(D);
  const id = `bpc${++seq}`;
  let k = 0;
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const btns = models.map((m, i) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { k = i; draw(); announce(); } }, m.label));
  const controls = el('div', { class: 'bpc-ctl', role: 'group', 'aria-labelledby': `${id}-l` }, el('span', { class: 'bpc-lbl', id: `${id}-l` }, 'Network'), ...btns);
  const nudgeHead = el('p', { class: 'bpc-head' });
  const nudgeCanvas = el('canvas', { class: 'bpc-canvas', role: 'img' });
  const bpHead = el('p', { class: 'bpc-head' });
  const bpPic = el('canvas', { class: 'bpc-canvas', role: 'img' });
  const key = el('p', { class: 'hint' });
  host.replaceChildren(controls,
    el('div', { class: 'bpc-row' }, nudgeHead, nudgeCanvas),
    el('div', { class: 'bpc-row' }, bpHead, bpPic), key, status);

  function draw() {
    const m = models[k];
    btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === k)));
    const w = Math.max(200, Math.min(640, Math.floor(nudgeCanvas.parentElement.clientWidth || 320)));
    const g = gridFor(m.nudgePasses, w);
    nudgeHead.innerHTML = `<b>Nudging</b> ${thousands(m.nudgePasses)} forward passes, one at the start and one for each parameter`;
    bpHead.innerHTML = `<b>Backpropagation</b> one forward and one backward pass, about as much work as ${about(m.backpropCost)} nudges`;
    key.textContent = g.per === 1
      ? 'Each blue square is one forward pass of a nudge. The red bar is backpropagation at the same scale. A ring marks it when it is too small to see.'
      : `Each blue pixel stands for ${g.per} forward passes of a nudge. The red bar is backpropagation at the same scale. A ring marks it when it is too small to see.`;
    nudgeCanvas.setAttribute('aria-label', `A block of ${thousands(m.nudgePasses)} forward passes.`);
    bpPic.setAttribute('aria-label', `A bar as long as ${about(m.backpropCost)} forward passes, at the same scale.`);
    const dpr = globalThis.devicePixelRatio || 1;
    const css = (name, dflt) => getComputedStyle(host).getPropertyValue(name).trim() || dflt;
    const fwd = css('--fwd', '#58c4dd'), bwd = css('--bwd', '#ff6b5e'), fg = css('--fg', '#e6e6e6');
    nudgeCanvas.width = Math.round(w * dpr); nudgeCanvas.height = Math.round(Math.max(g.height, 1) * dpr);
    nudgeCanvas.style.width = `${w}px`; nudgeCanvas.style.height = `${Math.max(g.height, 1)}px`;
    const ctx = nudgeCanvas.getContext?.('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, g.height);
    ctx.fillStyle = fwd;
    for (let i = 0; i < g.cells; i++) {
      const r = Math.floor(i / g.cols), c = i % g.cols;
      ctx.fillRect(c * (g.cell + g.gap), r * (g.cell + g.gap), g.cell, g.cell);
    }
    // backpropagation at the same scale: a bar as long as backpropCost cells of one pass each (divided by `per`)
    const len = Math.max((m.backpropCost / g.per) * g.cell, 0.6), h = Math.max(g.cell, 24), y0 = (h - g.cell) / 2, x0 = 14;
    bpPic.width = Math.round(w * dpr); bpPic.height = Math.round(h * dpr);
    bpPic.style.width = `${w}px`; bpPic.style.height = `${h}px`;
    const b = bpPic.getContext('2d');
    b.setTransform(dpr, 0, 0, dpr, 0, 0);
    b.clearRect(0, 0, w, h);
    b.fillStyle = bwd; b.fillRect(x0, y0, len, Math.max(g.cell, 1));
    if (len < 8) {
      b.strokeStyle = fg; b.lineWidth = 1.5;
      b.beginPath(); b.arc(x0 + len / 2, h / 2, 10, 0, 2 * Math.PI); b.stroke();
    }
  }
  function announce() {
    const m = models[k];
    status.textContent = `${m.label}. Nudging needs ${thousands(m.nudgePasses)} forward passes. Backpropagation costs about as much as ${about(m.backpropCost)} nudges.`;
  }
  draw();
  let lastW = host.clientWidth;
  const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => { if (Math.abs(host.clientWidth - lastW) > 8) { lastW = host.clientWidth; draw(); } }) : null;
  ro?.observe(host);
  return { destroy: () => ro?.disconnect(), choose: (i) => btns[i].click() };
}

const CSS = `
.bpc-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:.2rem 0 .6rem}
.bpc-lbl{color:var(--muted);font-size:.9rem;margin-right:.3rem}
.bpc-row{margin:.4rem 0 .8rem}
.bpc-head{margin:.2rem 0 .4rem;font-size:.95rem}
.bpc-head b{margin-right:.4rem}
.bpc-canvas{display:block;max-width:100%}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('bpc-styles')) return;
  document.head.append(el('style', { id: 'bpc-styles' }, CSS));
}
