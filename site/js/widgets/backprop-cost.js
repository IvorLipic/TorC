// widgets/backprop-cost.js — the nudge-vs-backprop counter of the lesson backprop (brief §6, §9). Arithmetic only: for a model
// with N parameters, nudging needs 1 + N forward passes (one at the start, one per nudge), and backpropagation needs one forward
// pass and one backward pass. Each forward pass is drawn as one small square, so the digit network's pile is a large block.
// The parameter counts come from the data: the tracked leaves of scene `why` (the notebook model), `net_counts.parameters` of
// data/autograd_trace.json (the small network) and the anchor `mlp_parameters` (the digit network). The DOM is touched only in
// mountBackpropCost; costModels and gridFor are pure and tested in site/test/backprop.test.mjs.
import { el } from '../core/dom.js';

// number of parameters in a scene: every element of every leaf that training changes
export const sceneParameters = (scene) => scene.nodes.filter((n) => n.role === 'leaf' && n.requires_grad)
  .reduce((s, n) => s + n.value.values.length, 0);

export function costModels(D) {
  return [
    { id: 'notebook', label: 'the notebook model', parameters: sceneParameters(D.scenes.why) },
    { id: 'small', label: 'the small network', parameters: D.files.autograd_trace.net_counts.parameters },
    { id: 'digits', label: 'the digit network', parameters: D.anchors.mlp_parameters },
  ].map((m) => ({ ...m, nudgePasses: 1 + m.parameters, backpropPasses: { forward: 1, backward: 1 } }));
}

// A grid of `count` square cells in a box `width` px wide and at most `maxH` px high: the cell size (px), the gap and the columns.
export function gridFor(count, width, maxH = 200) {
  let cell = Math.floor(Math.sqrt((width * maxH) / count));
  cell = Math.max(1, Math.min(22, cell));
  const gap = cell >= 6 ? 2 : 0;
  const cols = Math.max(1, Math.floor((width + gap) / (cell + gap)));
  const rows = Math.ceil(count / cols);
  return { cell, gap, cols, rows, height: rows * (cell + gap) - gap };
}

// the backward pass drawn as this many forward squares: the TorC timing of docs/lessons/backprop.md found 2.7 to 3.6
export const BACKWARD_CELLS = 3.5;

export const thousands = (n) => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');

let seq = 0;

export function mountBackpropCost(host, D) {
  injectStyles();
  const models = costModels(D);
  const id = `bpc${++seq}`;
  let k = 0;
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const btns = models.map((m, i) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { k = i; draw(); announce(); } }, m.label));
  const controls = el('div', { class: 'bpc-ctl', role: 'group', 'aria-labelledby': `${id}-l` }, el('span', { class: 'bpc-lbl', id: `${id}-l` }, 'Model'), ...btns);
  const nudgeHead = el('p', { class: 'bpc-head' });
  const nudgeCanvas = el('canvas', { class: 'bpc-canvas', role: 'img' });
  const bpHead = el('p', { class: 'bpc-head' });
  const bpPic = el('canvas', { class: 'bpc-canvas', role: 'img', 'aria-label': 'One blue square for the forward pass and one red bar for the backward pass, at the same scale.' });
  const key = el('p', { class: 'hint' }, 'Each blue square is one forward pass, and both rows use the same scale. The red bar is the backward pass. In a TorC timing it cost about as much as three or four forward passes. A ring marks the two passes when they are too small to see.');
  host.replaceChildren(controls,
    el('div', { class: 'bpc-row' }, nudgeHead, nudgeCanvas),
    el('div', { class: 'bpc-row' }, bpHead, bpPic), key, status);

  function draw() {
    const m = models[k];
    btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === k)));
    nudgeHead.innerHTML = `<b>Nudging</b> ${thousands(m.nudgePasses)} forward passes, one at the start and one for each of the ${thousands(m.parameters)} ${m.parameters === 1 ? 'parameter' : 'parameters'}`;
    bpHead.innerHTML = '<b>Backpropagation</b> one forward pass and one backward pass';
    const w = Math.max(200, Math.min(640, Math.floor(nudgeCanvas.parentElement.clientWidth || 320)));
    const g = gridFor(m.nudgePasses, w);
    const dpr = globalThis.devicePixelRatio || 1;
    nudgeCanvas.width = Math.round(w * dpr); nudgeCanvas.height = Math.round(Math.max(g.height, 1) * dpr);
    nudgeCanvas.style.width = `${w}px`; nudgeCanvas.style.height = `${Math.max(g.height, 1)}px`;
    nudgeCanvas.setAttribute('aria-label', `${thousands(m.nudgePasses)} small squares, one for each forward pass that nudging needs.`);
    const ctx = nudgeCanvas.getContext?.('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, g.height);
    ctx.fillStyle = getComputedStyle(host).getPropertyValue('--fwd').trim() || '#58c4dd';
    const fwd = getComputedStyle(host).getPropertyValue('--fwd').trim() || '#58c4dd';
    const bwd = getComputedStyle(host).getPropertyValue('--bwd').trim() || '#ff6b5e';
    ctx.fillStyle = fwd;
    for (let i = 0; i < m.nudgePasses; i++) {
      const r = Math.floor(i / g.cols), c = i % g.cols;
      ctx.fillRect(c * (g.cell + g.gap), r * (g.cell + g.gap), g.cell, g.cell);
    }
    // backpropagation at the same scale: one forward square and a backward bar BACKWARD_CELLS squares long
    const h = Math.max(g.cell, 24), y0 = (h - g.cell) / 2, x0 = 14, bw = BACKWARD_CELLS * g.cell;
    bpPic.width = Math.round(w * dpr); bpPic.height = Math.round(h * dpr);
    bpPic.style.width = `${w}px`; bpPic.style.height = `${h}px`;
    const b = bpPic.getContext('2d');
    b.setTransform(dpr, 0, 0, dpr, 0, 0);
    b.clearRect(0, 0, w, h);
    b.fillStyle = fwd; b.fillRect(x0, y0, g.cell, g.cell);
    b.fillStyle = bwd; b.fillRect(x0 + g.cell + Math.max(g.gap, 1), y0, bw, g.cell);
    if (g.cell < 8) {
      b.strokeStyle = getComputedStyle(host).getPropertyValue('--fg').trim() || '#e6e6e6';
      b.lineWidth = 1.5;
      b.beginPath(); b.arc(x0 + (g.cell + bw) / 2, h / 2, 10, 0, 2 * Math.PI); b.stroke();
    }
  }
  function announce() {
    const m = models[k];
    status.textContent = `${m.label}. Nudging needs ${thousands(m.nudgePasses)} forward passes. Backpropagation needs one forward pass and one backward pass.`;
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
