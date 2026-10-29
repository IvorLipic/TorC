// widgets/bowl.js — the one widget of the lesson why-gradients: the loss curve L(w) with the knob's dot and its tangent,
// a start slider, learning-rate presets and a Step button (w <- w - lr * gradient). Values are recomputed by the engine through
// landscape-model.js (the same code as the graph lab), and site/test/why-gradients.test.mjs checks that model against TorC's run.
// Controls are never re-rendered (focus stays put); no autoplay; the module touches the DOM only when mountBowl runs.
import { el } from '../core/dom.js';
import { math } from '../core/katex.js';
import { updateTex, fmtX as fmt } from './landscape-tex.js';
import { createPlot, ticks } from './plot.js';
import { makeModel } from './landscape-model.js';

export const PRESET_LRS = [0.05, 0.125, 0.25, 0.3];     // slow, exact, bounce, diverge (all four are runs TorC recorded)
export const MAX_STEPS = 40;
const X = [-1, 7], Y = [0, 72];

let seq = 0;

export function mountBowl(host, D) {
  injectStyles();
  const m = makeModel(D.scenes.why);
  const id = `bowl${++seq}`;
  let start = m.w0, lr = PRESET_LRS[0], path = [start];
  const cur = () => path[path.length - 1];

  // ---- controls ---------------------------------------------------------------------------------------------------------
  const startIn = el('input', { type: 'range', id: `${id}-w`, min: X[0], max: X[1], step: 0.05, value: start });
  const startOut = el('output', { for: `${id}-w`, class: 'bowl-val' });
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const restart = () => { path = [start]; status.textContent = ''; draw(); };
  startIn.addEventListener('input', () => { start = Number(startIn.value); restart(); });
  const lrBtns = PRESET_LRS.map((v) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { lr = v; restart(); } }, String(v)));
  const stepBtn = el('button', { type: 'button', class: 'btn primary', onclick: step }, 'Step');
  const resetBtn = el('button', { type: 'button', class: 'btn', onclick: () => restart() }, 'Reset');
  const controls = el('div', { class: 'bowl-ctl' },
    el('div', { class: 'bowl-field' }, el('label', { for: `${id}-w` }, 'start w'), startIn, startOut),
    el('div', { class: 'bowl-field', role: 'group', 'aria-label': 'Learning rate' }, el('span', { class: 'bowl-lbl', 'aria-hidden': 'true' }, 'lr'), ...lrBtns),
    el('div', { class: 'bowl-field' }, stepBtn, resetBtn), status);

  // ---- plot and readout ----------------------------------------------------------------------------------------------------
  const plotBox = el('div', { class: 'bowl-plot' });
  const readout = el('div', { class: 'bowl-read' });
  host.replaceChildren(controls, plotBox, readout);
  const curve = m.curve(X[0], X[1], 160).map((q) => [q.w, q.L]);
  const drawStatic = (pl) => {
    pl.axes({ xTicks: ticks(X, 8), yTicks: [0, 20, 40, 60] });
    pl.curve(curve, { class: 'plot-curve', layer: 'curve' });
    pl.layer('live');
  };
  const p = createPlot(plotBox, { width: 640, height: 300, x: X, y: Y, xLabel: 'w', yLabel: 'L', ariaLabel: 'The loss L(w): a bowl.', onResize: (pl) => { drawStatic(pl); draw(); } });
  drawStatic(p);

  function step() {
    if (path.length > MAX_STEPS) { status.textContent = `Stopped after ${MAX_STEPS} steps. Press Reset.`; return; }
    const w = cur(), g = m.at(w).grad;
    path.push(w - lr * g);
    draw();
    const q = m.at(cur());
    status.textContent = `Step ${path.length - 1}: w = ${fmt(q.w)}, L = ${fmt(q.L)}.`;
  }

  function draw() {
    const w = cur(), q = m.at(w);
    startOut.textContent = fmt(start);
    lrBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(PRESET_LRS[i] === lr)));
    stepBtn.setAttribute('aria-disabled', String(path.length > MAX_STEPS));
    p.clear('live');
    if (path.length > 1) {
      const pts = path.map((v) => [v, m.lossAt(v)]);
      p.polyline(pts, { class: 'bowl-path', layer: 'live' });
      pts.slice(0, -1).forEach(([a, b]) => p.dot(a, b, { class: 'bowl-pt', r: 4, layer: 'live' }));
    }
    p.tangent(w, q.L, q.grad, { half: 1.4, layer: 'live' });
    p.dot(w, q.L, { r: 7, layer: 'live' });
    const off = w < X[0] || w > X[1] || q.L > Y[1];
    p.root.setAttribute('aria-label', `The loss L(w), a bowl with its bottom at w = ${fmt(m.wMin)}. After ${path.length - 1} steps w = ${fmt(w)}, L = ${fmt(q.L)}, gradient ${fmt(q.grad)}.`);
    readout.innerHTML = `<span><b>w</b> ${fmt(w)}</span><span><b>L</b> ${fmt(q.L)}</span><span><b>gradient</b> ${fmt(q.grad)}</span>${off ? '<span class="bowl-off">off the chart</span>' : ''}`
      + `<span class="bowl-next">${math(updateTex(m, w, lr), false)}</span>`;
  }
  draw();
  return { step, draw };
}

const CSS = `
.bowl-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:.2rem 0 .4rem}
.bowl-field{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.bowl-field label,.bowl-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.bowl-field input[type=range]{width:9rem;accent-color:var(--fwd)}
.bowl-val{font-family:ui-monospace,monospace;font-size:.9rem;min-width:2.6em}
.bowl-read{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;align-items:baseline;font-family:ui-monospace,monospace;font-size:.92rem;margin:.3rem 0 0}
.bowl-read b{color:var(--muted);font-weight:600;margin-right:.3rem}
.bowl-next{font-family:inherit;overflow-x:auto;max-width:100%}
.bowl-off{color:var(--warn)}
.bowl-path{stroke:var(--fg);stroke-width:1.4;fill:none;opacity:.7}
.bowl-pt{fill:var(--panel2);stroke:var(--fg);stroke-width:1.8}
.btn[aria-disabled=true]{opacity:.45;cursor:default}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('bowl-styles')) return;
  document.head.append(el('style', { id: 'bowl-styles' }, CSS));
}
