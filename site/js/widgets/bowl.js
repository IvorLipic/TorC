// widgets/bowl.js — the bowl with fog mode of the lesson why-gradients (brief §9 "Fog mode", Appendix B, D20): the loss curve
// L(w) of the two-notebook model, learning-rate presets and a Step button (new w = w − lr × gradient, from w = 0). With the fog
// on, the curve and the path are hidden: the learner sees only the current point, a short piece of its tangent (the ground under
// the feet) and the readout (w, loss, gradient). "Lift the fog" shows the curve and the whole path. Values are recomputed by the
// engine through landscape-model.js (the same code as the graph lab), and site/test/why-gradients.test.mjs checks that model
// against TorC's run. Controls are never re-rendered (focus stays put); no autoplay; the module touches the DOM only in mountBowl.
import { el } from '../core/dom.js';
import { math } from '../core/katex.js';
import { updateTex, fmt3 } from './landscape-tex.js';
import { createPlot, ticks } from './plot.js';
import { makeModel } from './landscape-model.js';

export const PRESET_LRS = [0.05, 0.1, 0.25, 0.3];     // slow, faster, bounce, diverge (all four are runs TorC recorded)
export const MAX_STEPS = 40;
const X = [-1, 7], Y = [0, 72];

let seq = 0;

export function mountBowl(host, D) {
  injectStyles();
  const m = makeModel(D.scenes.why);
  const id = `bowl${++seq}`;
  let lr = PRESET_LRS[0], fog = false, path = [m.w0];
  const cur = () => path[path.length - 1];

  // ---- controls ---------------------------------------------------------------------------------------------------------
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const restart = () => { path = [m.w0]; status.textContent = ''; draw(); };
  const lrBtns = PRESET_LRS.map((v) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { lr = v; restart(); } }, String(v)));
  const fogBtn = el('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', onclick: () => {
    fog = !fog;
    status.textContent = fog ? 'The fog hides the loss curve.' : 'The fog is gone. You see the loss curve and your path.';
    draw();
  } }, 'Turn the fog on');
  const stepBtn = el('button', { type: 'button', class: 'btn primary', onclick: step }, 'Step');
  const resetBtn = el('button', { type: 'button', class: 'btn', onclick: () => restart() }, 'Reset');
  const controls = el('div', { class: 'bowl-ctl' },
    el('div', { class: 'bowl-field', role: 'group', 'aria-labelledby': `${id}-lr` }, el('span', { class: 'bowl-lbl', id: `${id}-lr` }, 'learning rate'), ...lrBtns),
    el('div', { class: 'bowl-field' }, stepBtn, resetBtn, fogBtn), status);

  // ---- plot and readout ----------------------------------------------------------------------------------------------------
  const plotBox = el('div', { class: 'bowl-plot' });
  const readout = el('div', { class: 'bowl-read' });
  const next = el('div', { class: 'bowl-next' });
  host.replaceChildren(controls, plotBox, readout, next);
  const curve = m.curve(X[0], X[1], 160).map((q) => [q.w, q.L]);
  const p = createPlot(plotBox, { width: 640, height: 300, x: X, y: Y, xLabel: 'w', yLabel: 'loss', ariaLabel: 'The loss of the two-notebook model against w.', onResize: () => draw() });

  function step() {
    if (path.length > MAX_STEPS) { status.textContent = `Stopped after ${MAX_STEPS} steps. Press Reset.`; return; }
    const w = cur(), g = m.at(w).grad;
    path.push(w - lr * g);
    draw();
    const q = m.at(cur());
    status.textContent = `Step ${path.length - 1}. w is ${fmt3(q.w)}, the loss is ${fmt3(q.L)}, and the gradient is ${fmt3(q.grad)}.`;
  }

  function draw() {
    const w = cur(), q = m.at(w);
    lrBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(PRESET_LRS[i] === lr)));
    fogBtn.setAttribute('aria-pressed', String(fog));
    fogBtn.textContent = fog ? 'Lift the fog' : 'Turn the fog on';
    stepBtn.setAttribute('aria-disabled', String(path.length > MAX_STEPS));
    p.clear('curve'); p.clear('live'); p.clear('labels');
    p.root.querySelectorAll('.plot-axes').forEach((g) => g.remove());
    p.axes({ xTicks: ticks(X, 8), yTicks: [0, 20, 40, 60] });
    p.root.classList.toggle('bowl-fog', fog);
    if (!fog) {
      p.curve(curve, { class: 'plot-curve', layer: 'curve' });
      p.text(5.3, m.lossAt(5.3), 'loss curve', { class: 'plot-text bowl-label', dx: 10, dy: 4, layer: 'labels' });
      p.line(m.wMin, Y[0], m.wMin, Y[1] * 0.92, { class: 'bowl-bottom', layer: 'curve' });
      p.text(m.wMin, Y[1] * 0.92, 'bottom', { class: 'plot-text bowl-label', anchor: 'middle', dy: -4, layer: 'labels' });
      if (path.length > 1) {
        const pts = path.map((v) => [v, m.lossAt(v)]);
        p.polyline(pts, { class: 'bowl-path', layer: 'live' });
        pts.slice(0, -1).forEach(([a, b]) => p.dot(a, b, { class: 'bowl-pt', r: 4, layer: 'live' }));
      }
    } else {
      p.text(X[0] + 0.2, Y[1] * 0.9, 'fog', { class: 'plot-text bowl-label', layer: 'labels' });
    }
    p.tangent(w, q.L, q.grad, { half: fog ? 0.45 : 1.4, layer: 'live' });
    p.dot(w, q.L, { r: 7, layer: 'live' });
    const off = w < X[0] || w > X[1] || q.L > Y[1];
    p.root.setAttribute('aria-label', fog
      ? `The loss curve is hidden by the fog. After ${path.length - 1} steps, w is ${fmt3(w)}, the loss is ${fmt3(q.L)}, and the gradient is ${fmt3(q.grad)}.`
      : `The loss of the two-notebook model against w, a bowl with its bottom at w = ${fmt3(m.wMin)}. After ${path.length - 1} steps, w is ${fmt3(w)}, the loss is ${fmt3(q.L)}, and the gradient is ${fmt3(q.grad)}.`);
    readout.innerHTML = `<span><b>steps</b> ${path.length - 1}</span><span><b>w</b> ${fmt3(w)}</span>`
      + `<span><b>loss</b> ${fmt3(q.L)} <small>lower is better</small></span><span><b>gradient</b> ${fmt3(q.grad)}</span>`
      + (off ? '<span class="bowl-off">off the plot</span>' : '');
    next.innerHTML = `<span class="bowl-lbl">next step</span> ${math(updateTex(m, w, lr), false)}`;
  }
  draw();
  return { step, draw, setLr: (v) => { lr = v; restart(); }, toggleFog: () => fogBtn.click() };
}

const CSS = `
.bowl-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:.2rem 0 .4rem}
.bowl-field{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.bowl-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.bowl-read{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;align-items:baseline;font-family:ui-monospace,monospace;font-size:.92rem;margin:.3rem 0 0}
.bowl-read b{color:var(--muted);font-weight:600;margin-right:.3rem}
.bowl-read small{color:var(--muted);font-family:system-ui,sans-serif}
.bowl-next{overflow-x:auto;max-width:100%;margin:.3rem 0 0}
.bowl-off{color:var(--warn)}
.bowl-path{stroke:var(--fg);stroke-width:1.4;fill:none;opacity:.7}
.bowl-pt{fill:var(--panel2);stroke:var(--fg);stroke-width:1.8}
.bowl-bottom{stroke:var(--muted);stroke-dasharray:4 4}
.bowl-label{fill:var(--muted);font-size:12px}
.plot.bowl-fog{background:repeating-linear-gradient(135deg,rgba(150,160,175,.10) 0 14px,rgba(150,160,175,.04) 14px 28px)}
.btn[aria-disabled=true]{opacity:.45;cursor:default}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('bowl-styles')) return;
  document.head.append(el('style', { id: 'bowl-styles' }, CSS));
}
