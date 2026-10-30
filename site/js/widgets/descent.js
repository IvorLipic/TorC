// widgets/descent.js — the widget of the lesson sgd-step: the loss map over (w, b) (contour2d) with the trail of SGD steps, the
// gradient drawn as an arrow that points uphill (scaled by lr, so it is the mirror image of the step), an optional fog that hides the
// map, learning-rate presets, Step and Reset, and a short readout. Plain gradient descent only (momentum is taught in `optimizers`).
// Runs are recomputed by optim.js (SGD's formulas on the engine's model); site/test/sgd-step.test.mjs checks them against TorC's
// recorded runs in data/optim.json. Importing this module does not touch the DOM.
//
//   mountDescent(host, D, { lrs: [0.05, 0.15, 0.2], fog: false })   (the trail starts at optim.json off_axis: w = 0, b = 3)
import { el } from '../core/dom.js';
import { mountContour2d } from './contour2d.js';
import { makeModel, simulate } from '../optim.js';
import { injectStyle, SHARED_CSS, presets } from './sgd-ui.js';

export const STEPS = 10;
export const LR_PRESETS = [0.05, 0.15, 0.2];          // curves, zig-zags across the valley, diverges (TorC's off_axis runs)
// The view the map starts with: it holds the start (0, 3), the lowest point (2, 1) and the lr 0.05 and 0.15 trails with their arrows.
export const BASE = { w: [-2.5, 4], b: [-0.5, 4.5] };

// The trail the widget shows: SGD (no momentum) from `start`, ten steps, recomputed by optim.js; rows[k] is the state after k steps.
export function trailFor(model, lr, start) {
  const run = simulate(model, { lr, steps: STEPS, start });
  return [...run.steps, run.final];
}

// The two arrows at a point of the trail: the gradient arrow ends at point + lr·gradient (uphill), the step ends at
// point − lr·gradient (SGD without momentum), so the two are mirror images through the point.
export function arrowsAt(p, lr) {
  return { uphill: { w: p.w + lr * p.gw, b: p.b + lr * p.gb }, step: { w: p.w - lr * p.gw, b: p.b - lr * p.gb } };
}

// A readout number: 3 significant figures, a comma from 1,000 up, the minus sign U+2212.
export function r3(x) {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? 'not a number' : (x > 0 ? 'infinity' : '−infinity');
  const n = Number(x.toPrecision(3));
  const s = Math.abs(n) >= 1000 ? n.toLocaleString('en-US', { maximumFractionDigits: 0 }) : String(n);
  return s.replace('-', '−');
}

const CSS = `
.sgd-read .bad b{color:var(--bwd)}
.sgd-w .fog-btn[aria-pressed=true]{border-color:#ffb454;color:#ffb454;background:rgba(255,180,84,.1)}
.sgd-w .lr-one{color:var(--muted);font-size:.9rem}
`;

export function mountDescent(host, D, { lrs = LR_PRESETS, fog: withFog = false } = {}) {
  injectStyle('sgd-ui-css', SHARED_CSS);
  injectStyle('sgd-descent-css', CSS);
  const model = makeModel(D.scenes.sgd);
  const off = D.files.optim.off_axis;
  const start = [off.w0, off.b0];
  const state = { lr: lrs[0], k: 0, fog: false };
  let rows = [];

  const restart = (why) => { state.k = 0; render(why); };
  const lrP = lrs.length > 1 ? presets('learning rate lr', lrs, (v) => { state.lr = v; restart(`The learning rate is now ${v}. The trail starts again.`); }) : null;
  const stepBtn = el('button', { type: 'button', class: 'btn primary', onclick: () => { if (state.k < STEPS) { state.k++; render(''); } } }, 'Step');
  const resetBtn = el('button', { type: 'button', class: 'btn', onclick: () => restart('The trail starts again at the start point.') }, 'Reset');
  const fogBtn = withFog ? el('button', { type: 'button', class: 'btn fog-btn', 'aria-pressed': 'false', onclick: () => { state.fog = !state.fog; render(state.fog ? 'The fog hides the map.' : 'The fog is lifted, and the map shows again.'); } }, 'Fog') : null;
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const map = el('div'), read = el('div', { class: 'sgd-read' });
  const root = el('div', { class: 'sgd-w' },
    el('div', { class: 'sgd-ctl' }, lrP ? lrP.root : el('span', { class: 'lr-one' }, `learning rate lr = ${lrs[0]}`),
      el('div', { class: 'row' }, stepBtn, resetBtn, ...(fogBtn ? [fogBtn] : [])), status),
    map, read);
  host.append(root);
  const contour = mountContour2d(map, { model, base: BASE, optimum: D.files.optim.data.optimum });

  // why: undefined = the first draw (silent), '' = a step (announce the new point), text = a control change (announce it too)
  function render(why) {
    rows = trailFor(model, state.lr, start);
    const cur = rows[state.k], prev = state.k > 0 ? rows[state.k - 1] : null;
    lrP?.set(state.lr);
    stepBtn.setAttribute('aria-disabled', String(state.k >= STEPS));
    fogBtn?.setAttribute('aria-pressed', String(state.fog));
    const { uphill } = arrowsAt(cur, state.lr);
    contour.draw({ trail: rows.slice(0, state.k + 1), next: state.k < STEPS ? rows[state.k + 1] : null, uphill, fog: state.fog, caption: `The learning rate is ${state.lr}.` });
    const rose = prev && cur.loss > prev.loss;
    read.innerHTML = `<span><b>step</b> ${state.k} of ${STEPS}</span><span><b>w</b> ${r3(cur.w)}</span><span><b>b</b> ${r3(cur.b)}</span>`
      + `<span><b>gradient</b> (${r3(cur.gw)}, ${r3(cur.gb)})</span>`
      + `<span${rose ? ' class="bad"' : ''}><b>loss</b> ${r3(cur.loss)}, ${rose ? 'higher than before' : 'lower is better'}</span>`
      + `<span><b>typical miss</b> ${r3(Math.sqrt(cur.loss))} euros</span>`;
    const where = `After ${state.k} steps, w is ${r3(cur.w)} and b is ${r3(cur.b)}. The loss is ${r3(cur.loss)}${rose ? ', higher than before' : ''}.`;
    if (why !== undefined) status.textContent = why ? `${why} ${where}` : where;
  }

  render();
  return { destroy() { contour.destroy(); root.remove(); }, state };
}
