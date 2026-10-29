// widgets/descent.js — the main widget of the lesson sgd-step: the loss landscape over (w, b) (contour2d) with the trail of
// optimizer.step() calls, learning-rate and momentum presets, Step/Reset, and the loss after each step on a log scale.
// Runs are recomputed by optim.js (SGD's formulas on the engine's model); site/test/sgd-step.test.mjs checks them against TorC's
// recorded runs. Importing this module does not touch the DOM.
import { el } from '../core/dom.js';
import { mountContour2d } from './contour2d.js';
import { makeModel, simulate } from '../optim.js';
import { injectStyle, SHARED_CSS, presets, logChart, fmt, onResize } from './sgd-ui.js';

export const STEPS = 10;
export const LR_PRESETS = [0.05, 0.1, 0.15, 0.2];     // steady, overshoots, zig-zags, diverges (all recorded by TorC with μ = 0)
export const MU_PRESETS = [0, 0.5, 0.9];              // recorded by TorC at lr 0.05

export function mountDescent(host, D) {
  injectStyle('sgd-ui-css', SHARED_CSS);
  const model = makeModel(D.scenes.sgd);
  const state = { lr: LR_PRESETS[0], mu: MU_PRESETS[0], k: 0 };
  let rows = [];

  const restart = () => { state.k = 0; render(); };
  const lrP = presets('lr', LR_PRESETS, (v) => { state.lr = v; restart(); });
  const muP = presets('momentum μ', MU_PRESETS, (v) => { state.mu = v; restart(); });
  const stepBtn = el('button', { type: 'button', class: 'btn primary', onclick: () => { if (state.k < STEPS) { state.k++; render(); } } }, 'Step');
  const resetBtn = el('button', { type: 'button', class: 'btn', onclick: restart }, 'Reset');
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const left = el('div'), chart = el('div', { class: 'chart' }), read = el('div', { class: 'sgd-read' });
  const root = el('div', { class: 'sgd-w' },
    el('div', { class: 'sgd-ctl' }, lrP.root, muP.root, el('div', { class: 'row' }, stepBtn, resetBtn), status),
    el('div', { class: 'sgd-grid' }, left, el('div', {}, el('h3', { class: 'sub' }, 'Loss after each step (log scale)'), chart)),
    read);
  host.append(root);
  const contour = mountContour2d(left, { model, base: { w: [-1, 5], b: [-1.5, 3] }, optimum: D.files.optim.data.optimum });

  function drawChart() {
    const shown = rows.slice(0, state.k + 1);
    logChart(chart, {
      series: [{ values: shown.map((r) => r.loss), cls: 'lc-main', dots: true }], range: rows.map((r) => r.loss), n: STEPS, xLabel: 'steps',
      ariaLabel: `Loss after each step, log scale: ${fmt(shown[0].loss)} at the start, ${fmt(shown.at(-1).loss)} after ${state.k}.`,
    });
  }

  function render() {
    const run = simulate(model, { lr: state.lr, momentum: state.mu, steps: STEPS });
    rows = [...run.steps, run.final];
    const cur = rows[state.k];
    lrP.set(state.lr); muP.set(state.mu);
    stepBtn.setAttribute('aria-disabled', String(state.k >= STEPS));
    contour.draw({ trail: rows.slice(0, state.k + 1), next: state.k < STEPS ? rows[state.k + 1] : null, caption: `Learning rate ${state.lr}, momentum ${state.mu}.` });
    drawChart();
    const up = state.k > 0 && cur.loss > rows[0].loss;
    read.innerHTML = `<span><b>step</b> ${state.k}/${STEPS}</span><span><b>W</b> ${fmt(cur.w)}</span><span><b>b</b> ${fmt(cur.b)}</span><span${up ? ' class="bad"' : ''}><b>loss</b> ${fmt(cur.loss)}</span>`;
    status.textContent = state.k ? `Step ${state.k}: W ${fmt(cur.w)}, b ${fmt(cur.b)}, loss ${fmt(cur.loss)}.` : '';
  }

  const ac = new AbortController();
  onResize(drawChart, ac.signal);
  render();
  return { destroy() { ac.abort(); root.remove(); }, state };
}
