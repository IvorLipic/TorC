// widgets/loop-timeline.js — the zero_grad trap: a ten-iteration training loop with one checkbox for the zero_grad() line. Two small
// charts: the W.grad that step() reads in each iteration (bars) and the loss (log scale), the chosen loop solid, the other dashed.
// Both loops are recomputed by optim.js trainingLoop; site/test/sgd-step.test.mjs checks them against TorC's recorded loops.
import { el, svg } from '../core/dom.js';
import { createPlot, ticks } from './plot.js';
import { makeModel, trainingLoop } from '../optim.js';
import { injectStyle, SHARED_CSS, logChart, fmt, fitWidth, onResize } from './sgd-ui.js';

export const ITERATIONS = 10;

const CSS = `
.sgd-loop label.chk{display:inline-flex;gap:.5rem;align-items:center;cursor:pointer}
.sgd-loop input[type=checkbox]{width:1.15rem;height:1.15rem;accent-color:var(--fwd)}
.sgd-loop .bar-on{fill:var(--bwd)}
.sgd-loop .bar-off{fill:none;stroke:var(--muted);stroke-width:1.4;stroke-dasharray:3 2}
.sgd-loop .lc-zero{stroke:var(--fg);stroke-width:1.2}
`;

export function mountLoopTimeline(host, D) {
  injectStyle('sgd-ui-css', SHARED_CSS); injectStyle('sgd-loop-css', CSS);
  const model = makeModel(D.scenes.sgd);
  const lr = D.files.optim.no_zero_grad.training_loop.lr, N = ITERATIONS;
  const withZ = trainingLoop(model, { lr, iterations: N, zeroGrad: true });
  const noZ = trainingLoop(model, { lr, iterations: N, zeroGrad: false });
  let zero = true;

  const chk = el('input', { type: 'checkbox', id: 'sgd-zg', checked: true });
  const barsHost = el('div', { class: 'chart' }), lossHost = el('div', { class: 'chart' }), read = el('div', { class: 'sgd-read', 'aria-live': 'polite' });
  const root = el('div', { class: 'sgd-w sgd-loop' },
    el('div', { class: 'sgd-ctl' }, el('label', { class: 'chk', for: 'sgd-zg' }, chk, el('span', {}, 'call ', el('code', {}, 'optimizer.zero_grad()'), ' every iteration'))),
    el('div', { class: 'sgd-grid' },
      el('div', {}, el('h3', { class: 'sub' }, 'W.grad that step() reads'), barsHost),
      el('div', {}, el('h3', { class: 'sub' }, 'Loss (log scale). The dashed line is the other loop.'), lossHost)),
    read);
  host.append(root);

  function bars() {
    barsHost.replaceChildren();
    const W = fitWidth(barsHost, 280, 460, 420), H = Math.round(W * 0.55), m = { l: 52, r: 10, t: 12, b: 38 };
    const R = Math.ceil(Math.max(...[...withZ, ...noZ].map((r) => Math.abs(r.gw))) / 10) * 10;
    const sel = zero ? withZ : noZ, other = zero ? noZ : withZ;
    const p = createPlot(barsHost, { width: W, height: H, x: [0.4, N + 0.6], y: [-R, R], xLabel: 'iteration', yLabel: 'W.grad', margin: m, responsive: false,
      ariaLabel: `Gradient of W read by step() in each iteration, ${zero ? 'with' : 'without'} zero_grad(): ${sel.map((r) => fmt(r.gw)).join(', ')}.` });
    p.axes({ xTicks: Array.from({ length: N }, (_, i) => i + 1), yTicks: ticks([-R, R], 4) });
    const layer = p.layer('bars'), bw = 0.62;
    const rect = (i, v, cls) => { const y0 = p.sy(0), y1 = p.sy(v); return svg('rect', { class: cls, x: p.sx(i - bw / 2), y: Math.min(y0, y1), width: p.sx(i + bw / 2) - p.sx(i - bw / 2), height: Math.max(Math.abs(y1 - y0), 0.5) }); };
    for (let i = 1; i <= N; i++) layer.append(rect(i, sel[i - 1].gw, 'bar-on'), rect(i, other[i - 1].gw, 'bar-off'));
    layer.append(svg('line', { class: 'lc-zero', x1: m.l, x2: W - m.r, y1: p.sy(0), y2: p.sy(0) }));
  }

  function render() {
    bars();
    const sel = zero ? withZ : noZ, other = zero ? noZ : withZ;
    logChart(lossHost, {
      series: [{ values: other.map((r) => r.loss), cls: 'lc-ghost' }, { values: sel.map((r) => r.loss), cls: 'lc-main', dots: true }],
      range: [...withZ, ...noZ].map((r) => r.loss), x0: 1, n: N - 1, xLabel: 'iteration',
      ariaLabel: `Loss at each iteration ${zero ? 'with' : 'without'} zero_grad(): ${fmt(sel[0].loss)} first, ${fmt(sel.at(-1).loss)} at iteration ${N}.`,
    });
    read.innerHTML = `<span><b>loss at iteration ${N}</b> ${fmt(sel.at(-1).loss)}</span>`;
  }
  chk.addEventListener('change', () => { zero = chk.checked; render(); });
  const ac = new AbortController();
  onResize(render, ac.signal);
  render();
  return { destroy() { ac.abort(); root.remove(); } };
}
