// widgets/generalization-widgets.js — the widget of the lesson generalization. It shows only what TorC recorded
// (data/generalization.json, tools/trace/trace_generalization.cpp):
//
//   mountGeneralization(host, D)   three runs that differ only in the weight decay. Top: the model's curve with the training and
//                                  test samples. Bottom: the training and test loss at each recorded step (log scales).
//                                  Presets pick the weight decay, a slider picks the step.
//
// The module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';
import { createPlot, srTable } from './plot.js';

// x position of a step on the log axis (step 0 sits at the left edge). Exported for tests.
export const stepX = (step) => Math.log10(step + 1);
export const DECAYS = [[0, 'no weight decay'], [0.1, 'weight decay 0.1'], [1, 'weight decay 1']];

const CSS = `
.gw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.gw-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .5rem}
.gw-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.gw-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.gw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.gw-field{display:flex;gap:.5rem;align-items:center}
.gw-field input[type=range]{width:12rem;accent-color:var(--fwd)}
.gw-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.2rem 0 .6rem}
.gw-read{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font:.9rem ui-monospace,monospace;margin:.4rem 0 0}
.gw-read b{color:var(--muted);font-weight:600;margin-right:.3rem}
.gw-train{fill:var(--fg);stroke:var(--bg);stroke-width:1.5}
.gw-test{fill:none;stroke:var(--muted);stroke-width:1.2}
.gw-model{stroke:var(--fwd);stroke-width:2.5;fill:none}
.gw-ltrain{stroke:var(--warn);stroke-width:2.2;stroke-dasharray:7 4;fill:none}
.gw-ltest{stroke:var(--fwd);stroke-width:2.5;fill:none}
.gw-floor{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:3 3}
.gw-dtrain{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.gw-dtest{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('gw-styles')) return;
  document.head.append(el('style', { id: 'gw-styles' }, CSS));
}
function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'gw-presets', role: 'group', 'aria-label': label }, el('span', { class: 'gw-lbl', 'aria-hidden': 'true' }, label), ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}
const decade = (d) => String(Number((10 ** d).toPrecision(1)));

// ======================================================================================================================
export function mountGeneralization(host, D) {
  injectStyles();
  const G = D.files.generalization, data = G.data;
  const steps = G.runs[0].checkpoints.map((c) => c.step);
  let wd = 0, k = steps.indexOf(G.setup.steps);
  const run = () => G.runs.find((r) => r.weight_decay === wd);
  const pr = presets('Weight decay', DECAYS.map(([v]) => [v, String(v)]), (v) => { wd = v; draw(); });
  const input = el('input', { type: 'range', id: 'gw-step', min: 0, max: steps.length - 1, step: 1, value: k });
  input.addEventListener('input', () => { k = Number(input.value); draw(); });
  const fitBox = el('div'), lossBox = el('div');
  const fitKey = el('div', { class: 'gw-key', 'aria-hidden': 'true' }, el('span', {}, 'filled dots: training samples'), el('span', {}, 'open dots: test samples'), el('span', {}, 'line: the model'));
  const lossKey = el('div', { class: 'gw-key', 'aria-hidden': 'true' }, el('span', {}, 'dashed line: training loss'), el('span', {}, 'solid line: test loss'), el('span', {}, 'thin line: the noise in the targets'));
  const read = el('div', { class: 'gw-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'gw' },
    el('div', { class: 'gw-ctl' }, pr.root, el('div', { class: 'gw-field' }, el('label', { for: 'gw-step', class: 'gw-lbl' }, 'after step'), input)),
    fitBox, fitKey, lossBox, lossKey, read));
  const table = srTable(host, { columns: ['step', 'training loss', 'test loss'], rows: [], caption: 'Losses of the selected run at each recorded step' });

  const paintFit = (pl) => {
    pl.axes({ xTicks: [-1, -0.5, 0, 0.5, 1], yTicks: [-1, 0, 1] });
    data.test_x.forEach((x, i) => pl.dot(x, data.test_t[i], { class: 'gw-test', r: 2.6, layer: 'data' }));
    data.train_x.forEach((x, i) => pl.dot(x, data.train_t[i], { class: 'gw-train', r: 4.5, layer: 'data' }));
    pl.layer('model');
    drawFit(pl);
  };
  const drawFit = (pl) => {
    pl.clear('model');
    pl.curve(data.grid.map((x, i) => [x, run().checkpoints[k].curve[i]]), { class: 'gw-model', layer: 'model' });
  };
  const xMax = stepX(G.setup.steps);
  const paintLoss = (pl) => {
    pl.axes({ xTicks: [0, 10, 100, 1000, 10000].map(stepX), yTicks: [-5, -4, -3, -2, -1, 0], xFmt: (v) => String(Math.round(10 ** v - 1)), yFmt: decade });
    const floor = Math.log10(G.derived.noise_variance);
    pl.layer('curves');
    pl.layer('live');
    drawLoss(pl, floor);
  };
  const drawLoss = (pl, floor = Math.log10(G.derived.noise_variance)) => {
    pl.clear('curves'); pl.clear('live');
    pl.layer('curves').append(svg('line', { class: 'gw-floor', x1: pl.sx(0), x2: pl.sx(xMax), y1: pl.sy(floor), y2: pl.sy(floor) }));
    const cs = run().checkpoints;
    pl.curve(cs.map((c) => [stepX(c.step), Math.log10(c.train_loss)]), { class: 'gw-ltrain', layer: 'curves' });
    pl.curve(cs.map((c) => [stepX(c.step), Math.log10(c.test_loss)]), { class: 'gw-ltest', layer: 'curves' });
    const c = cs[k];
    pl.dot(stepX(c.step), Math.log10(c.train_loss), { class: 'gw-dtrain', r: 5, layer: 'live' });
    pl.dot(stepX(c.step), Math.log10(c.test_loss), { class: 'gw-dtest', r: 5, layer: 'live' });
  };
  const fit = createPlot(fitBox, { width: 640, height: 230, x: [-1.05, 1.05], y: [-1.7, 1.7], xLabel: 'x', yLabel: 't', ariaLabel: '', onResize: paintFit });
  const loss = createPlot(lossBox, { width: 640, height: 230, x: [0, xMax], y: [-5, 0], xLabel: 'step', yLabel: 'loss (log scale)', ariaLabel: '', onResize: (pl) => paintLoss(pl) });

  function draw() {
    pr.set(wd);
    const c = run().checkpoints[k], name = DECAYS.find(([v]) => v === wd)[1];
    input.setAttribute('aria-valuetext', `after step ${c.step}`);
    drawFit(fit);
    drawLoss(loss);
    fit.root.setAttribute('aria-label', `The model with ${name} after step ${c.step}, with ${data.train_x.length} training samples and ${data.test_x.length} test samples of a sine curve with noise.`);
    loss.root.setAttribute('aria-label', `Training and test loss with ${name}, on log scales. After step ${c.step}: training loss ${fmt(c.train_loss)}, test loss ${fmt(c.test_loss)}.`);
    read.innerHTML = `<span><b>step</b> ${c.step}</span><span><b>training loss</b> ${fmt(c.train_loss)}</span><span><b>test loss</b> ${fmt(c.test_loss)}</span><span><b>sum of squared weights</b> ${fmt(c.weight_sum_sq)}</span>`;
    table.update(run().checkpoints.map((q) => [q.step, q.train_loss, q.test_loss]));
  }
  paintFit(fit);
  paintLoss(loss);
  draw();
  return { draw };
}
