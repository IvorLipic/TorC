// widgets/nonlinear-widgets.js — the two widgets of the lesson nonlinearity. Both show only what TorC recorded
// (data/nonlinearity.json, tools/trace/trace_nonlinear.cpp):
//
//   mountActivationPlot(host, D)   relu or sigmoid and its derivative on [-4, 4]; a slider picks one of the 81 recorded inputs
//   mountCurveFit(host, D)         two models fitted to t = x^2, with and without relu, at the recorded training steps
//
// The module touches the DOM only in mount*.
import { el } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';
import { createPlot, srTable, ticks } from './plot.js';

const CSS = `
.nl{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.nl-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .5rem}
.nl-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.nl-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.nl .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.nl-field{display:flex;gap:.5rem;align-items:center}
.nl-field input[type=range]{width:12rem;accent-color:var(--fwd)}
.nl-read{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font:.92rem ui-monospace,monospace;margin:.4rem 0 0}
.nl-read b{color:var(--muted);font-weight:600;margin-right:.3rem}
.nl-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.2rem 0 0}
.nl-deriv{stroke:var(--warn);stroke-width:2.2;stroke-dasharray:6 4;fill:none}
.nl-data{fill:var(--fg);stroke:var(--bg);stroke-width:1.5}
.nl-dot2{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('nl-styles')) return;
  document.head.append(el('style', { id: 'nl-styles' }, CSS));
}
// Split a sampled curve where it jumps (the derivative of ReLU goes from 0 to 1 between two samples), so that no line is drawn
// across the jump. Exported for tests.
export function pieces(xs, ys, jump = 0.5) {
  const out = [[]];
  xs.forEach((x, i) => {
    if (i > 0 && Math.abs(ys[i] - ys[i - 1]) > jump) out.push([]);
    out.at(-1).push([x, ys[i]]);
  });
  return out;
}

function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'nl-presets', role: 'group', 'aria-label': label }, el('span', { class: 'nl-lbl', 'aria-hidden': 'true' }, label), ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}

// ======================================================================================================================
export function mountActivationPlot(host, D) {
  injectStyles();
  const N = D.files.nonlinearity, z = N.z;
  let name = 'relu', k = 55;                                     // z = 1.5
  const fn = presets('Function', [['relu', 'ReLU'], ['sigmoid', 'Sigmoid']], (v) => { name = v; draw(); });
  const input = el('input', { type: 'range', id: 'nl-z', min: 0, max: z.length - 1, step: 1, value: k });
  input.addEventListener('input', () => { k = Number(input.value); mark(); });
  const box = el('div'), read = el('div', { class: 'nl-read', 'aria-live': 'polite' });
  const key = el('div', { class: 'nl-key', 'aria-hidden': 'true' }, el('span', {}, 'solid line: f(z)'), el('span', {}, 'dashed line: the derivative f′(z)'));
  host.replaceChildren(el('div', { class: 'nl' },
    el('div', { class: 'nl-ctl' }, fn.root, el('div', { class: 'nl-field' }, el('label', { for: 'nl-z', class: 'nl-lbl' }, 'z'), input)), box, key, read));
  let plot = null, act = null;

  function draw() {
    fn.set(name);
    act = N.activations.find((a) => a.name === name);
    const y = name === 'relu' ? [-0.5, 4.2] : [-0.1, 1.1];
    plot?.destroy();
    box.replaceChildren();
    const paint = (pl) => {
      pl.axes({ xTicks: ticks([-4, 4], 8), yTicks: name === 'relu' ? [0, 1, 2, 3, 4] : [0, 0.25, 0.5, 0.75, 1] });
      pl.curve(z.map((v, i) => [v, act.value[i]]), { class: 'plot-curve', layer: 'curves' });
      for (const part of pieces(z, act.grad)) pl.curve(part, { class: 'nl-deriv', layer: 'curves' });
      pl.layer('live');
      mark();
    };
    plot = createPlot(box, { width: 640, height: 280, x: [-4, 4], y, xLabel: 'z', yLabel: '', ariaLabel: `${name} and its derivative`, onResize: paint });
    paint(plot);
  }
  function mark() {
    plot.clear('live');
    plot.dot(z[k], act.value[k], { r: 6, layer: 'live' });
    plot.dot(z[k], act.grad[k], { class: 'nl-dot2', r: 5, layer: 'live' });
    const label = name === 'relu' ? 'ReLU' : 'sigmoid';
    input.setAttribute('aria-valuetext', `z = ${fmt(z[k])}`);
    plot.root.setAttribute('aria-label', `${label} and its derivative for z from -4 to 4. At z = ${fmt(z[k])} the value is ${fmt(act.value[k])} and the derivative is ${fmt(act.grad[k])}.`);
    read.innerHTML = `<span><b>z</b> ${fmt(z[k])}</span><span><b>f(z)</b> ${fmt(act.value[k])}</span><span><b>f′(z)</b> ${fmt(act.grad[k])}</span>`;
  }
  draw();
  return { draw };
}

// ======================================================================================================================
export function mountCurveFit(host, D) {
  injectStyles();
  const N = D.files.nonlinearity, steps = N.models[0].checkpoints.map((c) => c.step);
  let model = 'linear', at = steps.length - 1;
  const mp = presets('Model', [['linear', 'without ReLU'], ['relu', 'with ReLU']], (v) => { model = v; draw(); });
  const sp = presets('Training steps', steps.map((s, i) => [i, String(s)]), (i) => { at = i; draw(); });
  const box = el('div'), read = el('div', { class: 'nl-read', 'aria-live': 'polite' });
  const key = el('div', { class: 'nl-key', 'aria-hidden': 'true' }, el('span', {}, 'dots: the data, t = x²'), el('span', {}, 'line: the model'));
  host.replaceChildren(el('div', { class: 'nl' }, el('div', { class: 'nl-ctl' }, mp.root, sp.root), box, key, read));
  const table = srTable(host, { columns: ['x', 'model'], rows: [], caption: 'Prediction of the model on the grid' });

  let p = null;
  const paint = (pl) => {
    const c = N.models.find((m) => m.name === model).checkpoints[at];
    pl.axes({ xTicks: ticks([-2, 2], 4), yTicks: [0, 1, 2, 3, 4] });
    pl.curve(N.data.grid.map((x, i) => [x, c.prediction[i]]), { class: 'plot-curve', layer: 'model' });
    N.data.x.forEach((x, i) => pl.dot(x, N.data.t[i], { class: 'nl-data', r: 3.5, layer: 'data' }));
  };

  function draw() {
    mp.set(model); sp.set(at);
    const c = N.models.find((m) => m.name === model).checkpoints[at];
    p?.destroy();
    box.replaceChildren();
    p = createPlot(box, { width: 640, height: 300, x: [-2.2, 2.2], y: [-1.2, 4.6], xLabel: 'x', yLabel: 'value', ariaLabel: '', onResize: paint });
    paint(p);
    const label = model === 'relu' ? 'with ReLU' : 'without ReLU';
    p.root.setAttribute('aria-label', `The model ${label} after ${c.step} training steps, with the ${N.setup.points} data points of t = x squared. The loss is ${fmt(c.loss)}.`);
    read.innerHTML = `<span><b>model</b> ${label}</span><span><b>step</b> ${c.step}</span><span><b>loss</b> ${fmt(c.loss)}</span>`;
    table.update(N.data.grid.map((x, i) => [x, c.prediction[i]]));
  }
  draw();
  return { draw };
}
