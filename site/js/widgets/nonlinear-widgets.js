// widgets/nonlinear-widgets.js — the two widgets of the lesson nonlinearity. Both show only what TorC recorded
// (data/nonlinearity.json, tools/trace/trace_nonlinear.cpp):
//
//   mountActivationPlot(host, D)   relu or sigmoid and its gradient on [-4, 4]; a slider picks one of the 81 recorded inputs
//   mountCurveFit(host, D)         two models fitted to t = x^2, with and without relu, at the recorded training steps; the readout
//                                  gives the typical miss (√ of the recorded loss), not the bare loss; it opens on the model with
//                                  no activation at step 0, so it shows no end result before its task
// Curves are labelled on the plot; readouts show 2-3 significant figures.
//
// The module touches the DOM only in mount*.
import { el } from '../core/dom.js';
const sig3 = (v) => { if (Math.abs(v) < 5e-4) return '0'; const s = Number(v.toPrecision(3)).toString(); return s.replace('-', '−'); };
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
.nl-read.nl-sent{font:inherit;font-size:.95rem}
.nl-read.nl-sent b{color:var(--fg);margin:0}
.nl-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.2rem 0 0}
.nl-deriv{stroke:var(--warn);stroke-width:2.2;stroke-dasharray:6 4;fill:none}
.nl-data{fill:var(--fg);stroke:var(--bg);stroke-width:1.5}
.nl-dot2{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.nl-tf{fill:var(--fwd);font-size:12px;font-weight:700}
.nl-tg{fill:var(--warn);font-size:12px;font-weight:700}
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
  host.replaceChildren(el('div', { class: 'nl' },
    el('div', { class: 'nl-ctl' }, fn.root, el('div', { class: 'nl-field' }, el('label', { for: 'nl-z', class: 'nl-lbl' }, 'z'), input)), box, read));
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
      const label = name === 'relu' ? 'ReLU' : 'sigmoid';
      pl.text(-3.9, act.value[1], label, { class: 'nl-tf', dy: name === 'relu' ? -8 : -10, layer: 'curves' });
      pl.text(name === 'relu' ? 1.2 : 1.4, name === 'relu' ? 1 : act.grad[54], 'gradient', { class: 'nl-tg', dy: name === 'relu' ? -8 : -10, layer: 'curves' });
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
    input.setAttribute('aria-valuetext', `z = ${sig3(z[k])}`);
    plot.root.setAttribute('aria-label', `${label} and its gradient for z from −4 to 4. At z = ${sig3(z[k])} the value is ${sig3(act.value[k])} and the gradient is ${sig3(act.grad[k])}.`);
    read.innerHTML = `<span><b>z</b> ${sig3(z[k])}</span><span><b>${label}</b> ${sig3(act.value[k])}</span><span><b>gradient</b> ${sig3(act.grad[k])}</span>`;
  }
  draw();
  return { draw };
}

// ======================================================================================================================
export function mountCurveFit(host, D) {
  injectStyles();
  const N = D.files.nonlinearity, steps = N.models[0].checkpoints.map((c) => c.step);
  let model = 'linear', at = 0;                                // opens on the start of the model with no activation
  const mp = presets('Model', [['linear', 'no activation'], ['relu', 'with ReLU']], (v) => { model = v; draw(); });
  const sp = presets('Training steps', steps.map((s, i) => [i, String(s)]), (i) => { at = i; draw(); });
  const box = el('div'), read = el('div', { class: 'nl-read nl-sent', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'nl' }, el('div', { class: 'nl-ctl' }, mp.root, sp.root), box, read));
  const table = srTable(host, { columns: ['x', 'model'], rows: [], caption: 'Prediction of the model on the grid' });

  let p = null;
  const paint = (pl) => {
    const c = N.models.find((m) => m.name === model).checkpoints[at];
    pl.axes({ xTicks: ticks([-2, 2], 4), yTicks: [0, 1, 2, 3, 4] });
    pl.curve(N.data.grid.map((x, i) => [x, c.prediction[i]]), { class: 'plot-curve', layer: 'model' });
    N.data.x.forEach((x, i) => pl.dot(x, N.data.t[i], { class: 'nl-data', r: 3.5, layer: 'data' }));
    pl.text(N.data.grid[32], c.prediction[32], 'model', { class: 'nl-tf', dx: 10, dy: 16, layer: 'model' });
    pl.text(-2, 4, 'data', { class: 'plot-text', dx: 8, dy: 4, layer: 'data' });
  };

  function draw() {
    mp.set(model); sp.set(at);
    const c = N.models.find((m) => m.name === model).checkpoints[at];
    p?.destroy();
    box.replaceChildren();
    p = createPlot(box, { width: 640, height: 300, x: [-2.2, 2.2], y: [-1.2, 4.6], xLabel: 'x', yLabel: 'value', ariaLabel: '', onResize: paint });
    paint(p);
    const label = model === 'relu' ? 'with ReLU' : 'with no activation function';
    const miss = sig3(Math.sqrt(c.loss));
    p.root.setAttribute('aria-label', `The model ${label} after ${c.step} training steps, with the ${N.setup.points} data points of t = x squared. Its typical miss is about ${miss}.`);
    read.innerHTML = `<span>After <b>${c.step}</b> steps, the typical miss of the model ${label} is about <b>${miss}</b>.</span>`;
    table.update(N.data.grid.map((x, i) => [x, c.prediction[i]]));
  }
  draw();
  return { draw };
}
