// widgets/loss-widgets.js — the two widgets of the lesson losses. Both show only what TorC recorded (data/losses.json,
// tools/trace/trace_losses.cpp):
//
//   mountLossExplorer(host, D)   one sample with three classes: a row of logits and a correct class, the softmax probabilities,
//                                the cross-entropy loss and its gradient with respect to the logits
//   mountGradientCompare(host, D) two-class logits [z, 0] with class 0 correct: the gradient of cross-entropy and of the
//                                squared error on the probabilities, for z from -6 to 6
//
// The module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';
import { createPlot, srTable, ticks } from './plot.js';

const CSS = `
.lw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.lw-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .6rem}
.lw-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.lw-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.lw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.lw-cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;max-width:34rem}
.lw-col{min-width:0;background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.5rem .6rem;font:.85rem ui-monospace,monospace}
.lw-col.target{border-color:var(--ok)}
.lw-h{margin:0 0 .3rem;font:600 .85rem system-ui,sans-serif;color:var(--fg)}
.lw-row{display:flex;flex-wrap:wrap;justify-content:space-between;gap:0 .5rem}
.lw-row b{color:var(--muted);font-weight:600}
.lw-bar{height:.5rem;border-radius:4px;background:var(--fwd);margin:.15rem 0 .35rem}
.lw-gbar{height:.5rem;border-radius:4px;margin:.15rem 0 0}
.lw-gbar.up{background:var(--ok)}
.lw-gbar.down{background:var(--bwd)}
.lw-read{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font:.92rem ui-monospace,monospace;margin:.6rem 0 0}
.lw-read b{color:var(--muted);font-weight:600;margin-right:.3rem}
.lw-field{display:flex;gap:.5rem;align-items:center}
.lw-field input[type=range]{width:12rem;accent-color:var(--fwd)}
.lw-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.2rem 0 0}
.lw-sq{stroke:var(--warn);stroke-width:2.4;stroke-dasharray:6 4;fill:none}
.lw-dot2{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.lw-zero{stroke:var(--muted);stroke-width:1}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('lw-styles')) return;
  document.head.append(el('style', { id: 'lw-styles' }, CSS));
}
function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'lw-presets', role: 'group', 'aria-label': label }, el('span', { class: 'lw-lbl', 'aria-hidden': 'true' }, label), ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}

// ======================================================================================================================
export function mountLossExplorer(host, D) {
  injectStyles();
  const L = D.files.losses;
  let row = 'mild', target = 0;
  const rp = presets('Logits', L.ce.map((r) => [r.id, r.logits.map(fmt).join(', ')]), (k) => { row = k; draw(); });
  const tp = presets('Correct class', [[0, 'class 0'], [1, 'class 1'], [2, 'class 2']], (k) => { target = k; draw(); });
  const cols = el('div', { class: 'lw-cols', role: 'group', 'aria-label': 'The three classes' });
  const read = el('div', { class: 'lw-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'lw' }, el('div', { class: 'lw-ctl' }, rp.root, tp.root), cols, read));

  function draw() {
    rp.set(row); tp.set(target);
    const r = L.ce.find((x) => x.id === row), c = r.cases[target];
    cols.replaceChildren(...r.logits.map((z, j) => {
      const g = c.grad[j];
      return el('div', { class: `lw-col${j === target ? ' target' : ''}` },
        el('div', { class: 'lw-h' }, `class ${j}${j === target ? ' (correct)' : ''}`),
        el('div', { class: 'lw-row' }, el('b', {}, 'logit'), fmt(z)),
        el('div', { class: 'lw-row' }, el('b', {}, 'probability'), fmt(c.softmax[j])),
        el('div', { class: 'lw-bar', style: `width:${(100 * c.softmax[j]).toFixed(1)}%`, 'aria-hidden': 'true' }),
        el('div', { class: 'lw-row' }, el('b', {}, 'gradient'), fmt(g)),
        el('div', { class: `lw-gbar ${g < 0 ? 'up' : 'down'}`, style: `width:${(100 * Math.abs(g)).toFixed(1)}%`, 'aria-hidden': 'true' }));
    }));
    read.innerHTML = `<span><b>loss</b> ${fmt(c.loss)}</span><span><b>probability of the correct class</b> ${fmt(c.softmax[target])}</span>`;
  }
  draw();
  return { draw };
}

// ======================================================================================================================
export function mountGradientCompare(host, D) {
  injectStyles();
  const C = D.files.losses.curves, z = C.z;
  let k = 0;
  const input = el('input', { type: 'range', id: 'lw-z', min: 0, max: z.length - 1, step: 1, value: k });
  input.addEventListener('input', () => { k = Number(input.value); mark(); });
  const box = el('div'), read = el('div', { class: 'lw-read', 'aria-live': 'polite' });
  const key = el('div', { class: 'lw-key', 'aria-hidden': 'true' }, el('span', {}, 'solid line: cross-entropy'), el('span', {}, 'dashed line: squared error on the probabilities'));
  host.replaceChildren(el('div', { class: 'lw' }, el('div', { class: 'lw-ctl' }, el('div', { class: 'lw-field' }, el('label', { for: 'lw-z', class: 'lw-lbl' }, 'logit of the correct class'), input)), box, key, read));
  srTable(host, { columns: ['logit', 'cross-entropy gradient', 'squared error gradient'], rows: z.map((v, i) => [v, C.ce_grad[i], C.sq_grad[i]]), caption: 'Gradient with respect to the logit of the correct class' });
  const paint = (pl) => {
    pl.axes({ xTicks: ticks([-6, 6], 6), yTicks: [-1, -0.75, -0.5, -0.25, 0] });
    pl.layer('curves').append(svg('line', { class: 'lw-zero', x1: pl.sx(-6), x2: pl.sx(6), y1: pl.sy(0), y2: pl.sy(0) }));
    pl.curve(z.map((v, i) => [v, C.ce_grad[i]]), { class: 'plot-curve', layer: 'curves' });
    pl.curve(z.map((v, i) => [v, C.sq_grad[i]]), { class: 'lw-sq', layer: 'curves' });
    pl.layer('live');
  };
  const p = createPlot(box, { width: 640, height: 280, x: [-6, 6], y: [-1.05, 0.08], xLabel: 'logit of the correct class', yLabel: 'gradient', ariaLabel: '', onResize: (pl) => { paint(pl); mark(); } });
  paint(p);

  function mark() {
    p.clear('live');
    p.dot(z[k], C.ce_grad[k], { r: 6, layer: 'live' });
    p.dot(z[k], C.sq_grad[k], { class: 'lw-dot2', r: 5, layer: 'live' });
    input.setAttribute('aria-valuetext', `logit ${fmt(z[k])}`);
    p.root.setAttribute('aria-label', `Gradients for a logit of ${fmt(z[k])}: cross-entropy ${fmt(C.ce_grad[k])}, squared error ${fmt(C.sq_grad[k])}.`);
    read.innerHTML = `<span><b>logit</b> ${fmt(z[k])}</span><span><b>cross-entropy</b> loss ${fmt(C.ce_loss[k])}, gradient ${fmt(C.ce_grad[k])}</span><span><b>squared error</b> loss ${fmt(C.sq_loss[k])}, gradient ${fmt(C.sq_grad[k])}</span>`;
  }
  mark();
  return { mark };
}
