// widgets/generalization-widgets.js — the widgets of the lesson generalization (brief §6 Part 3, C2, C10). They show only what
// TorC recorded in data/generalization.json (tools/trace/trace_generalization.cpp): a polynomial of degree 15 (nn::Linear(15, 1) on
// the powers of x) trained with optim::AdamW on noisy points of the wave sin(pi x).
//
//   mountRemedies(host, D)   four runs of the same model: no weight decay, weight decay 0.1, weight decay 1 (all on the ten
//                            training points), and no weight decay on a hundred training points. Top: the model's curve with the
//                            training and validation points and the wave. Bottom: the typical miss on the training and the
//                            validation set at each recorded step, and the typical miss of the wave itself. Presets pick the
//                            run, a slider picks the step. The test set is never shown here (C2).
//   The validation-stop game is in widgets/val-stop.js and uses the helpers below.
//
// The typical miss is the square root of a loss (nn::MSELoss), so it is in the unit of the targets. Pure helpers are exported
// for site/test/generalization.test.mjs; the module touches the DOM only in mount* and in the draw* helpers that mount* calls.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable } from './plot.js';

// ---- pure helpers ---------------------------------------------------------------------------------------------------------
// x position of a step on the log axis (step 0 sits at the left edge)
export const stepX = (step) => Math.log10(step + 1);
// the typical miss of a mean squared error: its square root, in the unit of the targets
export const miss = (loss) => Math.sqrt(loss);
// a readout number with 2 significant figures (brief §9), with a comma from 1,000 up
export function fmt2(v) {
  if (!Number.isFinite(v)) return '?';
  if (v === 0) return '0';
  if (Math.abs(v) >= 100) return Number(v.toPrecision(2)).toLocaleString('en-US');   // 441 -> 440, as the prose
  const s = String(Number(v.toPrecision(2)));   // 0.2, not 0.20, as the prose numbers of T.human
  return s.startsWith('-') ? `−${s.slice(1)}` : s;
}
// a whole number with a comma every three digits from 1,000 up (decision 22)
export const fmtInt = (n) => Math.round(n).toLocaleString('en-US');
// the index of the checkpoint with the lowest validation loss (the first one on a tie): the validation stop
export function lowestVal(checkpoints) {
  let b = 0;
  checkpoints.forEach((c, i) => { if (c.val_loss < checkpoints[b].val_loss) b = i; });
  return b;
}
// Spread labels apart vertically: items [{y}] in pixels; returns the new y of each item (same order), at least `gap` apart and
// inside [top, bottom]. Used for the curve labels at the right end of the miss plot, so they never overlap on a phone.
export function spreadLabels(ys, top, bottom, gap = 14) {
  const order = ys.map((y, i) => [Math.min(bottom, Math.max(top, y)), i]).sort((a, b) => a[0] - b[0]);
  for (let j = 1; j < order.length; j++) order[j][0] = Math.max(order[j][0], order[j - 1][0] + gap);
  const over = order.length ? order.at(-1)[0] - bottom : 0;
  if (over > 0) for (const o of order) o[0] -= over;
  for (let j = order.length - 2; j >= 0; j--) order[j][0] = Math.min(order[j][0], order[j + 1][0] - gap);
  const out = new Array(ys.length);
  for (const [y, i] of order) out[i] = y;
  return out;
}
// The four runs of the remedies widget, in the order of the preset buttons. `run` is the trace object with `checkpoints`,
// `train` the training points that run used.
export function cases(G) {
  const ten = [G.data.train_x, G.data.train_t];
  return [
    { key: 'none', label: 'no weight decay', run: G.runs[0], train: ten },
    { key: 'mid', label: `weight decay ${G.runs[1].weight_decay}`, run: G.runs[1], train: ten },
    { key: 'high', label: `weight decay ${G.runs[2].weight_decay}`, run: G.runs[2], train: ten },
    { key: 'more', label: `${G.more_data.train_samples} training points`, run: G.more_data, train: [G.more_data.train_x, G.more_data.train_t] },
  ];
}
// the wave that made the data (input of the trace program, not library output)
export const wave = (x) => Math.sin(Math.PI * x);
export const X_TICKS = [0, 10, 100, 1000, 10000];
export const MISS_RANGE = [0, 0.8];

// ---- styles ---------------------------------------------------------------------------------------------------------------
const CSS = `
.gw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0;--val:#d2a8ff;--test:var(--ok);--tr:var(--warn)}
.gw-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .5rem}
.gw-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.gw-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.gw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.gw-field{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.gw-field input[type=range]{width:min(14rem,60vw);accent-color:var(--fwd)}
.gw-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.85rem;color:var(--muted);margin:.2rem 0 .6rem;max-width:44rem}
.gw-key i{display:inline-block;width:.7em;height:.7em;border-radius:50%;margin-right:.35em;vertical-align:-.05em}
.gw-key .k-train{background:var(--fg)}
.gw-key .k-val{border:2px solid var(--val)}
.gw-key .k-test{background:var(--test)}
.gw-read{display:grid;gap:.15rem;margin:.5rem 0 0;max-width:44rem;font-size:.95rem}
.gw-read p{margin:0}
.gw-read b{font-weight:600}
.gw-read .r-train b{color:var(--tr)}
.gw-read .r-val b{color:var(--val)}
.gw-read .r-test b{color:var(--test)}
.gw-read .r-note{color:var(--muted);font-size:.85rem}
.gw-train{fill:var(--fg);stroke:var(--bg);stroke-width:1.5}
.gw-val{fill:none;stroke:var(--val);stroke-width:1.6}
.gw-test{fill:var(--test);stroke:var(--bg);stroke-width:1}
.gw-model{stroke:var(--fwd);stroke-width:2.5;fill:none}
.gw-wave{stroke:var(--muted);stroke-width:1.4;stroke-dasharray:5 4;fill:none}
.gw-ltrain{stroke:var(--tr);stroke-width:2.2;stroke-dasharray:7 4;fill:none}
.gw-lval{stroke:var(--val);stroke-width:2.6;fill:none}
.gw-ltest{stroke:var(--test);stroke-width:2.2;stroke-dasharray:2 3;fill:none}
.gw-floor{stroke:var(--muted);stroke-width:1.3;stroke-dasharray:3 3}
.gw-now{stroke:var(--fg);stroke-width:1;stroke-dasharray:2 3;opacity:.7}
.gw-stop{stroke:var(--fg);stroke-width:1.6}
.gw-dtrain{fill:var(--tr);stroke:var(--bg);stroke-width:1.5}
.gw-dval{fill:var(--val);stroke:var(--bg);stroke-width:1.5}
.gw-dtest{fill:var(--test);stroke:var(--bg);stroke-width:1.5}
.gw-t{font-size:12px;paint-order:stroke;stroke:var(--panel);stroke-width:4px;stroke-linejoin:round}
.gw-t.c-model{fill:var(--fwd)}
.gw-t.c-wave,.gw-t.c-floor{fill:var(--muted)}
.gw-t.c-train{fill:var(--tr)}
.gw-t.c-val{fill:var(--val)}
.gw-t.c-test{fill:var(--test)}
.gw-result{margin:.6rem 0 0;padding:.6rem .8rem;border:1px solid var(--border);border-radius:9px;background:var(--panel2);max-width:44rem}
.gw-result[hidden]{display:none}
.gw-result p{margin:.2rem 0}
`;
export function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('gw-styles')) return;
  document.head.append(el('style', { id: 'gw-styles' }, CSS));
}

// ---- drawing (DOM) ----------------------------------------------------------------------------------------------------------
// The fit plot: the wave, the points and the model's curve. `points` = [{xs, ts, cls, r}], `curve` = the model on G.data.grid.
export function fitPlot(host, label) {
  return (onResize) => createPlot(host, { width: 640, height: 240, x: [-1.05, 1.05], y: [-1.7, 1.7], xLabel: 'x', yLabel: 'target', ariaLabel: label, onResize });
}
export function drawFit(pl, G, { points, curve }) {
  pl.axes({ xTicks: [-1, -0.5, 0, 0.5, 1], yTicks: [-1, 0, 1] });
  const xs = G.data.grid;
  pl.curve(xs.map((x) => [x, wave(x)]), { class: 'gw-wave', layer: 'wave' });
  for (const p of points) p.xs.forEach((x, i) => pl.dot(x, p.ts[i], { class: p.cls, r: p.r, layer: 'data' }));
  pl.curve(xs.map((x, i) => [x, Math.max(-3, Math.min(3, curve[i]))]), { class: 'gw-model', layer: 'model' });
  // labels: the wave under its lowest point; the model above the highest point of its curve between x = 0.2 and 0.8, so the
  // label never sits on the curve (kept on the plot)
  pl.text(-0.5, -1, 'the wave', { class: 'gw-t c-wave', layer: 'labels', dy: 18, anchor: 'middle' });
  const top = Math.max(...xs.map((x, i) => (x >= 0.2 && x <= 0.8 ? curve[i] : -9))), my = Math.min(1.45, Math.max(-1.3, top));
  pl.text(0.5, my, 'the model', { class: 'gw-t c-model', layer: 'labels', dy: -10, anchor: 'middle' });
}
// The miss plot: typical miss against the step (log axis). `lines` = [{cs: checkpoints, key: 'train_loss'|…, cls, label, lcls}],
// `floor` = an optional horizontal line {y, label}, `at` = the selected checkpoint index (a vertical line and a dot per curve),
// `stop` = an optional committed step (a solid vertical line).
export function missPlot(host, label) {
  return (onResize) => createPlot(host, { width: 640, height: 230, x: [0, stepX(20000)], y: MISS_RANGE, xLabel: 'training step (log axis)', yLabel: 'typical miss', margin: { l: 56, r: 86 }, ariaLabel: label, onResize });
}
export function drawMiss(pl, { lines, floor, at, stop, xMax }) {
  pl.axes({ xTicks: X_TICKS.map(stepX), yTicks: [0, 0.2, 0.4, 0.6, 0.8], xFmt: (v) => fmtInt(10 ** v - 1), yFmt: (v) => String(v) });
  const ends = [];
  if (floor) {
    pl.line(0, floor.y, xMax, floor.y, { class: 'gw-floor', layer: 'curves' });
    ends.push({ y: pl.sy(floor.y), text: floor.label, cls: 'c-floor' });
  }
  for (const l of lines) {
    pl.curve(l.cs.map((c) => [stepX(c.step), miss(c[l.key])]), { class: l.cls, layer: 'curves' });
    ends.push({ y: pl.sy(miss(l.cs.at(-1)[l.key])), text: l.label, cls: l.lcls });
  }
  if (stop !== undefined) pl.line(stepX(stop), MISS_RANGE[0], stepX(stop), MISS_RANGE[1], { class: 'gw-stop', layer: 'marks' });
  if (at !== undefined) {
    const s = lines[0].cs[at].step;
    pl.line(stepX(s), MISS_RANGE[0], stepX(s), MISS_RANGE[1], { class: 'gw-now', layer: 'marks' });
    for (const l of lines) pl.dot(stepX(s), miss(l.cs[at][l.key]), { class: l.dot, r: 5, layer: 'marks' });
  }
  // curve labels in the right margin, next to the end of each curve, spread so that they never overlap
  const { t, h } = pl.plotArea, ys = spreadLabels(ends.map((e) => e.y), t + 6, t + h - 2);
  ends.forEach((e, i) => pl.root.append(svg('text', { class: `gw-t ${e.cls}`, x: pl.plotArea.l + pl.plotArea.w + 6, y: ys[i] + 4 }, e.text)));
}
export function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return {
    root: el('div', { class: 'gw-presets', role: 'group', 'aria-label': label }, el('span', { class: 'gw-lbl', 'aria-hidden': 'true' }, label), ...btns),
    set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))),
  };
}

// ======================================================================================================================
export function mountRemedies(host, D) {
  injectStyles();
  const G = D.files.generalization, C = cases(G);
  const steps = G.runs[0].checkpoints.map((c) => c.step);
  const floorY = miss(G.derived.true_wave_loss.val);
  let which = C[0], k = steps.length - 1;
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  const pr = presets('Run', C.map((c) => [c.key, c.label]), (key) => { which = C.find((c) => c.key === key); draw(); announce(); });
  const input = el('input', { type: 'range', id: 'gw-step', min: 0, max: steps.length - 1, step: 1, value: k });
  input.addEventListener('input', () => { k = Number(input.value); draw(); });
  input.addEventListener('change', announce);
  const fitBox = el('div'), missBox = el('div');
  const key = el('p', { class: 'gw-key' }, el('span', {}, el('i', { class: 'k-train', 'aria-hidden': 'true' }), 'training points'), el('span', {}, el('i', { class: 'k-val', 'aria-hidden': 'true' }), 'validation points'));
  const note = el('p', { class: 'gw-key' }, 'On the step axis, each mark after 10 is ten times the mark before it. The grey line labelled “the wave” shows how much the wave itself misses a validation point.');
  const rStep = el('p'), rTrain = el('p', { class: 'r-train' }), rVal = el('p', { class: 'r-val' }), rW = el('p');
  host.replaceChildren(el('div', { class: 'gw' },
    el('div', { class: 'gw-ctl' }, pr.root),
    el('div', { class: 'gw-ctl' }, el('div', { class: 'gw-field' }, el('label', { for: 'gw-step', class: 'gw-lbl' }, 'after step'), input)),
    fitBox, key, missBox, note,
    el('div', { class: 'gw-read' }, rStep, rTrain, rVal, el('p', { class: 'r-note' }, 'A smaller miss is better.'), rW), say));
  const table = srTable(host, { columns: ['step', 'typical miss on the training set', 'typical miss on the validation set', 'sum of the squared weights'], rows: [], caption: 'Typical misses and weight size of the chosen run, step by step' });

  const paintFit = (pl) => {
    pl.clear('wave'); pl.clear('data'); pl.clear('model'); pl.clear('labels');
    const c = which.run.checkpoints[k];
    drawFit(pl, G, {
      points: [{ xs: G.data.val_x, ts: G.data.val_t, cls: 'gw-val', r: 3.4 }, { xs: which.train[0], ts: which.train[1], cls: 'gw-train', r: which.key === 'more' ? 3 : 4.5 }],
      curve: c.curve,
    });
  };
  const paintMiss = (pl) => {
    pl.clear('curves'); pl.clear('marks');
    pl.root.querySelectorAll(':scope > text.gw-t').forEach((n) => n.remove());
    const cs = which.run.checkpoints;
    drawMiss(pl, {
      lines: [
        { cs, key: 'val_loss', cls: 'gw-lval', label: 'validation', lcls: 'c-val', dot: 'gw-dval' },
        { cs, key: 'train_loss', cls: 'gw-ltrain', label: 'training', lcls: 'c-train', dot: 'gw-dtrain' },
      ],
      floor: { y: floorY, label: 'the wave' }, at: k, xMax: stepX(G.setup.steps),
    });
  };
  const fit = fitPlot(fitBox, '')((pl) => paintFit(pl));
  const loss = missPlot(missBox, '')((pl) => paintMiss(pl));
  // axes are drawn by drawFit/drawMiss; a redraw without resize must not stack them, so redraw whole plots through these
  const redraw = (pl, paint) => { pl.root.querySelectorAll(':scope > g.plot-axes').forEach((g) => g.remove()); paint(pl); };

  function announce() {
    const c = which.run.checkpoints[k];
    say.textContent = `${which.label}, after step ${fmtInt(c.step)}: training miss about ${fmt2(miss(c.train_loss))}, validation miss about ${fmt2(miss(c.val_loss))}, size of the weights about ${fmt2(c.weight_sum_sq)}.`;
  }
  function draw() {
    pr.set(which.key);
    const c = which.run.checkpoints[k];
    input.setAttribute('aria-valuetext', `after step ${fmtInt(c.step)}`);
    redraw(fit, paintFit);
    redraw(loss, paintMiss);
    fit.root.setAttribute('aria-label', `The model with ${which.label} after step ${fmtInt(c.step)}, with ${which.train[0].length} training points, ${G.data.val_x.length} validation points and the wave.`);
    loss.root.setAttribute('aria-label', `Typical miss on the training set and on the validation set with ${which.label}, against the training step on a log axis. After step ${fmtInt(c.step)}: training ${fmt2(miss(c.train_loss))}, validation ${fmt2(miss(c.val_loss))}. The wave itself misses a validation point by about ${fmt2(floorY)}.`);
    rStep.innerHTML = `<b>${which.label}</b>, after step ${fmtInt(c.step)}`;
    rTrain.innerHTML = `The model misses a <b>training point</b> by about ${fmt2(miss(c.train_loss))}.`;
    rVal.innerHTML = `It misses a <b>validation point</b> by about ${fmt2(miss(c.val_loss))}.`;
    rW.innerHTML = `<b>Size of the weights</b> (the sum of their squares) is about ${fmt2(c.weight_sum_sq)}.`;
    table.update(which.run.checkpoints.map((q) => [q.step, miss(q.train_loss), miss(q.val_loss), q.weight_sum_sq]));
  }
  draw();
  return { draw, destroy() { fit.destroy(); loss.destroy(); } };
}
