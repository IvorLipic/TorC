// widgets/training-widgets.js — the widgets of the lesson training-loop. Each one shows only what TorC recorded (data/training.json,
// tools/trace/trace_training.cpp):
//
//   mountBatchViewer(host, D)       a data loader over the 100 samples: batch size 10 or 32, shuffle off or on. "Next batch" takes the
//                                   next recorded batch, "Next epoch" starts the second recorded epoch (a new order when shuffle is on)
//   mountGradientEstimate(host, D)  the gradient of each batch of 10 (first epoch of training, seed 7) as a dot in the plane
//                                   (gradient for w, gradient for b), next to the gradient on all 100 samples, and their average
//   mountBatchCompare(host, D)      the typical miss on all 100 samples after each step: batch 10 (three shuffled orders) against
//                                   batch 100, by steps or by samples seen; and batch 10 in sorted order against shuffled order
//
// Pure helpers (exported for the tests): loaderState, typicalMiss, sig, compareView. The module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable } from './plot.js';
import { spreadLabels } from './curve-labels.js';

const CSS = `
.tl{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.tl-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .6rem}
.tl-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.tl-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.tl .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.tl .btn:disabled{opacity:.45;cursor:default}
.tl-calls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 .6rem}
.tl-grid{display:grid;grid-template-columns:repeat(10,minmax(0,2.1rem));gap:3px;max-width:100%}
.tl-cell{font:.72rem ui-monospace,monospace;text-align:center;padding:.22rem 0;border-radius:4px;border:1px solid var(--border);background:var(--panel2);color:var(--muted)}
.tl-cell.used{opacity:.35}
.tl-cell.cur{border-color:var(--fwd);background:rgba(88,196,221,.25);color:var(--fg);font-weight:700}
.tl-status{font-size:.95rem;margin:.6rem 0 0}
.tl-batch{font-size:.88rem;color:var(--muted);margin:.3rem 0 0;overflow-wrap:anywhere}
.tl-note{font-size:.85rem;color:var(--muted);margin:.4rem 0 0}
.tl-field{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.tl-field input[type=range]{width:12rem;max-width:60vw;accent-color:var(--fwd)}
.tl-read{font-size:.92rem;margin:.5rem 0 0}
.tl-c10{stroke:var(--fwd);stroke-width:1.6;fill:none}
.tl-c100{stroke:var(--warn);stroke-width:2.6;fill:none;stroke-dasharray:8 4}
.tl-csort{stroke:var(--bwd);stroke-width:2.4;fill:none;stroke-dasharray:3 3}
.tl-floor{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:2 4}
.tl-mark{stroke:var(--fg);stroke-width:1;stroke-dasharray:3 3;opacity:.6}
.tl-t10{fill:var(--fwd);font-size:12px;font-weight:600}
.tl-t100{fill:var(--warn);font-size:12px;font-weight:600}
.tl-tsort{fill:var(--bwd);font-size:12px;font-weight:600}
.tl-tfloor{fill:var(--muted);font-size:11px}
.tl-dot{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.tl-dot.cur{fill:var(--fg);stroke:var(--fwd);stroke-width:2.5}
.tl-star{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.tl-avg{fill:none;stroke:var(--ok);stroke-width:2.5}
.tl-tstar{fill:var(--warn);font-size:12px;font-weight:600}
.tl-tavg{fill:var(--ok);font-size:12px;font-weight:600}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('tl-styles')) return;
  document.head.append(el('style', { id: 'tl-styles' }, CSS));
}
// A toggle group: each button keeps its fixed label and carries aria-pressed.
function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'tl-presets', role: 'group', 'aria-label': label }, el('span', { class: 'tl-lbl', 'aria-hidden': 'true' }, label), ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}

// v with n significant figures, with the minus sign U+2212 (widget readouts use 2-3 figures, docs/authoring.md "Widgets")
export function sig(v, n = 2) {
  if (!Number.isFinite(v)) return String(v);
  const s = String(Number(Math.abs(v).toPrecision(n)));
  return v < 0 && Number(s) !== 0 ? `−${s}` : s;
}
// the typical miss: the square root of the mean squared miss (the loss), in the unit of the targets
export const typicalMiss = (loss) => Math.sqrt(loss);

// ======================================================================================================================
// The state of a recorded loader after `taken` batches of `epoch` (0 or 1). Exported for tests.
export function loaderState(loader, epoch, taken) {
  const batches = loader.epochs[epoch];
  const used = new Set(batches.slice(0, Math.max(0, taken - 1)).flat());
  const current = taken > 0 ? batches[taken - 1] : [];
  return { batches: batches.length, hasNext: taken < batches.length, used, current, steps: epoch * batches.length + taken };
}

export function mountBatchViewer(host, D) {
  injectStyles();
  const R = D.files.training, N = R.setup.samples;
  const epochsRecorded = R.loaders[0].epochs.length;
  let bs = 10, shuffle = false, epoch = 0, taken = 0, over = false;
  const loader = () => R.loaders.find((l) => l.batch_size === bs && l.shuffle === shuffle);
  const restart = (msg) => { epoch = 0; taken = 0; over = false; draw(msg); };
  const sizes = presets('Batch size', [...new Set(R.loaders.map((l) => l.batch_size))].map((b) => [b, String(b)]), (k) => { bs = k; restart(`Batch size ${k}. Back to the start of epoch 1.`); });
  const order = presets('Shuffle', [[false, 'off'], [true, 'on']], (k) => { shuffle = k; restart(`Shuffle ${k ? 'on' : 'off'}. Back to the start of epoch 1.`); });
  const nextBtn = el('button', { type: 'button', class: 'btn small primary' }, 'Next batch');
  const epochBtn = el('button', { type: 'button', class: 'btn small' }, 'Next epoch');
  const againBtn = el('button', { type: 'button', class: 'btn small' }, 'Start again');
  nextBtn.addEventListener('click', () => { const s = loaderState(loader(), epoch, taken); if (s.hasNext) taken += 1; else over = true; draw(); });
  epochBtn.addEventListener('click', () => { if (epoch + 1 < epochsRecorded) { epoch += 1; taken = 0; over = false; draw(); } });
  againBtn.addEventListener('click', () => restart('Back to the start of epoch 1.'));
  const cells = Array.from({ length: N }, (_, i) => el('div', { class: 'tl-cell' }, String(i)));
  const grid = el('div', { class: 'tl-grid', 'aria-hidden': 'true' }, ...cells);
  const status = el('p', { class: 'tl-status', 'aria-live': 'polite' });
  const batch = el('p', { class: 'tl-batch' });
  const note = el('p', { class: 'tl-note' }, `The squares are the ${N} samples, numbered from 0. The bright squares are the batch of the current step.`);
  host.replaceChildren(el('div', { class: 'tl' },
    el('div', { class: 'tl-ctl' }, sizes.root, order.root),
    el('div', { class: 'tl-calls', role: 'group', 'aria-label': 'Loader buttons' }, nextBtn, epochBtn, againBtn),
    grid, status, batch, note));

  function draw(msg) {
    sizes.set(bs); order.set(shuffle);
    const s = loaderState(loader(), epoch, taken);
    const cur = new Set(s.current);
    cells.forEach((c, i) => { c.className = `tl-cell${cur.has(i) ? ' cur' : s.used.has(i) ? ' used' : ''}`; });
    const where = `Epoch ${epoch + 1}, batch ${taken} of ${s.batches}. Steps so far: ${s.steps}.`;
    const end = over ? (epoch + 1 < epochsRecorded ? ` Epoch ${epoch + 1} is over. Every sample was in one batch. Press Next epoch.` : ' This epoch is over too. This viewer has two epochs.') : '';
    status.textContent = `${msg ? `${msg} ` : ''}${where}${end}`;
    batch.textContent = taken > 0 ? `This batch holds ${s.current.length} samples: ${s.current.join(', ')}.` : 'No batch yet. Press Next batch.';
    nextBtn.disabled = over;
    epochBtn.disabled = epoch + 1 >= epochsRecorded;
  }
  draw();
  return { draw };
}

// ======================================================================================================================
export function mountGradientEstimate(host, D) {
  injectStyles();
  const G = D.files.training.batch_compare.gradient_estimate, B = G.batches;
  let shown = 0, avg = false;
  const nextBtn = el('button', { type: 'button', class: 'btn small primary' }, 'Next batch');
  const avgBtn = el('button', { type: 'button', class: 'btn small' }, 'Show the average of the dots');
  const resetBtn = el('button', { type: 'button', class: 'btn small' }, 'Reset');
  nextBtn.addEventListener('click', () => { if (shown < B.length) { shown += 1; mark(); } });
  avgBtn.addEventListener('click', () => { if (shown === B.length) { avg = true; mark(); } });
  resetBtn.addEventListener('click', () => { shown = 0; avg = false; mark('The dots are cleared.'); });
  const box = el('div');
  const read = el('p', { class: 'tl-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'tl' }, el('div', { class: 'tl-calls', role: 'group', 'aria-label': 'Batch buttons' }, nextBtn, avgBtn, resetBtn), box, read,
    el('p', { class: 'tl-note' }, 'Each dot is the gradient of one batch of 10 samples at w = 0 and b = 0. Its position across is the gradient for w, and its height is the gradient for b.')));
  srTable(host, { columns: ['batch', 'gradient for w', 'gradient for b'], rows: [...B.map((b, i) => [i + 1, b.gw, b.gb]), ['all samples', G.full.gw, G.full.gb]], caption: 'The gradient of each batch and of all samples' });
  const lo = Math.floor(Math.min(G.range.gw_min, G.range.gb_min) * 2) / 2 - 0.25;
  const paint = (pl) => {
    pl.axes({ xTicks: [-3, -2, -1, 0].filter((t) => t >= lo), yTicks: [-3, -2, -1, 0].filter((t) => t >= lo), fmt: (t) => sig(t) });
    pl.layer('dots'); pl.layer('marks');
    mark();
  };
  const p = createPlot(box, { width: 420, height: 380, x: [lo, 0.5], y: [lo, 0.5], xLabel: 'gradient for w', yLabel: 'gradient for b', ariaLabel: '', onResize: paint, margin: { r: 12 } });

  function mark(msg) {
    p.clear('dots'); p.clear('marks');
    B.slice(0, shown).forEach((b, i) => p.dot(b.gw, b.gb, { class: `tl-dot${i === shown - 1 && !avg ? ' cur' : ''}`, r: i === shown - 1 && !avg ? 7 : 5, layer: 'dots' }));
    // the star: the gradient on all samples
    const sx = p.sx(G.full.gw), sy = p.sy(G.full.gb), r = 8;
    const star = Array.from({ length: 10 }, (_, k) => { const a = -Math.PI / 2 + (k * Math.PI) / 5, q = k % 2 ? r * 0.45 : r; return `${(sx + q * Math.cos(a)).toFixed(1)},${(sy + q * Math.sin(a)).toFixed(1)}`; }).join(' ');
    p.layer('marks').append(svg('polygon', { class: 'tl-star', points: star }));
    p.text(G.full.gw, G.full.gb, 'all 100 samples', { class: 'tl-tstar', dx: 11, dy: -10, layer: 'marks' });
    if (avg) {
      p.layer('marks').append(svg('circle', { class: 'tl-avg', cx: p.sx(G.mean.gw), cy: p.sy(G.mean.gb), r: 12 }));
      p.text(G.mean.gw, G.mean.gb, 'average of the dots', { class: 'tl-tavg', dx: 15, dy: 18, layer: 'marks' });
    }
    nextBtn.disabled = shown >= B.length;
    avgBtn.disabled = shown < B.length || avg;
    const full = `All 100 samples give ${sig(G.full.gw)} for w and ${sig(G.full.gb)} for b.`;
    let text;
    if (avg) text = `The average of the ten dots is ${sig(G.mean.gw)} for w and ${sig(G.mean.gb)} for b. ${full}`;
    else if (shown === 0) text = 'No batch yet. Press Next batch.';
    else { const b = B[shown - 1]; text = `Batch ${shown} of ${B.length} gives ${sig(b.gw)} for w and ${sig(b.gb)} for b. ${full}`; }
    read.textContent = `${msg ? `${msg} ` : ''}${text}`;
    p.root.setAttribute('aria-label', `The gradients of ${shown} of ${B.length} batches as dots, and the gradient of all samples as a star. ${text}`);
  }
  paint(p);
  return { mark };
}

// ======================================================================================================================
// The three views of the comparison, as data for the plot: curves of [x, typical miss] and the slider range. Exported for tests.
export function compareView(R, view) {
  const C = R.batch_compare, S = R.sorted_order;
  const runs = (bs) => C.runs.filter((r) => r.batch_size === bs);
  const miss = (loss, upTo, perStep = 1) => loss.slice(0, upTo + 1).map((l, k) => [k * perStep, typicalMiss(l)]);
  if (view === 'steps') {
    const n = 100;
    return { xLabel: 'steps', xMax: n, step: 1, unit: 'steps', curves: [
      ...runs(10).map((r) => ({ key: 'b10', seed: r.seed, pts: miss(r.loss, n) })),
      { key: 'b100', pts: miss(runs(100)[0].loss, n) }] };
  }
  if (view === 'samples') {
    const n = C.equal_samples[0].samples;
    return { xLabel: 'samples seen', xMax: n, step: 100, unit: 'samples', curves: [
      ...runs(10).map((r) => ({ key: 'b10', seed: r.seed, pts: miss(r.loss, n / 10, 10) })),
      { key: 'b100', pts: miss(runs(100)[0].loss, n / 100, 100) }] };
  }
  const n = 60;
  return { xLabel: 'steps', xMax: n, step: 1, unit: 'steps', curves: [
    { key: 'b10', seed: S.shuffled_seed, pts: miss(S.shuffled_loss, n) },
    { key: 'sort', pts: miss(S.sorted_loss, n) }] };
}
// the value of a curve at x (the last recorded point at or before x)
export const valueAt = (pts, x) => { let v = pts[0][1]; for (const [px, py] of pts) if (px <= x + 1e-9) v = py; return v; };

const NAMES = { b10: 'batch 10', b100: 'batch 100', sort: 'batch 10, sorted' };
const LABEL = { steps: { b10: 'batch 10 (three orders)', b100: 'batch 100' }, samples: { b10: 'batch 10 (three orders)', b100: 'batch 100' }, sorted: { b10: 'batch 10, shuffled', sort: 'batch 10, sorted' } };
const CLS = { b10: 'tl-c10', b100: 'tl-c100', sort: 'tl-csort' };
const TCLS = { b10: 'tl-t10', b100: 'tl-t100', sort: 'tl-tsort' };
const TICKS = { steps: [0, 25, 50, 75, 100], samples: [0, 1000, 2000, 3000], sorted: [0, 20, 40, 60] };
const START_AT = { steps: 10, samples: 1000, sorted: 1 };   // where the marker starts in each view (not an answer of a task)

export function mountBatchCompare(host, D) {
  injectStyles();
  const R = D.files.training, floor = typicalMiss(R.true_line_loss);
  let view = 'steps', at = START_AT.steps, plot = null;
  const views = presets('Compare', [['steps', 'Same number of steps'], ['samples', 'Same number of samples'], ['sorted', 'Sorted or shuffled']], (k) => {
    view = k;
    at = START_AT[k];
    build();
    mark(`Showing ${k === 'steps' ? 'the same number of steps' : k === 'samples' ? 'the same number of samples' : 'sorted against shuffled order'}.`);
  });
  const slider = el('input', { type: 'range', id: 'tl-at', min: 0, max: 100, step: 1, value: at });
  slider.addEventListener('input', () => { at = Number(slider.value); mark(); });
  const label = el('label', { for: 'tl-at', class: 'tl-lbl' }, 'after');
  const box = el('div');
  const read = el('p', { class: 'tl-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'tl' }, el('div', { class: 'tl-ctl' }, views.root), el('div', { class: 'tl-ctl' }, el('div', { class: 'tl-field' }, label, slider)), box, read,
    el('p', { class: 'tl-note' }, 'The height of a curve is the typical miss of the line on all 100 samples, the square root of the loss. Lower is better.')));
  const sr = srTable(host, { columns: ['x', 'curve', 'typical miss'], rows: [], caption: 'Typical miss after each step' });

  // the x range changes with the view, so each view gets a new plot; paint() draws it again on resize
  function build() {
    const v = compareView(R, view);
    views.set(view);
    slider.max = v.xMax; slider.step = v.step; slider.value = at;
    plot?.destroy();
    box.replaceChildren();
    plot = createPlot(box, { width: 640, height: 300, x: [0, v.xMax], y: [0, 2], xLabel: v.xLabel, yLabel: 'typical miss', ariaLabel: '', margin: { r: 16 }, onResize: paint });
    paint(plot);
  }
  function paint(pl) {
    const v = compareView(R, view), xs = v.xMax;
    pl.axes({ xTicks: TICKS[view], yTicks: [0, 0.5, 1, 1.5, 2], fmt: (t) => (t >= 1000 ? t.toLocaleString('en-US') : sig(t, 2)) });
    pl.line(0, floor, xs, floor, { class: 'tl-floor', layer: 'curves' });
    // the floor label sits where no curve runs: at the right end in the samples view, at the left end in the other two
    const right = view === 'samples';
    pl.text(right ? xs : 0, floor, `the true line: ${sig(floor)}`, { class: 'tl-tfloor', anchor: right ? 'end' : 'start', dx: right ? -4 : 6, dy: -6, layer: 'curves' });
    for (const c of v.curves) pl.curve(c.pts, { class: CLS[c.key], layer: 'curves' });
    // one label per kind of curve, near the left, moved apart if they would collide
    const kinds = [...new Set(v.curves.map((c) => c.key))];
    const lx = xs * (view === 'sorted' ? 0.12 : 0.1);
    const want = kinds.map((k) => pl.sy(valueAt(v.curves.find((c) => c.key === k).pts, lx)) - 10);
    const ys = spreadLabels(want, { gap: 16, min: pl.plotArea.t + 12, max: pl.plotArea.t + pl.plotArea.h - 6 });
    kinds.forEach((k, i) => pl.layer('labels').append(svg('text', { class: TCLS[k], x: pl.sx(lx) + 6, y: ys[i] }, LABEL[view][k])));
    sr.update(v.curves.flatMap((c) => c.pts.filter((_, i) => i % 10 === 0).map(([x, y]) => [x, `${NAMES[c.key]}${c.seed ? `, order ${c.seed}` : ''}`, y])));
    pl.layer('live');
    mark();
  }

  function mark(msg) {
    const v = compareView(R, view);
    plot.clear('live');
    plot.line(at, 0, at, 2, { class: 'tl-mark', layer: 'live' });
    const vals = (key) => v.curves.filter((c) => c.key === key).map((c) => valueAt(c.pts, at));
    for (const c of v.curves) plot.dot(at, valueAt(c.pts, at), { class: TCLS[c.key], r: 3.5, layer: 'live' });
    const range = (a) => { const lo = sig(Math.min(...a)), hi = sig(Math.max(...a)); return lo === hi ? lo : `${lo} to ${hi}`; };
    let text;
    if (view === 'sorted') text = `After ${at} steps of batch 10, the line misses by ${range(vals('sort'))} in sorted order and by ${range(vals('b10'))} in shuffled order.`;
    else {
      const s100 = view === 'samples' ? Math.floor(at / 100) : at, s10 = view === 'samples' ? at / 10 : at;
      text = `After ${view === 'samples' ? `${at.toLocaleString('en-US')} samples` : `${at} steps`}, batch 10 (${s10} steps) misses by ${range(vals('b10'))}. Batch 100 (${s100} steps) misses by ${range(vals('b100'))}.`;
    }
    const where = view === 'samples' ? `after ${at.toLocaleString('en-US')} samples` : `after ${at} steps`;
    slider.setAttribute('aria-valuetext', where);
    label.textContent = where;
    read.textContent = `${msg ? `${msg} ` : ''}${text}`;
    plot.root.setAttribute('aria-label', `Typical miss on all samples, by ${v.xLabel}. ${text}`);
  }
  build();
  return { mark: () => mark() };
}
