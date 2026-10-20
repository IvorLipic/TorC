// widgets/training-widgets.js — the two widgets of the lesson training-loop. Both show only what TorC recorded
// (data/training.json, tools/trace/trace_training.cpp):
//
//   mountBatchViewer(host, D)     a DataLoader over 100 samples: batch size 10 or 32, with or without shuffle. The buttons call
//                                 next_batch() and reset() on the recorded batches of two epochs, and show the error TorC
//                                 throws after the last batch.
//   mountTrainingCurves(host, D)  three training runs that differ only in the batch size: the loss on all samples after each
//                                 epoch (log scale), and the loss of the true line
//
// The module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';
import { createPlot, srTable } from './plot.js';

const CSS = `
.tl{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.tl-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .6rem}
.tl-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.tl-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.tl .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.tl .btn[aria-disabled=true]{opacity:.45;cursor:default}
.tl-calls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 .6rem}
.tl-calls .btn{font-family:ui-monospace,monospace}
.tl-grid{display:grid;grid-template-columns:repeat(10,minmax(0,2.1rem));gap:3px;max-width:100%}
.tl-cell{font:.72rem ui-monospace,monospace;text-align:center;padding:.22rem 0;border-radius:4px;border:1px solid var(--border);background:var(--panel2);color:var(--muted)}
.tl-cell.used{opacity:.35}
.tl-cell.cur{border-color:var(--fwd);background:rgba(88,196,221,.25);color:var(--fg);font-weight:700}
.tl-status{font:.92rem ui-monospace,monospace;margin:.6rem 0 0}
.tl-status b{color:var(--muted);font-weight:600;margin-right:.3rem}
.tl-batch{font:.85rem ui-monospace,monospace;color:#c4ccd5;margin:.3rem 0 0;overflow-wrap:anywhere}
.tl-err{border:1px solid var(--bwd);border-radius:8px;padding:.45rem .7rem;font:.88rem ui-monospace,monospace;color:#ffb3ab;background:#2b1513;margin:.5rem 0 0;overflow-wrap:anywhere}
.tl-note{font-size:.85rem;color:var(--muted);margin:.4rem 0 0}
.tl-field{display:flex;gap:.5rem;align-items:center}
.tl-field input[type=range]{width:12rem;accent-color:var(--fwd)}
.tl-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.2rem 0 0}
.tl-read{display:grid;grid-template-columns:auto auto auto auto auto;gap:.1rem 1rem;font:.85rem ui-monospace,monospace;margin:.5rem 0 0;width:max-content;max-width:100%}
.tl-read b{color:var(--muted);font-weight:600}
.tl-b100{stroke:var(--ok);stroke-width:2.2;stroke-dasharray:2 4;fill:none;stroke-linecap:round}
.tl-b32{stroke:var(--warn);stroke-width:2.2;stroke-dasharray:8 4;fill:none}
.tl-b10{stroke:var(--fwd);stroke-width:2.5;fill:none}
.tl-floor{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:3 3}
.tl-dot{stroke:var(--bg);stroke-width:1.5}
.tl-d10{fill:var(--fwd)}
.tl-d32{fill:var(--warn)}
.tl-d100{fill:var(--ok)}
@media(max-width:560px){.tl-read{grid-template-columns:auto auto auto;font-size:.8rem}.tl-read .wide{display:none}}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('tl-styles')) return;
  document.head.append(el('style', { id: 'tl-styles' }, CSS));
}
function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'tl-presets', role: 'group', 'aria-label': label }, el('span', { class: 'tl-lbl', 'aria-hidden': 'true' }, label), ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}

// The state of a recorded loader after `taken` calls of next_batch() in `epoch` (0 or 1). Exported for tests.
export function loaderState(loader, epoch, taken) {
  const batches = loader.epochs[epoch];
  const used = new Set(batches.slice(0, Math.max(0, taken - 1)).flat());
  const current = taken > 0 ? batches[taken - 1] : [];
  return { batches: batches.length, hasNext: taken < batches.length, used, current };
}

// ======================================================================================================================
export function mountBatchViewer(host, D) {
  injectStyles();
  const R = D.files.training, N = R.setup.samples;
  let bs = 32, shuffle = false, epoch = 0, taken = 0, error = false;
  const restart = () => { epoch = 0; taken = 0; error = false; draw(); };
  const sizes = presets('Batch size', [[10, '10'], [32, '32']], (k) => { bs = k; restart(); });
  const order = presets('Shuffle', [[false, 'off'], [true, 'on']], (k) => { shuffle = k; restart(); });
  const nextBtn = el('button', { type: 'button', class: 'btn small' }, 'next_batch()');
  const resetBtn = el('button', { type: 'button', class: 'btn small' }, 'reset()');
  nextBtn.addEventListener('click', () => { const s = loaderState(loader(), epoch, taken); if (s.hasNext) { taken += 1; error = false; } else error = true; draw(); });
  resetBtn.addEventListener('click', () => { if (epoch === 0) { epoch = 1; taken = 0; error = false; draw(); } });
  const cells = Array.from({ length: N }, (_, i) => el('div', { class: 'tl-cell' }, String(i)));
  const grid = el('div', { class: 'tl-grid', 'aria-hidden': 'true' }, ...cells);
  const status = el('p', { class: 'tl-status' });
  const batch = el('p', { class: 'tl-batch' });
  const err = el('p', { class: 'tl-err', hidden: true });
  const note = el('p', { class: 'tl-note' });
  const seed = el('p', { class: 'tl-note' }, `This loader uses the seed ${R.setup.loader_seed}. A loader without a seed gets a new random order each time the program runs.`);
  const live = el('div', { 'aria-live': 'polite' }, status, err);   // the list of up to 32 indices is not read out on each click
  host.replaceChildren(el('div', { class: 'tl' },
    el('div', { class: 'tl-ctl' }, sizes.root, order.root),
    el('div', { class: 'tl-calls', role: 'group', 'aria-label': 'Calls on the loader' }, nextBtn, resetBtn),
    grid, live, batch, note, seed));
  const loader = () => R.loaders.find((l) => l.batch_size === bs && l.shuffle === shuffle);

  function draw() {
    sizes.set(bs); order.set(shuffle);
    const s = loaderState(loader(), epoch, taken);
    const cur = new Set(s.current);
    cells.forEach((c, i) => { c.className = `tl-cell${cur.has(i) ? ' cur' : s.used.has(i) ? ' used' : ''}`; });
    status.innerHTML = `<span><b>epoch</b> ${epoch + 1}</span> <span><b>batch</b> ${taken} of ${s.batches}</span> <span><b>has_next()</b> ${s.hasNext}</span>`;
    batch.textContent = taken > 0 ? `Batch ${taken} holds ${s.current.length} samples: ${s.current.join(', ')}.` : 'No batch yet. Call next_batch().';
    err.hidden = !error;
    err.textContent = error ? `${R.exhausted.exception}: ${R.exhausted.what}` : '';
    resetBtn.setAttribute('aria-disabled', String(epoch === 1));
    note.textContent = epoch === 1 ? 'The trace records two epochs. Change the batch size or the shuffle setting to start again.' : '';
  }
  draw();
  return { draw };
}

// ======================================================================================================================
const RUNS = [[10, 'tl-b10', 'solid line: batch size 10'], [32, 'tl-b32', 'dashed line: batch size 32'], [100, 'tl-b100', 'dotted line: batch size 100']];
const LOG = [Math.log10(0.005), Math.log10(5)];
const decade = (d) => String(Number((10 ** d).toPrecision(1)));

export function mountTrainingCurves(host, D) {
  injectStyles();
  const R = D.files.training, E = R.setup.epochs;
  const run = (bs) => R.runs.find((r) => r.batch_size === bs);
  let k = 5;
  const input = el('input', { type: 'range', id: 'tl-epoch', min: 0, max: E, step: 1, value: k });
  input.addEventListener('input', () => { k = Number(input.value); mark(); });
  const box = el('div');
  const key = el('div', { class: 'tl-key', 'aria-hidden': 'true' }, ...RUNS.map(([, , t]) => el('span', {}, t)), el('span', {}, 'thin line: the loss of the true line'));
  const read = el('div', { class: 'tl-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'tl' },
    el('div', { class: 'tl-ctl' }, el('div', { class: 'tl-field' }, el('label', { for: 'tl-epoch', class: 'tl-lbl' }, 'after epoch'), input)), box, key, read));
  srTable(host, { columns: ['epoch', ...RUNS.map(([bs]) => `loss, batch size ${bs}`)], rows: run(10).epochs.map((e, i) => [e.epoch, ...RUNS.map(([bs]) => run(bs).epochs[i].loss)]), caption: 'Loss on all samples after each epoch' });

  const paint = (pl) => {
    pl.axes({ xTicks: [0, 5, 10, 15, 20, 25, 30].filter((t) => t <= E), yTicks: [-2, -1, 0], yFmt: decade });
    pl.layer('curves').append(svg('line', { class: 'tl-floor', x1: pl.sx(0), x2: pl.sx(E), y1: pl.sy(Math.log10(R.true_line_loss)), y2: pl.sy(Math.log10(R.true_line_loss)) }));
    for (const [bs, cls] of RUNS) pl.curve(run(bs).epochs.map((e) => [e.epoch, Math.log10(e.loss)]), { class: cls, layer: 'curves' });
    pl.layer('live');
    mark();
  };
  const p = createPlot(box, { width: 640, height: 290, x: [0, E], y: LOG, xLabel: 'epoch', yLabel: 'loss (log scale)', ariaLabel: '', onResize: paint });

  function mark() {
    p.clear('live');
    for (const [bs] of RUNS) {
      const e = run(bs).epochs[k];
      p.dot(e.epoch, Math.log10(e.loss), { class: `tl-dot tl-d${bs}`, r: 5, layer: 'live' });
    }
    input.setAttribute('aria-valuetext', `after epoch ${k}`);
    p.root.setAttribute('aria-label', `Loss after each epoch for batch sizes 10, 32 and 100, on a log scale. After epoch ${k}: ${RUNS.map(([bs]) => `batch size ${bs}, loss ${fmt(run(bs).epochs[k].loss)}`).join('. ')}.`);
    read.replaceChildren(
      el('b', {}, 'batch size'), el('b', {}, 'steps'), el('b', {}, 'loss'), el('b', { class: 'wide' }, 'weight'), el('b', { class: 'wide' }, 'bias'),
      ...RUNS.flatMap(([bs]) => { const e = run(bs).epochs[k]; return [el('span', {}, String(bs)), el('span', {}, String(e.steps)), el('span', {}, fmt(e.loss)), el('span', { class: 'wide' }, fmt(e.weight)), el('span', { class: 'wide' }, fmt(e.bias))]; }));
  }
  paint(p);
  return { mark };
}
