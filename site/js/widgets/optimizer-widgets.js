// widgets/optimizer-widgets.js — the widget of the lesson optimizers. It shows only what TorC recorded (data/optimizers.json,
// tools/trace/trace_optimizers.cpp):
//
//   mountOptimizerRace(host, D)   the valley L = 0.5 w1² + 50 w2² as contour lines (from the recorded loss grid) and the paths of
//                                 optim::SGD, optim::SGD with momentum and optim::Adam; a slider picks the step
//
// The module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';
import { createPlot, srTable } from './plot.js';
import { contourSegments } from './contour2d.js';

export const LEVELS = [0.25, 1, 4, 16];
// id, label, line class, marker shape (shape and line style, not only colour, tell the runs apart)
export const RUNS = [
  ['sgd', 'SGD', 'or-sgd', 'circle', 'dotted line and circle: SGD'],
  ['momentum', 'SGD with momentum', 'or-mom', 'square', 'dashed line and square: SGD with momentum'],
  ['adam', 'Adam', 'or-adam', 'diamond', 'solid line and diamond: Adam'],
];

const CSS = `
.or{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.or-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .5rem}
.or-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.or-field{display:flex;gap:.5rem;align-items:center}
.or-field input[type=range]{width:12rem;accent-color:var(--fwd)}
.or-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.2rem 0 0}
.or-read{display:grid;grid-template-columns:auto auto auto auto;gap:.1rem 1rem;font:.85rem ui-monospace,monospace;margin:.5rem 0 0;width:max-content;max-width:100%}
.or-read b{color:var(--muted);font-weight:600}
.or-level{stroke:var(--border);stroke-width:1.2;fill:none}
.or-sgd{stroke:var(--ok);stroke-width:2;stroke-dasharray:2 4;stroke-linecap:round;fill:none}
.or-mom{stroke:var(--warn);stroke-width:2;stroke-dasharray:7 4;fill:none}
.or-adam{stroke:var(--fwd);stroke-width:2.4;fill:none}
.or-m-sgd{fill:var(--ok);stroke:var(--bg);stroke-width:1.5}
.or-m-mom{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.or-m-adam{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.or-min{fill:none;stroke:var(--fg);stroke-width:1.5}
@media(max-width:560px){.or-read{font-size:.8rem}}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('or-styles')) return;
  document.head.append(el('style', { id: 'or-styles' }, CSS));
}

function marker(p, shape, x, y, cls, layer) {
  const px = p.sx(x), py = p.sy(y), r = 6;
  const g = p.layer(layer);
  if (shape === 'circle') g.append(svg('circle', { class: cls, cx: px, cy: py, r }));
  else if (shape === 'square') g.append(svg('rect', { class: cls, x: px - r, y: py - r, width: 2 * r, height: 2 * r }));
  else g.append(svg('path', { class: cls, d: `M${px},${py - r - 1.5} L${px + r + 1.5},${py} L${px},${py + r + 1.5} L${px - r - 1.5},${py} Z` }));
}

// ======================================================================================================================
export function mountOptimizerRace(host, D) {
  injectStyles();
  const V = D.files.optimizers.valley, S = V.surface, last = V.steps;
  const run = (id) => V.runs.find((r) => r.id === id);
  let k = 10;
  const input = el('input', { type: 'range', id: 'or-step', min: 0, max: last, step: 1, value: k });
  input.addEventListener('input', () => { k = Number(input.value); mark(); });
  const box = el('div');
  const key = el('div', { class: 'or-key', 'aria-hidden': 'true' }, ...RUNS.map((r) => el('span', {}, r[4])), el('span', {}, 'thin lines: equal loss'));
  const read = el('div', { class: 'or-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'or' },
    el('div', { class: 'or-ctl' }, el('div', { class: 'or-field' }, el('label', { for: 'or-step', class: 'or-lbl' }, 'after step'), input)), box, key, read));
  srTable(host, { columns: ['step', ...RUNS.map((r) => `loss, ${r[1]}`)], rows: run('sgd').path.map((_, i) => [i, ...RUNS.map((r) => run(r[0]).path[i].loss)]), caption: 'Loss after each step' });
  const grid = { xs: S.w1, ys: S.w2, z: S.loss };
  const contours = LEVELS.map((lv) => contourSegments(grid, lv));

  const paint = (pl) => {
    pl.axes({ xTicks: [-4, -3, -2, -1, 0, 1], yTicks: [-0.5, -0.25, 0, 0.25, 0.5] });
    for (const segs of contours) for (const [[x1, y1], [x2, y2]] of segs) pl.layer('levels').append(svg('line', { class: 'or-level', x1: pl.sx(x1), y1: pl.sy(y1), x2: pl.sx(x2), y2: pl.sy(y2) }));
    pl.layer('levels').append(svg('circle', { class: 'or-min', cx: pl.sx(0), cy: pl.sy(0), r: 4 }));
    pl.layer('paths');
    pl.layer('live');
    mark();
  };
  const p = createPlot(box, { width: 640, height: 300, x: [S.w1[0], S.w1.at(-1)], y: [S.w2[0], S.w2.at(-1)], xLabel: 'w1 (flat direction)', yLabel: 'w2 (steep)', ariaLabel: '', onResize: paint });

  function mark() {
    p.clear('paths'); p.clear('live');
    for (const [id, , cls, shape] of RUNS) {
      const pts = run(id).path.slice(0, k + 1).map((q) => [q.w1, q.w2]);
      if (pts.length > 1) p.curve(pts, { class: cls, layer: 'paths' });
      const q = run(id).path[k];
      marker(p, shape, q.w1, q.w2, `or-m-${cls.slice(3)}`, 'live');
    }
    input.setAttribute('aria-valuetext', `after step ${k}`);
    p.root.setAttribute('aria-label', `The paths of three optimizers in a narrow valley, the lowest point at w1 = 0, w2 = 0. After step ${k}: ${RUNS.map(([id, name]) => { const q = run(id).path[k]; return `${name} at w1 ${fmt(q.w1)}, w2 ${fmt(q.w2)}, loss ${fmt(q.loss)}`; }).join('. ')}.`);
    read.replaceChildren(el('b', {}, 'optimizer'), el('b', {}, 'w1'), el('b', {}, 'w2'), el('b', {}, 'loss'),
      ...RUNS.flatMap(([id, name]) => { const q = run(id).path[k]; return [el('span', {}, name), el('span', {}, fmt(q.w1)), el('span', {}, fmt(q.w2)), el('span', {}, fmt(q.loss))]; }));
  }
  paint(p);
  return { mark };
}
