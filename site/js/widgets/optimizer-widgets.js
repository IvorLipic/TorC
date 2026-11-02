// widgets/optimizer-widgets.js — the race of the lesson optimizers. It shows only what TorC recorded (data/optimizers.json,
// tools/trace/trace_optimizers.cpp): each optimizer at its own best learning rate from the sweep (`race`, `bowl.race`, brief C4).
//
//   mountOptimizerRace(host, D, { ids, problems })   a map of the loss (contour lines from the recorded loss grid) with the path of
//        each optimizer up to the chosen step, and a plot of the loss against the step (log axis), each curve labelled on the plot
//        (brief D18, no table). `ids` picks the optimizers (default all four), `problems` the problems ('valley', 'bowl').
//
// Pure helpers (exported for the tests): raceOf, LEVELS, logLoss, FLOOR. The module touches the DOM only in mount*.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable } from './plot.js';
import { contourSegments } from './contour2d.js';
import { spreadLabels } from './curve-labels.js';

// contour levels of the two problems (the valley starts at loss 16, the bowl at 32.3)
export const LEVELS = { valley: [0.25, 1, 4, 16], bowl: [1, 4, 16, 32] };
// the loss axis is a log axis from FLOOR to 100; a loss below FLOOR (the bowl's plain SGD reaches 0) is drawn on the floor
export const FLOOR = 1e-4;
export const logLoss = (v) => Math.log10(Math.max(v, FLOOR));
// id, short name, line class, marker shape (shape and line style, not only colour, tell the runs apart)
export const RUNS = [
  ['sgd', 'plain SGD', 'or-sgd', 'circle'],
  ['momentum', 'momentum', 'or-mom', 'square'],
  ['adam', 'Adam', 'or-adam', 'diamond'],
  ['adamw', 'AdamW', 'or-adamw', 'triangle'],
];
const PROBLEM = { valley: 'Narrow valley', bowl: 'Round bowl' };
// the recorded loss grid that each map draws (the valley's race map uses the wider `race_surface`)
export const surfaceOf = (O, problem) => (problem === 'bowl' ? O.bowl.surface : O.race_surface);
// the race entries of one problem, in RUNS order
export function raceOf(O, problem, ids = RUNS.map((r) => r[0])) {
  const race = problem === 'bowl' ? O.bowl.race : O.race;
  return RUNS.filter((r) => ids.includes(r[0])).map(([id, name, cls, shape]) => ({ id, name, cls, shape, run: race.find((r) => r.id === id) }));
}
const num = (v) => { const s = String(Number(v.toPrecision(2))); return s.startsWith('-') ? `−${s.slice(1)}` : s; };
const lossText = (v) => (v < FLOOR ? 'below 0.0001' : num(v));

const CSS = `
.or{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.or-ctl{display:flex;flex-wrap:wrap;gap:.5rem 1.4rem;align-items:center;margin:0 0 .5rem}
.or-presets{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.or-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.or .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.or-field{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.or-field input[type=range]{width:12rem;max-width:60vw;accent-color:var(--fwd)}
.or-plots{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
@media(max-width:760px){.or-plots{grid-template-columns:1fr}}
.or-cap{font-size:.85rem;color:var(--muted);margin:.2rem 0 0}
.or-read{font-size:.92rem;margin:.5rem 0 0}
.or-level{stroke:var(--border);stroke-width:1.2;fill:none}
.or-sgd{stroke:var(--ok);stroke-width:2.2;stroke-dasharray:2 4;stroke-linecap:round;fill:none}
.or-mom{stroke:var(--warn);stroke-width:2.2;stroke-dasharray:7 4;fill:none}
.or-adam{stroke:var(--fwd);stroke-width:2.4;fill:none}
.or-adamw{stroke:var(--bwd);stroke-width:1.8;stroke-dasharray:1 0;fill:none;opacity:.85}
.or-m-sgd,.or-t-sgd{fill:var(--ok)}
.or-m-mom,.or-t-mom{fill:var(--warn)}
.or-m-adam,.or-t-adam{fill:var(--fwd)}
.or-m-adamw,.or-t-adamw{fill:var(--bwd)}
.or-mk{stroke:var(--bg);stroke-width:1.5}
.or-t{font-size:12px;font-weight:600}
.or-min{fill:none;stroke:var(--fg);stroke-width:1.5}
.or-mark{stroke:var(--fg);stroke-width:1;stroke-dasharray:3 3;opacity:.6}
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
  else if (shape === 'diamond') g.append(svg('path', { class: cls, d: `M${px},${py - r - 1.5} L${px + r + 1.5},${py} L${px},${py + r + 1.5} L${px - r - 1.5},${py} Z` }));
  else g.append(svg('path', { class: cls, d: `M${px},${py - r - 1} L${px + r + 1},${py + r} L${px - r - 1},${py + r} Z` }));
}

let seq = 0;
export function mountOptimizerRace(host, D, { ids = RUNS.map((r) => r[0]), problems = ['valley', 'bowl'] } = {}) {
  injectStyles();
  const O = D.files.optimizers, last = O.valley.steps, uid = `or${++seq}`;
  let problem = problems[0], k = 0, map = null, lossPlot = null, labelG = null;
  const pick = problems.length > 1 ? (() => {
    const btns = problems.map((pr) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { problem = pr; k = 0; slider.value = 0; build(); mark(`${PROBLEM[pr]}. Back to step 0.`); } }, PROBLEM[pr]));
    return { root: el('div', { class: 'or-presets', role: 'group', 'aria-label': 'Problem' }, el('span', { class: 'or-lbl', 'aria-hidden': 'true' }, 'Problem'), ...btns), set: () => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(problems[i] === problem))) };
  })() : null;
  const slider = el('input', { type: 'range', id: `${uid}-step`, min: 0, max: last, step: 1, value: k });
  const label = el('label', { for: `${uid}-step`, class: 'or-lbl' }, 'after step 0');
  slider.addEventListener('input', () => { k = Number(slider.value); mark(); });
  const mapBox = el('div'), lossBox = el('div');
  const read = el('p', { class: 'or-read', 'aria-live': 'polite' });
  const rates = el('p', { class: 'or-cap' });
  host.replaceChildren(el('div', { class: 'or' },
    el('div', { class: 'or-ctl' }, ...(pick ? [pick.root] : []), el('div', { class: 'or-field' }, label, slider)),
    el('div', { class: 'or-plots' },
      el('div', {}, mapBox, el('p', { class: 'or-cap' }, 'The map. Each thin line joins points of equal loss. The ring is the lowest point.')),
      el('div', {}, lossBox, el('p', { class: 'or-cap' }, 'The loss after each step. Each grid line up is a hundred times higher.'))),
    read, rates));
  const sr = srTable(host, { columns: ['step', ...raceOf(O, 'valley', ids).map((r) => `loss, ${r.name}`)], rows: [], caption: 'Loss after each step' });

  function build() {
    pick?.set();
    map?.destroy(); lossPlot?.destroy();
    map = null; lossPlot = null; labelG?.remove(); labelG = null;
    mapBox.replaceChildren(); lossBox.replaceChildren();
    const S = surfaceOf(O, problem);
    const grid = { xs: S.w1, ys: S.w2, z: S.loss };
    const contours = LEVELS[problem].map((lv) => contourSegments(grid, lv));
    const paintMap = (pl) => {
      pl.axes({ yTicks: problem === 'bowl' ? [-2, 0, 2] : [-0.5, 0, 0.5], xTicks: problem === 'bowl' ? [-4, -2, 0, 2] : [-4, -3, -2, -1, 0, 1, 2], fmt: num });
      for (const segs of contours) for (const [[x1, y1], [x2, y2]] of segs) pl.layer('levels').append(svg('line', { class: 'or-level', x1: pl.sx(x1), y1: pl.sy(y1), x2: pl.sx(x2), y2: pl.sy(y2) }));
      pl.layer('levels').append(svg('circle', { class: 'or-min', cx: pl.sx(0), cy: pl.sy(0), r: 5 }));
      pl.layer('paths'); pl.layer('live');
      if (lossPlot && labelG) mark();
    };
    map = createPlot(mapBox, { width: 420, height: 300, x: [S.w1[0], S.w1.at(-1)], y: [S.w2[0], S.w2.at(-1)], xLabel: problem === 'bowl' ? 'w₁' : 'w₁ (flat in the valley)', yLabel: 'w₂', ariaLabel: '', onResize: paintMap, margin: { l: 48, r: 10 } });
    const paintLoss = (pl) => {
      pl.axes({ xTicks: [0, 10, 20, 30, 40, 50], yTicks: [-4, -2, 0, 2], fmt: (t) => String(t), yFmt: (t) => (t < 0 ? `0.${'0'.repeat(-t - 1)}1` : String(10 ** t)) });
      pl.layer('curves'); pl.layer('live');
      labelG?.remove();
      labelG = svg('g', { class: 'or-labels' });   // not a layer: layers are clipped to the plot area, and the labels use the margin
      pl.root.append(labelG);
      mark();
    };
    lossPlot = createPlot(lossBox, { width: 420, height: 300, x: [0, last], y: [Math.log10(FLOOR), 2], xLabel: 'step', yLabel: 'loss', ariaLabel: '', onResize: paintLoss, margin: { l: 56, r: 78 } });
    paintMap(map);
    paintLoss(lossPlot);
    const R = raceOf(O, problem, ids);
    sr.update(R[0].run.path.map((_, i) => [i, ...R.map((r) => r.run.path[i].loss)]));
    rates.textContent = `Each optimizer runs at its own best learning rate here: ${R.map((r) => `${r.name} ${num(r.run.lr)}`).join(', ').replace(/, ([^,]*)$/, ' and $1')}.`;
  }

  function mark(msg) {
    const R = raceOf(O, problem, ids);
    map.clear('paths'); map.clear('live');
    lossPlot.clear('curves'); lossPlot.clear('live'); labelG.replaceChildren();
    for (const r of R) {
      const pts = r.run.path.slice(0, k + 1);
      if (pts.length > 1) {
        map.curve(pts.map((q) => [q.w1, q.w2]), { class: r.cls, layer: 'paths' });
        lossPlot.curve(pts.map((q, i) => [i, logLoss(q.loss)]), { class: r.cls, layer: 'curves' });
      }
      const q = r.run.path[k];
      marker(map, r.shape, q.w1, q.w2, `or-mk or-m-${r.cls.slice(3)}`, 'live');
      marker(lossPlot, r.shape, k, logLoss(q.loss), `or-mk or-m-${r.cls.slice(3)}`, 'live');
    }
    // labels: on the loss plot, right of the current points (the right margin holds them); spread so they never collide
    const want = R.map((r) => lossPlot.sy(logLoss(r.run.path[k].loss)) + 4);
    const area = lossPlot.plotArea;
    const ys = spreadLabels(want, { gap: 15, min: area.t + 10, max: area.t + area.h });
    R.forEach((r, i) => labelG.append(svg('text', { class: `or-t or-t-${r.cls.slice(3)}`, x: lossPlot.sx(k) + 10, y: ys[i] }, r.name)));
    lossPlot.line(k, Math.log10(FLOOR), k, 2, { class: 'or-mark', layer: 'live' });
    label.textContent = `after step ${k}`;
    slider.setAttribute('aria-valuetext', `after step ${k}`);
    const text = `After step ${k}, the loss is ${R.map((r) => `${lossText(r.run.path[k].loss)} for ${r.name}`).join(', ').replace(/, ([^,]*)$/, ' and $1')}. Lower is better.`;
    read.textContent = `${msg ? `${msg} ` : ''}${text}`;
    map.root.setAttribute('aria-label', `${PROBLEM[problem]}: the paths of the optimizers on a map of the loss, the lowest point at w₁ = 0, w₂ = 0. ${text}`);
    lossPlot.root.setAttribute('aria-label', `${PROBLEM[problem]}: the loss after each step. ${text}`);
  }
  build();
  mark();
  return { mark: () => mark() };
}
