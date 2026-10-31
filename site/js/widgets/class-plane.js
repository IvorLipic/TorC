// widgets/class-plane.js — the 2D classification playground of the lesson classification (brief §6, §9; decision 21).
// Orange points fill a disc, blue points form a ring around it (data/classification.json, tools/trace/trace_classification.cpp).
// The learner chooses a model (0, 2, 4 or 8 hidden neurons) and a checkpoint of its training (before training, 25, 100 and 1,000
// steps). The plane shows what TorC recorded for that model and checkpoint:
//   - the shading is the probability of orange on the recorded 41 × 41 grid (`prob1`, softmax of the two scores);
//   - the white line is the decision boundary, the 50 % level of that grid (marching squares, drawn here from the recorded grid);
//     with 0 hidden neurons it lies on the recorded decision line (`derived.decision_line`, which the tests compare with it);
//   - each hidden neuron's line w1·x + w2·y + b = 0 (`units[]` in the trace), with a short arrow to the side where the neuron is
//     on (where ReLU passes its value on);
//   - the readout is the mistakes per 100 training points, with the same rounding as T.human in core/numbers.js.
// The words follow the lesson: "neuron" for a hidden unit, "probability" for softmax output (docs/authoring.md).
// Only training points and training numbers are shown: the comparison of models uses the training points (brief C2). Nothing is
// trained here (precomputed presets, decision 21). site/test/classification.test.mjs checks the pure helpers against the data.
// The DOM is touched only in mountClassPlane; pointsFigure() returns an SVG string for the lesson text.
import { el, svg } from '../core/dom.js';
import { humanValue, groupThousands } from '../core/numbers.js';
import { createPlot } from './plot.js';

export const ORANGE = '#f0883e';
export const BLUE = '#58a6ff';
const BG = [14, 17, 22];
const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

// ---- pure helpers -------------------------------------------------------------------------------------------------------
export const modelFor = (files, hidden) => {
  const m = files.models.find((x) => x.hidden === hidden);
  if (!m) throw new Error(`no recorded model with ${hidden} hidden units`);
  return m;
};
export const checkpointFor = (files, hidden, step) => {
  const c = modelFor(files, hidden).checkpoints.find((x) => x.step === step);
  if (!c) throw new Error(`no checkpoint at step ${step} for ${hidden} hidden units`);
  return c;
};

// The points of a set ('train' or 'test') as [{x, y, label}] (label 1 = orange disc, 0 = blue ring).
export function pointsOf(files, set = 'train') {
  const xy = files.data[`${set}_xy`], lab = files.data[`${set}_label`];
  return lab.map((label, i) => ({ x: xy[2 * i], y: xy[2 * i + 1], label }));
}

// The recorded probability of orange as a grid {xs, ys, z[iy][ix]} (prob1[j * n + i] is the point (axis[i], axis[j])).
export function gridOf(files, cp) {
  const { axis, n } = files.grid;
  const z = [];
  for (let j = 0; j < n; j++) z.push(cp.prob1.slice(j * n, (j + 1) * n));
  return { xs: axis, ys: axis, z };
}

// The level line of a grid (marching squares with straight-line interpolation along the cell edges): [[x1, y1, x2, y2], …].
// A saddle cell (two crossings on each diagonal pair) is split by the mean of its four corners.
export function levelSegments(grid, level = 0.5) {
  const { xs, ys, z } = grid, out = [];
  const lerp = (a, b, va, vb) => a + ((level - va) / (vb - va)) * (b - a);
  for (let j = 0; j + 1 < ys.length; j++) for (let i = 0; i + 1 < xs.length; i++) {
    const v = [z[j][i], z[j][i + 1], z[j + 1][i + 1], z[j + 1][i]];        // corners: bottom-left, bottom-right, top-right, top-left
    const c = [[xs[i], ys[j]], [xs[i + 1], ys[j]], [xs[i + 1], ys[j + 1]], [xs[i], ys[j + 1]]];
    const pts = [];
    for (let e = 0; e < 4; e++) {
      const a = e, b = (e + 1) % 4;
      if ((v[a] >= level) !== (v[b] >= level)) pts.push([lerp(c[a][0], c[b][0], v[a], v[b]), lerp(c[a][1], c[b][1], v[a], v[b]), e]);
    }
    if (pts.length === 2) out.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
    else if (pts.length === 4) {
      const centre = (v[0] + v[1] + v[2] + v[3]) / 4 >= level;
      const pair = (centre === (v[0] >= level)) ? [[0, 3], [1, 2]] : [[0, 1], [2, 3]];
      for (const [a, b] of pair) out.push([pts[a][0], pts[a][1], pts[b][0], pts[b][1]]);
    }
  }
  return out;
}

// The part of the line w1·x + w2·y + b = 0 inside the square [lo, hi]², as [[x, y], [x, y]], or null when it misses the square.
export function clipLine(w1, w2, b, lo, hi) {
  const hits = [];
  const add = (x, y) => { if (x >= lo - 1e-9 && x <= hi + 1e-9 && y >= lo - 1e-9 && y <= hi + 1e-9 && !hits.some((h) => Math.hypot(h[0] - x, h[1] - y) < 1e-9)) hits.push([x, y]); };
  if (Math.abs(w2) > 1e-12) for (const x of [lo, hi]) add(x, -(w1 * x + b) / w2);
  if (Math.abs(w1) > 1e-12) for (const y of [lo, hi]) add(-(w2 * y + b) / w1, y);
  if (hits.length < 2) return null;
  let best = [hits[0], hits[1]], d = -1;   // a line through a corner can hit three edges: keep the two points farthest apart
  for (let a = 0; a < hits.length; a++) for (let c = a + 1; c < hits.length; c++) {
    const dd = Math.hypot(hits[a][0] - hits[c][0], hits[a][1] - hits[c][1]);
    if (dd > d) { d = dd; best = [hits[a], hits[c]]; }
  }
  return best;
}

// The lines of a checkpoint: one per hidden neuron ({unit 1..H, seg, normal toward the side where the neuron is on}), or, with
// no hidden layer, the one recorded decision line (score of orange = score of blue), with `normal` toward orange. The widget
// draws the neuron lines; the decision line is the boundary itself, so the widget does not draw it again (the tests use it).
export function unitLines(files, cp) {
  const { lo, hi } = files.grid;
  const list = cp.units ? cp.units.map((u) => ({ unit: u.unit + 1, w1: u.w1, w2: u.w2, b: u.b, active: u.active_train_points }))
    : [{ unit: 0, ...cp.derived.decision_line }];
  return list.map((u) => {
    const n = Math.hypot(u.w1, u.w2);
    return { ...u, seg: clipLine(u.w1, u.w2, u.b, lo, hi), normal: n > 0 ? [u.w1 / n, u.w2 / n] : [0, 0] };
  });
}

// The readout, rounded the way T.human rounds it (core/numbers.js), so the widget and the prose never disagree.
export function readout(files, cp) {
  return { mistakes: humanValue('per100', cp.train_correct, files.setup.train_samples) };
}

export const stepLabel = (step) => (step === 0 ? 'before training' : `${groupThousands(String(step))} steps`);

// A static picture of the training points for the lesson text (an SVG string, no DOM).
export function pointsFigure(files, { size = 240, caption = '' } = {}) {
  const { lo, hi } = files.grid, pad = 8;
  const s = (v) => pad + ((v - lo) / (hi - lo)) * (size - 2 * pad);
  const t = (v) => size - s(v);
  const marks = pointsOf(files, 'train').map((p) => (p.label === 1
    ? `<circle cx="${s(p.x).toFixed(1)}" cy="${t(p.y).toFixed(1)}" r="3.2" fill="${ORANGE}"/>`
    : `<rect x="${(s(p.x) - 2.8).toFixed(1)}" y="${(t(p.y) - 2.8).toFixed(1)}" width="5.6" height="5.6" fill="${BLUE}"/>`)).join('');
  const frame = `<rect x="${pad}" y="${pad}" width="${size - 2 * pad}" height="${size - 2 * pad}" fill="none" stroke="#2d3541"/>`;
  return `<figure class="cp-fig"><svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" role="img" aria-label="${files.setup.train_samples} points in a square. Orange points fill a disc in the middle. Blue points form a ring around the disc." focusable="false">${frame}${marks}</svg>${caption ? `<figcaption>${caption}</figcaption>` : ''}</figure>`;
}

// the shading: the probability of orange, from deep blue (0) through the background (0.5) to deep orange (1)
export function shade(p) {
  const t = Math.max(-1, Math.min(1, (p - 0.5) * 2)), k = Math.abs(t) * 0.62;
  const c = rgb(t >= 0 ? ORANGE : BLUE);
  return `rgb(${BG.map((b, i) => Math.round(b + (c[i] - b) * k)).join(',')})`;
}

// ---- the widget ---------------------------------------------------------------------------------------------------------
let seq = 0;
export function mountClassPlane(host, D, { hidden: h0 = 0, step: s0 = null } = {}) {
  injectStyles();
  const files = D.files.classification;
  const HIDDEN = files.setup.hidden, STEPS = files.setup.checkpoints;
  const id = `cp${++seq}`;
  let hidden = h0, step = s0 ?? STEPS[STEPS.length - 1], showLines = true;
  const pts = pointsOf(files, 'train');

  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const hBtns = HIDDEN.map((h) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { hidden = h; draw(); announce(); } }, String(h)));
  const sBtns = STEPS.map((s) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => { step = s; draw(); announce(); } }, stepLabel(s)));
  const box = el('input', { type: 'checkbox', id: `${id}-lines`, checked: true });
  box.addEventListener('change', () => { showLines = box.checked; draw(); announce(); });
  const controls = el('div', { class: 'cp-ctl' },
    el('div', { class: 'cp-row', role: 'group', 'aria-labelledby': `${id}-h` }, el('span', { class: 'cp-lbl', id: `${id}-h` }, 'hidden neurons'), ...hBtns),
    el('div', { class: 'cp-row', role: 'group', 'aria-labelledby': `${id}-s` }, el('span', { class: 'cp-lbl', id: `${id}-s` }, 'training'), ...sBtns),
    el('div', { class: 'cp-row' }, box, el('label', { for: `${id}-lines`, class: 'cp-check' }, 'show the line of each hidden neuron')));
  const plotBox = el('div', { class: 'cp-plot' });
  const read = el('div', { class: 'cp-read' });
  const hint = el('p', { class: 'hint cp-hint' }, 'Orange shading means that the model says orange, and blue shading means blue. A deeper colour means that the model is more sure. The white line is the boundary, where the model is undecided. Circles are orange points, and squares are blue points.');
  host.replaceChildren(el('div', { class: 'cp' }, controls, plotBox, read, hint, status));
  const { lo, hi } = files.grid;
  const p = createPlot(plotBox, { width: 440, height: 440, x: [lo, hi], y: [lo, hi], xLabel: 'x', yLabel: 'y', margin: { l: 40, r: 12, t: 10, b: 36 }, ariaLabel: '', onResize: () => paint() });

  function paint() {
    const cp = checkpointFor(files, hidden, step);
    for (const n of ['heat', 'boundary', 'units', 'points']) p.clear(n);
    p.root.querySelectorAll('.plot-axes').forEach((g) => g.remove());
    p.axes({ xTicks: [-2, -1, 0, 1, 2], yTicks: [-2, -1, 0, 1, 2] });
    p.heat(gridOf(files, cp), { color: shade, layer: 'heat' });
    const segs = levelSegments(gridOf(files, cp));
    p.layer('boundary').append(svg('path', { class: 'cp-boundary', d: segs.map(([a, b, c, d]) => `M${p.sx(a).toFixed(1)},${p.sy(b).toFixed(1)}L${p.sx(c).toFixed(1)},${p.sy(d).toFixed(1)}`).join('') }));
    const lines = hidden > 0 && showLines ? unitLines(files, cp) : [];
    const ul = p.layer('units');
    for (const u of lines) {
      if (!u.seg) continue;
      const [[x1, y1], [x2, y2]] = u.seg;
      ul.append(svg('line', { class: 'cp-unit', x1: p.sx(x1), y1: p.sy(y1), x2: p.sx(x2), y2: p.sy(y2) }));
      // the arrow and the number sit near the start of the visible part of the line, away from the boundary in the middle
      const mx = x1 + (x2 - x1) * 0.1, my = y1 + (y2 - y1) * 0.1;
      const ax = p.sx(mx), ay = p.sy(my), dx = u.normal[0], dy = -u.normal[1];
      ul.append(svg('line', { class: 'cp-unit-arrow', x1: ax, y1: ay, x2: ax + dx * 16, y2: ay + dy * 16, 'marker-end': `url(#${id}-arrow)` }));
      ul.append(svg('text', { class: 'cp-unit-num', x: ax - dx * 11, y: ay - dy * 11 + 4, 'text-anchor': 'middle' }, String(u.unit)));
    }
    const pl = p.layer('points');
    for (const q of pts) {
      if (q.label === 1) pl.append(svg('circle', { class: 'cp-o', cx: p.sx(q.x), cy: p.sy(q.y), r: 3.2 }));
      else pl.append(svg('rect', { class: 'cp-b', x: p.sx(q.x) - 2.8, y: p.sy(q.y) - 2.8, width: 5.6, height: 5.6 }));
    }
    ensureMarker();
    return cp;
  }
  function ensureMarker() {
    if (p.root.querySelector(`#${id}-arrow`)) return;
    p.root.prepend(svg('defs', {}, svg('marker', { id: `${id}-arrow`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, svg('path', { d: 'M0,0 L10,5 L0,10 z', class: 'cp-arrow-head' }))));
  }
  function describe(cp) {
    const r = readout(files, cp);
    const n = hidden === 0 ? 'no hidden neurons' : `${hidden} hidden neurons`;
    return `${n}, ${stepLabel(step)}. ${r.mistakes} mistakes per 100 training points.`;
  }
  function draw() {
    hBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(HIDDEN[i] === hidden)));
    sBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(STEPS[i] === step)));
    box.disabled = hidden === 0;
    const cp = paint();
    const r = readout(files, cp);
    read.innerHTML = `<span><b>mistakes on the training points</b> ${r.mistakes} per 100</span>`;
    p.root.setAttribute('aria-label', `The plane with the training points and the boundary of the model. ${describe(cp)}`);
  }
  function announce() { status.textContent = describe(checkpointFor(files, hidden, step)); }
  draw();
  return { set: (h, s) => { hidden = h; if (s !== undefined) step = s; draw(); announce(); } };
}

const CSS = `
.cp{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.cp-ctl{display:flex;flex-direction:column;gap:.45rem;margin:0 0 .5rem}
.cp-row{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.cp-lbl{color:var(--muted);font-size:.9rem;min-width:6.5rem}
.cp-check{font-size:.92rem;color:var(--fg)}
.cp input[type=checkbox]{width:1.1rem;height:1.1rem;accent-color:var(--fwd)}
.cp .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.cp-plot{max-width:30rem}
.cp-boundary{stroke:#fff;stroke-width:2.6;fill:none;stroke-linecap:round}
.cp-unit{stroke:#c9b6ff;stroke-width:1.6;stroke-dasharray:6 4}
.cp-unit-arrow{stroke:#c9b6ff;stroke-width:1.6}
.cp-arrow-head{fill:#c9b6ff}
.cp-unit-num{fill:#c9b6ff;font:600 12px system-ui,sans-serif;paint-order:stroke;stroke:#0e1116;stroke-width:3px}
.cp-o{fill:${ORANGE};stroke:#0e1116;stroke-width:.6}
.cp-b{fill:${BLUE};stroke:#0e1116;stroke-width:.6}
.cp-read{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;font-family:ui-monospace,monospace;font-size:.92rem;margin:.4rem 0 0}
.cp-read b{color:var(--muted);font-weight:600;margin-right:.3rem;font-family:system-ui,sans-serif}
.cp-hint{font-size:.88rem;margin:.5rem 0 0}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('cp-styles')) return;
  document.head.append(el('style', { id: 'cp-styles' }, CSS));
}
