// widgets/landscape1d.js — the interactive "one weight, one score" widgets of the lesson why-gradients.
//
//   mountKnob(host, D)       a slider for w; live p, e, L; a dot on the L(w) curve
//   mountSlope(host, D)      the same plus the tangent, and a secant whose second point h you can slide toward zero
//   mountStepper(host, D)    "step" button: w <- w - lr*slope, with the tangent's promise next to what really happened
//   mountStepSize(host, D)   learning-rate presets, the live factor 1 - 8*lr, and an 8-step trajectory table
//
// Honesty: every live value is recomputed in the browser by the engine (engine.buildGraph / runBackward on trace scene `why`, the
// same code as the graph lab; see landscape-model.js). Each widget says so, and says when the weight sits on a point TorC recorded
// (data/why-gradients.json) and the browser agrees with that recording. Every plot has a "Show as table" toggle.
// No auto-play; sliders are native <input type=range> with labels and aria-valuetext; re-renders keep focus (data-key + keepFocus).
// The module defines functions only: importing it in Node touches no DOM.
import { el, svg, keepFocus } from '../core/dom.js';
import { math } from '../core/katex.js';
import { fmt } from '../ops/texutil.js';
import { knobTex, secantTex, updateTex, lastStepTex, factorTex, fmtN } from './landscape-tex.js';
import { createPlot, tableToggle, ticks } from './plot.js';
import { makeModel, sameNumber, recordedPoint, recordedTrajectory, recordedSecant } from './landscape-model.js';

export const H_STOPS = [2, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001];     // the secant's second point h (five of them are recorded by TorC)
export const JUMPS = [0, 1, 3, 6];                                     // quick positions for the knob (all recorded by TorC as probes)
export const MAX_STEPS = 40;                                           // the manual stepper stops here (numbers stay printable)

// ---- tiny shared helpers (also used by why-gears.js) -----------------------------------------------------------------
let counter = 0;
export const uid = (prefix = 'ls') => `${prefix}${++counter}`;
export const M = (latex) => math(latex, false);                        // inline KaTeX
export const eqLine = (latex) => el('div', { class: 'ls-eq', html: M(latex) });

export const BADGE = {
  match: (what) => ({ cls: 'ok', text: `✓ ${what}` }),
  browser: (what) => ({ cls: 'warn', text: `↻ ${what}` }),
};
export function badgeEl() {
  const b = el('span', { class: 'ls-badge ok' });
  b.set = ({ cls, text }) => { b.className = `ls-badge ${cls}`; b.textContent = text; };
  return b;
}

// keep ←/→ on this widget's controls from also stepping the graph lab further down the page (it listens on `document`)
export function isolateArrows(host) {
  host.addEventListener('keydown', (e) => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') e.stopPropagation(); });
}

export function slider({ key, label, min, max, step, value, onInput, show, valueText, hint }) {
  const id = uid('lsin');
  const input = el('input', { type: 'range', id, min, max, step, value, 'data-key': key });
  const out = el('output', { class: 'ls-val', for: id });
  const row = el('div', { class: 'ls-slider' },
    el('label', { for: id, html: label }), input, out, hint ? el('div', { class: 'ls-hint' }, hint) : null);
  const paint = () => {
    const v = Number(input.value);
    out.textContent = show(v);
    input.setAttribute('aria-valuetext', valueText(v));
  };
  input.addEventListener('input', () => { paint(); onInput(Number(input.value)); });
  paint();
  return { row, input, out, paint, set(v) { input.value = v; paint(); } };
}

function legend(items) {
  return el('div', { class: 'ls-legend', 'aria-hidden': 'true' }, items.map(([kind, text]) => {
    const sw = svg('svg', { class: 'ls-sw', viewBox: '0 0 26 12', width: 26, height: 12 });
    if (kind === 'curve') sw.append(svg('line', { x1: 1, x2: 25, y1: 6, y2: 6, class: 'plot-curve' }));
    if (kind === 'tangent') sw.append(svg('line', { x1: 1, x2: 25, y1: 6, y2: 6, class: 'plot-tangent' }));
    if (kind === 'secant') sw.append(svg('line', { x1: 3, x2: 23, y1: 6, y2: 6, class: 'ls-secant' }), svg('rect', { x: 0, y: 3, width: 6, height: 6, class: 'ls-sq' }), svg('rect', { x: 20, y: 3, width: 6, height: 6, class: 'ls-sq' }));
    if (kind === 'dot') sw.append(svg('circle', { cx: 13, cy: 6, r: 5, class: 'plot-dot' }));
    if (kind === 'step') sw.append(svg('circle', { cx: 13, cy: 6, r: 5, class: 'ls-pt' }));
    if (kind === 'promise') sw.append(svg('path', { d: 'M13,0.5 L19,6 L13,11.5 L7,6 Z', class: 'ls-diamond' }));
    if (kind === 'ref') sw.append(svg('line', { x1: 1, x2: 25, y1: 6, y2: 6, class: 'ls-ref' }));
    return el('span', {}, sw, ' ', text);
  }));
}

// the L(w) curve on [-1, 7], drawn by the engine at 161 weights; layer "live" is for whatever moves
const isNarrow = () => typeof window !== 'undefined' && window.matchMedia?.('(max-width: 640px)').matches;   // plots are drawn in a narrower box on phones so their text stays legible
function curvePlot(m, host, ariaLabel) {
  const narrow = isNarrow();
  const p = createPlot(host, { width: narrow ? 400 : 640, height: narrow ? 300 : 330, x: [-1, 7], y: [0, 72], xLabel: 'w  (the knob)', yLabel: 'L  (the score)', ariaLabel });
  p.axes({ xTicks: ticks([-1, 7], 8), yTicks: [0, 20, 40, 60] });
  p.curve(m.curve(-1, 7, 160).map((q) => [q.w, q.L]), { class: 'plot-curve', layer: 'curve' });
  p.layer('live');
  return p;
}
function curveTable(m, host) {
  const rows = Array.from({ length: 33 }, (_, i) => m.at(-1 + i * 0.25)).map((q) => [q.w, q.p, q.e, q.L, q.grad]);
  return tableToggle(host, { columns: ['w', 'p = w·x', 'e = p − t', 'L = e·e', 'slope dL/dw'], rows, caption: 'The curve above at w = −1, −0.75, …, 7 (computed by the same engine code as the graph lab)' });
}
const sqMarker = (p, x, y) => svg('rect', { x: p.sx(x) - 4.5, y: p.sy(y) - 4.5, width: 9, height: 9, class: 'ls-sq' });
const diamond = (p, x, y, r = 7) => svg('path', { d: `M${p.sx(x)},${p.sy(y) - r} L${p.sx(x) + r},${p.sy(y)} L${p.sx(x)},${p.sy(y) + r} L${p.sx(x) - r},${p.sy(y)} Z`, class: 'ls-diamond' });

function frame(host, name) {
  injectStyles();
  host.classList.add('ls-host');
  isolateArrows(host);
  const root = el('div', { class: `ls ls-${name}` });
  const plotBox = el('div', { class: 'ls-plotbox' });
  const side = el('div', { class: 'ls-side' });
  const foot = el('p', { class: 'ls-foot' });
  root.append(plotBox, side);
  host.replaceChildren(root, foot);
  return { root, plotBox, side, foot };
}
const FOOT = 'Values are recomputed in your browser by a small JavaScript port of TorC’s rules — the same code the graph lab below uses — and tested against the numbers the real library printed.';

// ======================================================================================================================
// 1. the knob
// ======================================================================================================================
export function mountKnob(host, D) {
  const files = D.files['why-gradients'], m = makeModel(D.scenes.why);
  const { plotBox, side, foot } = frame(host, 'knob');
  const key = uid('k');
  let w = m.w0, best = null;
  const p = curvePlot(m, plotBox, 'The score L as the knob w turns: a bowl-shaped curve.');
  plotBox.append(legend([['curve', 'L(w)'], ['dot', 'where the knob is now']]));
  curveTable(m, plotBox);

  const sl = slider({
    key: `${key}:w`, label: 'Turn the knob <code>w</code>', min: -1, max: 7, step: 0.05, value: w,
    show: (v) => `w = ${fmt(v)}`, valueText: (v) => { const q = m.at(v); return `w = ${fmt(v)}, loss ${fmt(q.L)}`; },
    onInput: (v) => { w = v; update(); },
  });
  const jumps = el('div', { class: 'ls-jumps', role: 'group', 'aria-label': 'Jump the knob to a value' },
    JUMPS.map((v) => el('button', { type: 'button', class: 'btn small', 'data-key': `${key}:j${v}`, onclick: () => { w = v; sl.set(v); update(); } }, `w = ${v}`)));
  const eqs = el('div', { class: 'ls-eqs' });
  const score = el('div', { class: 'ls-score', 'aria-hidden': 'true' });
  const bestEl = el('div', { class: 'ls-hint' });
  const badge = badgeEl();
  side.append(el('div', { class: 'ls-panel' }, sl.row, jumps), el('div', { class: 'ls-panel' }, eqs, score, bestEl), badge);
  foot.textContent = FOOT;

  function update() {
    keepFocus(() => {
      const q = m.at(w);
      eqs.replaceChildren(...knobTex(m, w).map(eqLine));
      score.textContent = `score  L = ${fmt(q.L)}`;
      if (!best || q.L < best.L) best = q;
      bestEl.textContent = `Lowest score you have reached: L = ${fmt(best.L)} at w = ${fmt(best.w)}.`;
      p.clear('live');
      p.line(w, 0, w, q.L, { class: 'ls-drop', layer: 'live' });
      p.dot(w, q.L, { layer: 'live', r: 7 });
      p.root.setAttribute('aria-label', `The score L as the knob w turns: a bowl-shaped curve. The knob is at w = ${fmt(w)}, where L = ${fmt(q.L)}.`);
      const rec = recordedPoint(files, w);
      badge.set(rec && sameNumber(q.L, rec.L) && sameNumber(q.p, rec.p) && sameNumber(q.e, rec.e)
        ? BADGE.match(`w = ${fmt(w)} is a point TorC recorded: its float32 run printed L = ${fmt(rec.L)}, the browser agrees`)
        : BADGE.browser('Recomputed in your browser (no TorC recording at exactly this w)'));
    }, host);
  }
  update();
  return { update };
}

// ======================================================================================================================
// 2. slope: tangent and secant
// ======================================================================================================================
export function mountSlope(host, D) {
  const files = D.files['why-gradients'], m = makeModel(D.scenes.why);
  const { plotBox, side, foot } = frame(host, 'slope');
  const key = uid('s');
  let w = m.w0, hi = H_STOPS.indexOf(1);
  const p = curvePlot(m, plotBox, 'The loss curve with a tangent line and a secant line.');
  plotBox.append(legend([['curve', 'L(w)'], ['dot', 'the knob'], ['tangent', 'tangent = the slope TorC’s backward() reports'], ['secant', 'secant through the knob and w + h']]));

  const slW = slider({
    key: `${key}:w`, label: 'Knob <code>w</code>', min: -1, max: 7, step: 0.05, value: w,
    show: (v) => `w = ${fmt(v)}`, valueText: (v) => `w = ${fmt(v)}, slope ${fmt(m.at(v).grad)}`,
    onInput: (v) => { w = v; update(); },
  });
  const slH = slider({
    key: `${key}:h`, label: 'Second point: <code>h</code> (how far we look)', min: 0, max: H_STOPS.length - 1, step: 1, value: hi,
    show: (i) => `h = ${fmt(H_STOPS[i])}`, valueText: (i) => `h = ${fmt(H_STOPS[i])}, secant slope ${fmt(m.secant(w, H_STOPS[i]).slope)}`,
    onInput: (i) => { hi = i; update(); },
    hint: 'Slide left to right: h shrinks toward 0.',
  });
  const paper = el('div', { class: 'ls-eqs' });
  const verdict = el('p', { class: 'ls-verdict' });
  const recorded = el('p', { class: 'ls-hint' });
  const badge = badgeEl();
  side.append(el('div', { class: 'ls-panel' }, slW.row, slH.row), el('div', { class: 'ls-panel' }, paper, verdict, recorded), badge);
  // table: the secants at the current knob for every h stop
  const tbl = tableToggle(plotBox, { columns: ['h', 'L(w+h)', 'secant slope', 'secant − slope'], rows: [], fmt: (v) => (typeof v === 'number' ? fmtN(v, 8) : String(v)), caption: 'The secant for every h at the current knob position (the last row is the limit TorC’s backward() reports); the curve itself is tabulated in the first widget' });
  foot.textContent = `${FOOT} The secant table and the picture show the same numbers.`;

  function update() {
    keepFocus(() => {
      const h = H_STOPS[hi], q = m.at(w), s = m.secant(w, h);
      paper.replaceChildren(...secantTex(m, w, h).map(eqLine));
      const dir = Math.abs(q.grad) < 1e-9 ? '→ flat here: the score does not change when w changes a little'
        : q.grad < 0 ? '↘ negative: turning w up lowers the score' : '↗ positive: turning w up raises the score';
      verdict.textContent = `The slope at w = ${fmt(w)} is ${fmtN(q.grad)} score units per unit of w, ${dir}. The secant says ${fmtN(s.slope)}; the gap is ${fmtN(s.gap, 6)}.`;
      const rec = Math.abs(w - m.w0) < 1e-9 ? recordedSecant(files, h) : null;
      recorded.textContent = rec ? `TorC’s own float32 run of this secant at w = ${fmt(w)} printed ${fmtN(rec.slope, 6)}${Math.abs(rec.slope - s.slope) > 5e-4 ? ` (the browser, in double precision: ${fmtN(s.slope, 6)} — float32 keeps about seven digits, and a tiny h subtracts two nearly equal numbers)` : ''}.` : '';
      p.clear('live');
      p.tangent(w, q.L, q.grad, { half: 1.6, layer: 'live' });
      const x1 = w - 0.5, x2 = w + h + 0.5;
      p.line(x1, q.L + s.slope * (x1 - w), x2, q.L + s.slope * (x2 - w), { class: 'ls-secant', layer: 'live' });
      p.layer('live').append(sqMarker(p, w + h, s.Lh));
      p.dot(w, q.L, { layer: 'live', r: 7 });
      p.root.setAttribute('aria-label', `The loss curve at w = ${fmt(w)}. Tangent slope ${fmt(q.grad)}; secant to w + ${fmt(h)} has slope ${fmt(s.slope)}.`);
      tbl.update([...H_STOPS.map((hh) => { const t = m.secant(w, hh); return [hh, t.Lh, t.slope, t.gap]; }), ['h → 0', q.L, q.grad, 0]]);
      const recW = recordedPoint(files, w);
      badge.set(recW && sameNumber(q.grad, recW.grad)
        ? BADGE.match(`w = ${fmt(w)} is a point TorC recorded: its backward() printed ${fmt(recW.grad)}, the browser agrees`)
        : BADGE.browser('Recomputed in your browser (no TorC recording at exactly this w)'));
    }, host);
  }
  update();
  return { update };
}

// ======================================================================================================================
// 3. step against the slope (manual)
// ======================================================================================================================
export function mountStepper(host, D) {
  const files = D.files['why-gradients'], m = makeModel(D.scenes.why);
  const { plotBox, side, foot } = frame(host, 'step');
  const key = uid('t');
  let lr = 0.05;
  let hist = [];                                      // completed steps: {k, w, L, grad, lr, wNext, promise, actual}
  const cur = () => (hist.length ? hist[hist.length - 1].wNext : m.w0);
  const p = curvePlot(m, plotBox, 'The loss curve with the path of the steps taken.');
  plotBox.append(legend([['curve', 'L(w)'], ['dot', 'where w is now'], ['step', 'earlier positions (numbered)'], ['tangent', 'tangent at the dot'], ['promise', 'what the tangent promised for the last step']]));
  curveTable(m, plotBox);

  const slLr = slider({
    key: `${key}:lr`, label: 'Step size <code>lr</code>', min: 0.01, max: 0.35, step: 0.005, value: lr,
    show: (v) => `lr = ${fmt(v)}`, valueText: (v) => `learning rate ${fmt(v)}`,
    onInput: (v) => { lr = v; update(); },
  });
  const stepBtn = el('button', { type: 'button', class: 'btn primary', 'data-key': `${key}:step`, onclick: () => { stepOnce(); } }, 'Step: w ← w − lr · slope');
  const resetBtn = el('button', { type: 'button', class: 'btn', 'data-key': `${key}:reset`, onclick: () => { hist = []; update(); } }, 'Reset to w = 0');
  const status = el('div', { class: 'sr', 'aria-live': 'polite' });
  const next = el('div', { class: 'ls-eqs' });
  const last = el('div', { class: 'ls-eqs' });
  const badge = badgeEl();
  const table = el('div', { class: 'ls-tablewrap', tabindex: 0, role: 'region', 'aria-label': 'Steps taken so far (scrollable)' });
  side.append(el('div', { class: 'ls-panel' }, slLr.row, el('div', { class: 'ls-buttons' }, stepBtn, resetBtn), status),
    el('div', { class: 'ls-panel' }, el('h4', {}, 'Next step'), next, el('h4', {}, 'Last step'), last), badge);
  host.querySelector('.ls').append(table);                 // after the controls, so on a phone the Step button comes before the table
  foot.textContent = FOOT;

  function stepOnce() {
    if (hist.length >= MAX_STEPS) return;
    const w = cur(), q = m.at(w), wNext = w - lr * q.grad;
    hist.push({ k: hist.length, w, L: q.L, grad: q.grad, lr, wNext, promise: q.L + q.grad * (wNext - w), actual: m.lossAt(wNext) });
    const h = hist[hist.length - 1];
    update();
    status.textContent = `Step ${hist.length}: w moved from ${fmt(h.w)} to ${fmt(h.wNext)}. The loss went from ${fmt(h.L)} to ${fmt(h.actual)}; the tangent had promised ${fmt(h.promise)}.`;
  }

  function update() {
    keepFocus(() => {
      const w = cur(), q = m.at(w), wNext = w - lr * q.grad;
      stepBtn.disabled = hist.length >= MAX_STEPS;
      resetBtn.disabled = hist.length === 0;
      next.replaceChildren(eqLine(updateTex(m, w, lr)));
      const l = hist[hist.length - 1];
      last.replaceChildren(...(l ? lastStepTex(l).map(eqLine) : [el('p', { class: 'ls-hint' }, 'Nothing yet — press Step.')]));
      // plot
      p.clear('live');
      const live = p.layer('live');
      if (hist.length) p.polyline([...hist.map((h) => [h.w, h.L]), [w, q.L]], { class: 'ls-path', layer: 'live' });
      p.tangent(w, q.L, q.grad, { half: 1.6, layer: 'live' });
      if (l) {
        p.line(l.wNext, l.promise, l.wNext, l.actual, { class: 'ls-drop', layer: 'live' });
        live.append(diamond(p, l.wNext, l.promise));
      }
      hist.forEach((h) => {
        live.append(svg('circle', { class: 'ls-pt', cx: p.sx(h.w), cy: p.sy(h.L), r: 5 }));
        p.text(h.w, h.L, String(h.k), { class: 'ls-ptlabel', dx: 9, dy: -8, layer: 'live' });
      });
      p.dot(w, q.L, { layer: 'live', r: 7 });
      p.text(w, q.L, String(hist.length), { class: 'ls-ptlabel', dx: 10, dy: -9, layer: 'live' });
      p.root.setAttribute('aria-label', `The loss curve; after ${hist.length} step${hist.length === 1 ? '' : 's'} w is ${fmt(w)} and L is ${fmt(q.L)}.`);
      // table
      const rows = hist.map((h) => el('tr', {}, [h.k, h.w, h.L, h.grad, h.lr, h.wNext].map((v) => el('td', {}, fmt(v)))));
      rows.push(el('tr', { class: 'now' }, [hist.length, w, q.L, q.grad, lr].map((v) => el('td', {}, fmt(v))), el('td', {}, `next: ${fmt(wNext)}`)));
      table.replaceChildren(el('table', { class: 'ls-table' }, el('caption', {}, 'Steps so far; the last row is where w is now'),
        el('thead', {}, el('tr', {}, ['step', 'w', 'L(w)', 'slope g', 'lr used', 'new w'].map((c) => el('th', { scope: 'col' }, c)))), el('tbody', {}, rows)));
      table.scrollTop = table.scrollHeight;
      // honesty badge: identical to TorC's recorded run only for an unchanged lr and a fresh start
      const rec = recordedTrajectory(files, lr);
      const same = rec && hist.length >= 1 && hist.length <= 8 && hist.every((h, i) => Math.abs(h.lr - lr) < 1e-9 && sameNumber(h.wNext, rec.states[i].w_next));
      badge.set(same ? BADGE.match(`These ${hist.length} step${hist.length === 1 ? '' : 's'} equal the run TorC recorded with its real optim::SGD at lr = ${fmt(lr)}`)
        : BADGE.browser('Recomputed in your browser (w ← w − lr · slope, with the slope from the same engine as the graph lab)'));
    }, host);
  }
  update();
  return { update };
}

// ======================================================================================================================
// 4. step size: presets, the factor 1 - 8*lr, 8 steps at once
// ======================================================================================================================
export const PRESET_LRS = [0.05, 0.1, 0.125, 0.25, 0.3];            // the five learning rates TorC recorded (checked in the tests)
const STEPS = 8;
const VERDICT_LABEL = { smooth: '✓ Converges', exact: '✓ Lands exactly', zigzag: '✓ Converges, zig-zagging', bounce: '⇄ Never settles', diverge: '✗ Diverges' };

export function mountStepSize(host, D) {
  const files = D.files['why-gradients'], m = makeModel(D.scenes.why);
  const { plotBox, side, foot } = frame(host, 'size');
  const key = uid('z');
  let lr = PRESET_LRS[0];
  const p = curvePlot(m, plotBox, 'The loss curve with the positions of eight steps.');
  const note = el('p', { class: 'ls-hint' });
  plotBox.append(legend([['curve', 'L(w)'], ['step', 'position after k steps (numbered)'], ['dot', 'start, k = 0']]), note);
  curveTable(m, plotBox);
  // second row of the grid: w against the step number, and the table
  const q2 = el('div', { class: 'ls-plotbox ls-second' });

  const presetRow = el('div', { class: 'ls-presets', role: 'group', 'aria-label': 'Learning-rate presets' });
  const presetBtns = PRESET_LRS.map((v) => el('button', { type: 'button', class: 'btn small', 'data-key': `${key}:p${v}`, 'aria-pressed': 'false', onclick: () => { lr = v; slLr.set(v); update(); } }, `lr = ${fmt(v)}`));
  presetRow.append(...presetBtns);
  const slLr = slider({
    key: `${key}:lr`, label: 'Step size <code>lr</code>', min: 0.01, max: 0.35, step: 0.005, value: lr,
    show: (v) => `lr = ${fmt(v)}`, valueText: (v) => `learning rate ${fmt(v)}: ${m.verdict(v).text}`,
    onInput: (v) => { lr = v; update(); },
  });
  const factorEl = el('div', { class: 'ls-eqs' });
  const verdictEl = el('p', { class: 'ls-verdict' });
  const line = el('div', { class: 'ls-line' });
  const table = el('div', { class: 'ls-tablewrap', tabindex: 0, role: 'region', 'aria-label': 'Eight steps (scrollable)' });
  const badge = badgeEl();
  side.append(el('div', { class: 'ls-panel' }, presetRow, slLr.row),
    el('div', { class: 'ls-panel' }, factorEl, verdictEl, line), badge);
  host.querySelector('.ls').append(q2, table);
  foot.textContent = FOOT;

  const secondTable = tableToggle(q2, { columns: ['step k', 'w'], rows: [], caption: 'w after k steps' });

  function numberLine() {
    const W = 400, H = 84, a = 14, b = 386, hi = 0.35, X = (v) => a + (v / hi) * (b - a);
    const crit = m.criticalLr, newton = m.newtonLr;
    const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'ls-numline', role: 'img', 'aria-label': `Learning-rate axis: below ${fmt(crit)} every step shrinks the distance to the minimum; from ${fmt(crit)} up it does not. Your lr is ${fmt(lr)}.` });
    s.append(
      svg('rect', { x: X(0), y: 26, width: X(crit) - X(0), height: 22, class: 'ls-zone ok' }),
      svg('rect', { x: X(crit), y: 26, width: X(hi) - X(crit), height: 22, class: 'ls-zone bad' }),
      svg('text', { x: (X(0) + X(crit)) / 2, y: 41, 'text-anchor': 'middle', class: 'ls-zl' }, 'distance shrinks'),
      svg('text', { x: (X(crit) + X(hi)) / 2, y: 41, 'text-anchor': 'middle', class: 'ls-zl' }, 'no'));
    for (const [v, txt, anchor] of [[0, '0', 'start'], [newton, `${fmt(newton)}`, 'middle'], [crit, fmt(crit), 'middle'], [hi, fmt(hi), 'end']]) {
      s.append(svg('line', { x1: X(v), x2: X(v), y1: 48, y2: 54, class: 'ls-tick' }), svg('text', { x: X(v), y: 68, 'text-anchor': anchor, class: 'ls-tl' }, txt));
    }
    s.append(svg('path', { d: `M${X(lr)},25 l-6,-9 h12 z`, class: 'ls-mark' }), svg('text', { x: Math.min(Math.max(X(lr), 50), W - 50), y: 11, 'text-anchor': 'middle', class: 'ls-ml' }, `your lr = ${fmt(lr)}`));
    line.replaceChildren(s);
  }

  function update() {
    keepFocus(() => {
      const st = m.iterate(m.w0, lr, STEPS), f = m.factor(lr), v = m.verdict(lr);
      presetBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(Math.abs(PRESET_LRS[i] - lr) < 1e-9)));
      factorEl.replaceChildren(...factorTex(m, lr).map(eqLine));
      verdictEl.innerHTML = '';
      verdictEl.append(el('strong', {}, VERDICT_LABEL[v.id]), ` — the distance to the minimum is multiplied by ${fmt(f)} each step; it ${v.text}.`);
      numberLine();
      // landscape plot
      p.clear('live');
      const live = p.layer('live');
      const inside = st.filter((s) => s.w >= -1 - 1e-9 && s.w <= 7 + 1e-9);
      p.polyline(inside.map((s) => [s.w, s.L]), { class: 'ls-path', layer: 'live' });
      const groups = new Map();
      inside.forEach((s) => { const g = fmt(s.w); (groups.get(g) ?? groups.set(g, []).get(g)).push(s); });
      let lastX = -1e9, lastY = -1e9;                                    // thin the numbers where dots crowd together (the table has them all)
      for (const g of groups.values()) {
        const first = g[0], ks = g.map((s) => s.k), y = Math.min(first.L, 71);
        live.append(svg('circle', { class: first.k === 0 ? 'plot-dot' : 'ls-pt', cx: p.sx(first.w), cy: p.sy(y), r: first.k === 0 ? 7 : 5 }));
        if (Math.hypot(p.sx(first.w) - lastX, p.sy(y) - lastY) < 24) continue;
        lastX = p.sx(first.w); lastY = p.sy(y);
        const lab = ks.length > 3 ? `${ks.slice(0, 3).join(',')}…` : ks.join(',');
        p.text(first.w, y, lab, { class: 'ls-ptlabel', dx: 9, dy: -8, layer: 'live' });
      }
      const outside = st.filter((s) => !inside.includes(s)).map((s) => s.k);
      note.textContent = outside.length ? `Steps ${outside.join(', ')} land outside this picture (w beyond the edges, or the loss above the top); the table has them.` : '';
      p.root.setAttribute('aria-label', `The loss curve with ${STEPS} steps at lr = ${fmt(lr)}: ${v.text}.`);
      // w against k
      const ws = st.map((s) => s.w), lo = Math.min(...ws, m.wMin), hiV = Math.max(...ws, m.wMin), pad = (hiV - lo) * 0.12 || 1;
      const yr = [lo - pad, hiV + pad];
      q2.querySelector('svg')?.remove();
      const pp = createPlot(document.createElement('div'), { width: isNarrow() ? 400 : 640, height: isNarrow() ? 220 : 200, x: [0, STEPS], y: yr, xLabel: 'step k', yLabel: 'w', ariaLabel: `w against the step number at lr = ${fmt(lr)}: ${v.text}.` });
      pp.axes({ xTicks: ticks([0, STEPS], 8), yTicks: ticks(yr, 4) });
      pp.line(0, m.wMin, STEPS, m.wMin, { class: 'ls-ref', layer: 'ref' });
      pp.polyline(st.map((s) => [s.k, s.w]), { class: 'ls-path', layer: 'path' });
      st.forEach((s) => pp.dot(s.k, s.w, { class: s.k === 0 ? 'plot-dot' : 'ls-pt', r: s.k === 0 ? 6 : 4.5, layer: 'path' }));
      pp.text(STEPS, m.wMin, `minimum, w = ${fmt(m.wMin)}`, { class: 'ls-ptlabel', anchor: 'end', dy: -6, layer: 'ref' });
      q2.insertBefore(pp.root, q2.firstChild);
      secondTable.update(st.map((s) => [s.k, s.w]));
      // table: k, w, w - 3, L, slope, ratio
      const rows = st.map((s, i) => {
        const prev = i ? st[i - 1].w - m.wMin : null, dist = s.w - m.wMin;
        const ratio = prev !== null && Math.abs(prev) > 1e-9 ? fmt(dist / prev) : '—';
        return el('tr', {}, [s.k, s.w, dist, s.L, s.grad].map((x) => el('td', {}, fmt(x))), el('td', {}, ratio));
      });
      table.replaceChildren(el('table', { class: 'ls-table' }, el('caption', {}, `Eight steps from w = ${fmt(m.w0)} at lr = ${fmt(lr)}. ratio = (w − ${fmt(m.wMin)}) ÷ the previous (w − ${fmt(m.wMin)}).`),
        el('thead', {}, el('tr', {}, ['k', 'w', `w − ${fmt(m.wMin)}`, 'L', 'slope g', 'ratio'].map((c) => el('th', { scope: 'col' }, c)))), el('tbody', {}, rows)));
      const rec = recordedTrajectory(files, lr);
      if (rec) {
        const diff = Math.max(...rec.states.map((r, i) => Math.abs(r.w - st[i].w)));
        badge.set(BADGE.match(`TorC recorded this exact run with its real optim::SGD (momentum 0, ${STEPS} steps); the browser agrees within ${diff < 1e-12 ? '1e-12' : diff.toExponential(0)} in w`));
      } else badge.set(BADGE.browser('No TorC recording for this lr: recomputed in your browser (w ← w − lr · slope)'));
    }, host);
  }
  update();
  return { update };
}

// ======================================================================================================================
// styles — injected once; nothing in the shared stylesheets is touched
// ======================================================================================================================
const CSS = `
.ls-host{margin:14px 0}
.ls{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(290px,1fr);gap:10px 22px;align-items:start}
.ls-plotbox{min-width:0}
.ls-plotbox .plot{max-height:min(46vh,340px)}
.ls-side{display:flex;flex-direction:column;gap:10px;min-width:0}
.ls-panel{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.65rem .9rem;min-width:0}
.ls-panel h4{margin:.5rem 0 .1rem}.ls-panel h4:first-child{margin-top:0}
.ls-slider{display:grid;grid-template-columns:1fr auto;gap:.1rem .6rem;align-items:center;margin:.25rem 0 .55rem}
.ls-slider label{grid-column:1/-1;font-size:.93rem;color:#c4ccd5}
.ls-slider input[type=range]{width:100%;min-width:0;height:30px;margin:0;accent-color:var(--fwd)}
.ls-val{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9rem;min-width:5.6em;text-align:right;color:var(--fg)}
.ls-hint{grid-column:1/-1;font-size:.84rem;color:var(--muted);margin:.1rem 0}
p.ls-hint{margin:.3rem 0}
.ls-jumps,.ls-presets,.ls-buttons{display:flex;flex-wrap:wrap;gap:6px}
.ls-buttons{margin:.2rem 0 .1rem}
.ls-eqs{min-width:0}
.ls-eq{overflow-x:auto;overflow-y:hidden;padding:.05rem 0;font-size:.95rem}
.ls-eq .katex{font-size:1.02em}
.ls-score{font-size:1.35rem;font-weight:700;margin:.25rem 0 .1rem;font-variant-numeric:tabular-nums}
.ls-verdict{margin:.45rem 0 .1rem;font-size:.93rem;color:#c4ccd5}
.ls-verdict strong{color:var(--fg)}
.ls-badge{display:inline-block;align-self:flex-start;font-size:.8rem;border:1px solid;border-radius:10px;padding:.15rem .7rem;line-height:1.4}
.ls-badge.ok{color:var(--ok)}.ls-badge.warn{color:var(--warn)}
.ls-foot{font-size:.84rem;color:var(--muted);margin:.4rem 0 0}
.ls-legend{display:flex;flex-wrap:wrap;gap:.1rem 1rem;font-size:.82rem;color:var(--muted);margin:.2rem 0}
.ls-sw{vertical-align:middle;overflow:visible}
.ls-drop{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:2 3}
.ls-secant{stroke:var(--warn);stroke-width:2.4}
.ls-sq{fill:var(--warn);stroke:var(--bg);stroke-width:1.5}
.ls-diamond{fill:var(--bg);stroke:var(--warn);stroke-width:2.4}
.ls-path{stroke:var(--fg);stroke-width:1.5;fill:none;opacity:.8}
.ls-pt{fill:var(--panel2);stroke:var(--fg);stroke-width:2}
.ls-ptlabel{fill:var(--fg);font-size:11px;font-weight:700;paint-order:stroke;stroke:var(--bg);stroke-width:3.5px}
.ls-ref{stroke:var(--ok);stroke-width:1.6;stroke-dasharray:6 4;fill:none}
.ls-tablewrap{overflow:auto;max-height:230px;margin:.5rem 0 0;border:1px solid var(--border);border-radius:var(--r)}
.ls-size .ls-tablewrap{max-height:none;margin:0}
.ls-size .ls-second{margin-top:0}
.ls-table{width:100%;border-collapse:collapse;font-size:.85rem}
.ls-table caption{text-align:left;color:var(--muted);padding:.35rem .6rem}
.ls-table th,.ls-table td{padding:.2rem .5rem;text-align:right;border-top:1px solid var(--border);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap}
.ls-table thead th{position:sticky;top:0;background:var(--panel2);color:var(--muted);font-weight:600}
.ls-table tr.now td{color:var(--fg);background:rgba(88,196,221,.08)}
.ls-second{margin-top:.6rem}
.ls-numline{display:block;width:100%;height:auto;max-height:110px}
.ls-numline text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.ls-zone{stroke-width:1.5}.ls-zone.ok{fill:rgba(131,193,103,.14);stroke:var(--ok)}.ls-zone.bad{fill:rgba(252,98,85,.1);stroke:var(--bwd);stroke-dasharray:5 3}
.ls-zl{fill:var(--fg);font-size:12px}.ls-tl{fill:var(--muted);font-size:12px}.ls-ml{fill:var(--fg);font-size:12px;font-weight:700}
.ls-tick{stroke:var(--muted);stroke-width:1.5}.ls-mark{fill:var(--fg)}
@media(max-width:860px){.ls{grid-template-columns:minmax(0,1fr)}.ls-plotbox .plot{max-height:none}}
@media(max-width:480px){.ls-panel{padding:.55rem .65rem}.ls-eq{font-size:.88rem}.ls-table{font-size:.78rem}.ls-table th,.ls-table td{padding:.2rem .3rem}}
`;
export function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('ls-styles')) return;
  const s = document.createElement('style');
  s.id = 'ls-styles';
  s.textContent = CSS;
  document.head.append(s);
}
