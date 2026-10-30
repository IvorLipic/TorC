// widgets/fit-game.js — the hand-fitting game of the lesson what-is-learning (brief §6 Part 0, §3 "Productive failure", R4).
// Data: data/whole-game.json (tools/trace/trace_whole_game.cpp): hours of practice and test scores of ten training students and
// five hidden students, and the run of nn::Linear(1, 1) + nn::MSELoss + optim::SGD that TorC made on the ten training points.
//
//   mountFitGame(host, D)   1. The learner moves a line with two handles (drag, or focus a handle and use the arrow keys). Each
//                              miss on a training point is drawn as a square whose side is the miss, so the loss (the mean of
//                              the squares) is visible. A readout gives the loss and its square root, the typical miss.
//                           2. "Let the computer fit the line" shows TorC's run step by step (Step, Play, To the end). Play
//                              never starts by itself, and with reduced motion it goes to the last step at once.
//                           3. "Show the hidden points" adds the five hidden students and scores both lines on them.
// The loss of the learner's line is computed here (mse, a port of nn::MSELoss for a line); site/test/what-is-learning.test.mjs
// compares it with TorC's probes[]. The computer's line and its losses are TorC's own numbers (runs[0].steps). Pure helpers are
// exported for the tests; the module touches the DOM only in mountFitGame.
import { el, svg } from '../core/dom.js';
import { createPlot, srTable } from './plot.js';

// ---- pure helpers ---------------------------------------------------------------------------------------------------------
// the mean of the squared misses of the line y = w·x + b on the points (xs, ys): what nn::MSELoss computes for nn::Linear(1, 1)
export function mse(w, b, xs, ys) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) { const r = w * xs[i] + b - ys[i]; s += r * r; }
  return s / xs.length;
}
// the square root of a mean squared error: the typical size of one miss, in the unit of the score
export const typicalMiss = (loss) => Math.sqrt(loss);
// the line through two points (x1, y1), (x2, y2)
export function lineThrough(x1, y1, x2, y2) {
  const w = (y2 - y1) / (x2 - x1);
  return { w, b: y1 - w * x1 };
}
// a readout number with 2 significant figures (brief §9), with the minus sign U+2212; tiny values read as 0
export function fmt2(v) {
  if (!Number.isFinite(v)) return '?';
  if (Math.abs(v) < 0.005) return '0';
  const s = String(Number(v.toPrecision(2)));
  return s.startsWith('-') ? `−${s.slice(1)}` : s;
}
// "score = 1.6 × hours + 1.9" for the line (w, b)
export function equation(w, b) {
  const bb = fmt2(Math.abs(b));
  return `score = ${fmt2(w)} × hours ${b < 0 && bb !== '0' ? '−' : '+'} ${bb}`;
}

// The two handles sit at these hours; the start line is flat at score 5 (TorC scores the same line as the probe "flat-5").
export const HANDLE_X = [0.5, 4];
export const START = { w: 0, b: 5 };
const X = [0, 4.5], Y = [0, 10];
const clampY = (v) => Math.min(Y[1], Math.max(Y[0], v));

// ---- styles ---------------------------------------------------------------------------------------------------------------
const CSS = `
.fg{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.fg-plot{max-width:720px}
.fg-plot svg{touch-action:none;user-select:none;-webkit-user-select:none}
.fg-key{font-size:.85rem;color:var(--muted);margin:.3rem 0 .6rem;max-width:44rem}
.fg-btns{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:.4rem 0}
.fg-btns[hidden]{display:none}
.fg-run{margin:0}
.fg .btn[aria-pressed=true]{border-color:var(--hid);color:var(--hid)}
.fg-read{display:grid;gap:.4rem;margin:.6rem 0 0;max-width:44rem}
.fg-read p{margin:0;font-size:.95rem}
.fg-read b{display:block;font-size:.8rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
.fg-read .you b{color:var(--fg)}
.fg-read .pc b{color:var(--ok)}
.fg-read .hid b{color:var(--hid)}
.fg{--hid:#d2a8ff;--sq:#e3b341}
.fg-sq{fill:rgba(227,179,65,.18);stroke:var(--sq);stroke-width:1.2}
.fg-sq.pc{fill:rgba(131,193,103,.16);stroke:var(--ok)}
.fg-you{stroke:var(--fg);stroke-width:2.5;fill:none}
.fg-you.back{stroke-dasharray:7 5;opacity:.75}
.fg-pc{stroke:var(--ok);stroke-width:3;fill:none}
.fg-pt{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.fg-ht{fill:none;stroke:var(--hid);stroke-width:2.2}
.fg-h{fill:var(--bg);stroke:var(--fg);stroke-width:2.5;cursor:ns-resize}
.fg-h:focus{outline:none}
.fg-h:focus-visible,.fg-h.drag{stroke:var(--warn);stroke-width:4}
.fg-lbl{font-size:12px;fill:var(--fg);paint-order:stroke;stroke:var(--panel);stroke-width:4px;stroke-linejoin:round}
.fg-lbl.pc{fill:var(--ok)}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('fg-styles')) return;
  document.head.append(el('style', { id: 'fg-styles' }, CSS));
}
const reducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

// ======================================================================================================================
export function mountFitGame(host, D) {
  injectStyles();
  const G = D.files['whole-game'];
  const tx = G.data.train.x, ty = G.data.train.y, hx = G.data.hidden.x, hy = G.data.hidden.y;
  const run = G.runs[0], last = run.steps.length - 1;
  const ys = [START.w * HANDLE_X[0] + START.b, START.w * HANDLE_X[1] + START.b];   // the heights of the two handles
  let phase = 'hand', k = 0, hiddenOn = false, timer = null, dragging = -1;

  const plotHost = el('div', { class: 'fg-plot' });
  const key = el('p', { class: 'fg-key' });
  const you = el('p', { class: 'you' }), pc = el('p', { class: 'pc', hidden: true }), hid = el('p', { class: 'hid', hidden: true });
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  const goBtn = el('button', { type: 'button', class: 'btn primary' }, 'Let the computer fit the line');
  const stepBtn = el('button', { type: 'button', class: 'btn' }, 'Step');
  const playBtn = el('button', { type: 'button', class: 'btn' }, 'Play');
  const endBtn = el('button', { type: 'button', class: 'btn' }, 'To the end');
  const hidBtn = el('button', { type: 'button', class: 'btn', 'aria-pressed': 'false' }, 'Show the hidden points');
  const runBtns = el('div', { class: 'fg-btns fg-run', role: 'group', 'aria-label': 'Steps of the computer', hidden: true }, stepBtn, playBtn, endBtn);
  host.replaceChildren(el('div', { class: 'fg' }, plotHost, key,
    el('div', { class: 'fg-btns' }, goBtn, runBtns, hidBtn), el('div', { class: 'fg-read' }, you, pc, hid), say));

  const handles = [0, 1].map((i) => svg('circle', {
    class: 'fg-h', r: 9, tabindex: 0, role: 'slider', 'aria-valuemin': Y[0], 'aria-valuemax': Y[1],
    'aria-label': i === 0 ? 'Left handle of your line. Use the up and down arrow keys.' : 'Right handle of your line. Use the up and down arrow keys.',
  }));
  const plot = createPlot(plotHost, {
    width: 640, height: 360, x: X, y: Y, xLabel: 'hours of practice in a week', yLabel: 'test score, out of 10',
    ariaLabel: 'Test score against hours of practice, with your line and the square of each miss', onResize: () => drawAll(),
  });
  srTable(plotHost, {
    caption: 'The students: hours of practice and test score',
    columns: ['hours', 'score', 'set'],
    rows: G.data.points.map((p) => [p.x, p.y, p.set === 'train' ? 'training point' : 'hidden point']),
  });

  const yourLine = () => lineThrough(HANDLE_X[0], ys[0], HANDLE_X[1], ys[1]);
  const pcLine = () => ({ w: run.steps[k].w, b: run.steps[k].b });

  function drawLine(l, cls, label, layer, above = true) {   // the label sits above the line, or below it when another line is higher
    const y0 = l.w * X[0] + l.b, y1 = l.w * X[1] + l.b;
    plot.curve([[X[0], y0], [X[1], y1]], { class: cls, layer });
    const lx = HANDLE_X[1] - 0.2, ly = Math.min(Y[1] - 0.8, Math.max(Y[0] + 0.8, l.w * lx + l.b));   // the label stays on the plot
    plot.text(lx, ly, label, { class: `fg-lbl${cls === 'fg-pc' ? ' pc' : ''}`, layer: 'labels', dy: above ? -10 : 20, anchor: 'end' });
  }
  function drawSquares(l, cls) {
    for (let i = 0; i < tx.length; i++) {
      const px = plot.sx(tx[i]), py = plot.sy(ty[i]), pp = plot.sy(l.w * tx[i] + l.b);
      const side = Math.abs(pp - py), left = tx[i] > 2.25;   // squares point toward the middle, so they stay on the plot
      plot.layer('sq').append(svg('rect', { class: cls, x: left ? px - side : px, y: Math.min(py, pp), width: side, height: side }));
    }
  }
  function draw() {
    ['sq', 'lines', 'pts', 'labels', 'handles'].forEach((n) => plot.clear(n));
    const mine = yourLine();
    if (phase === 'hand') drawSquares(mine, 'fg-sq');
    else drawSquares(pcLine(), 'fg-sq pc');
    const at = (l) => l.w * (HANDLE_X[1] - 0.2) + l.b, pcAbove = phase !== 'hand' && at(pcLine()) >= at(mine);
    drawLine(mine, phase === 'hand' ? 'fg-you' : 'fg-you back', 'your line', 'lines', !pcAbove);
    if (phase !== 'hand') drawLine(pcLine(), 'fg-pc', 'computer', 'lines', pcAbove);
    for (let i = 0; i < tx.length; i++) plot.dot(tx[i], ty[i], { class: 'fg-pt', r: 5, layer: 'pts' });
    if (hiddenOn) for (let i = 0; i < hx.length; i++) plot.dot(hx[i], hy[i], { class: 'fg-ht', r: 6, layer: 'pts' });
    const g = plot.layer('handles');
    handles.forEach((h, i) => {
      h.setAttribute('cx', plot.sx(HANDLE_X[i]));
      h.setAttribute('cy', plot.sy(ys[i]));
      h.setAttribute('aria-valuenow', ys[i].toFixed(1));
      h.setAttribute('aria-valuetext', `score ${fmt2(ys[i])} at ${HANDLE_X[i]} hours`);
      g.append(h);
    });
    text();
  }
  function drawAll() { plot.axes({ xTicks: [0, 1, 2, 3, 4], yTicks: [0, 2, 4, 6, 8, 10] }); draw(); }
  function text() {
    const mine = yourLine(), L = mse(mine.w, mine.b, tx, ty);
    key.textContent = phase === 'hand'
      ? 'Each blue dot is one training point. Each orange square shows one miss of your line. Its side is the miss.'
      : 'Each blue dot is one training point. Each green square shows one miss of the computer’s line. Its side is the miss.';
    if (hiddenOn) key.textContent += ' Each purple ring is a hidden point.';
    you.replaceChildren(el('b', {}, 'Your line'), `The line is ${equation(mine.w, mine.b)}. The loss is ${fmt2(L)}, and lower is better. The typical miss is ${fmt2(typicalMiss(L))} on a test out of 10.`);
    pc.hidden = phase === 'hand';
    if (phase !== 'hand') {
      const s = run.steps[k];
      pc.replaceChildren(el('b', {}, 'The computer’s line'), `After ${k} of ${last} steps, the line is ${equation(s.w, s.b)}. The loss is ${fmt2(s.train_loss)}, and the typical miss is ${fmt2(typicalMiss(s.train_loss))}.`);
    }
    hid.hidden = !hiddenOn;
    if (hiddenOn) {
      const mineH = mse(mine.w, mine.b, hx, hy);
      const parts = [`Your line misses a hidden point by about ${fmt2(typicalMiss(mineH))}.`];
      if (phase !== 'hand') parts.push(`The computer’s line misses one by about ${fmt2(typicalMiss(run.steps[k].hidden_loss))}.`);
      else parts.push('Let the computer fit its line to compare.');
      hid.replaceChildren(el('b', {}, 'On the hidden points'), parts.join(' '));
    }
  }
  const announce = () => { say.textContent = [...you.childNodes].slice(1).map((n) => n.textContent).join(' ') + (phase === 'hand' ? '' : ` ${pc.lastChild.textContent}`); };

  // ---- the handles: drag, or arrow keys ----
  const yAt = (clientY) => {
    const r = plot.root.getBoundingClientRect(), py = ((clientY - r.top) * plot.height) / r.height, a = plot.plotArea;
    return clampY(Y[0] + ((a.t + a.h - py) / a.h) * (Y[1] - Y[0]));
  };
  const nearest = (clientX) => {
    const r = plot.root.getBoundingClientRect(), px = ((clientX - r.left) * plot.width) / r.width;
    return Math.abs(px - plot.sx(HANDLE_X[0])) <= Math.abs(px - plot.sx(HANDLE_X[1])) ? 0 : 1;
  };
  plot.root.addEventListener('pointerdown', (e) => {
    dragging = e.target.classList?.contains('fg-h') ? handles.indexOf(e.target) : nearest(e.clientX);
    plot.root.setPointerCapture(e.pointerId);
    handles[dragging].classList.add('drag');
    ys[dragging] = yAt(e.clientY);
    draw();
    e.preventDefault();
  });
  plot.root.addEventListener('pointermove', (e) => { if (dragging >= 0) { ys[dragging] = yAt(e.clientY); draw(); } });
  const endDrag = () => { if (dragging < 0) return; handles[dragging].classList.remove('drag'); dragging = -1; announce(); };
  plot.root.addEventListener('pointerup', endDrag);
  plot.root.addEventListener('pointercancel', endDrag);
  handles.forEach((h, i) => h.addEventListener('keydown', (e) => {
    const d = { ArrowUp: 0.1, ArrowRight: 0.1, ArrowDown: -0.1, ArrowLeft: -0.1, PageUp: 1, PageDown: -1 }[e.key];
    if (d === undefined && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault();
    ys[i] = e.key === 'Home' ? Y[0] : e.key === 'End' ? Y[1] : clampY(Math.round((ys[i] + d) * 10) / 10);
    draw();
    handles[i].focus();
    announce();
  }));

  // ---- the computer's run ----
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } playBtn.textContent = 'Play'; };
  const show = (to) => { k = Math.max(0, Math.min(last, to)); draw(); };
  goBtn.addEventListener('click', () => {
    phase = 'computer';
    goBtn.hidden = true;
    runBtns.hidden = false;
    show(0);
    stepBtn.focus();
    say.textContent = `The computer starts with the line ${equation(run.steps[0].w, run.steps[0].b)}. Press Step to watch it learn.`;
  });
  stepBtn.addEventListener('click', () => { stop(); show(k + 1); say.textContent = pc.lastChild.textContent; });
  endBtn.addEventListener('click', () => { stop(); show(last); say.textContent = pc.lastChild.textContent; });
  playBtn.addEventListener('click', () => {
    if (timer) { stop(); say.textContent = pc.lastChild.textContent; return; }
    if (k >= last) show(0);
    if (reducedMotion()) { show(last); say.textContent = pc.lastChild.textContent; return; }
    playBtn.textContent = 'Pause';
    timer = setInterval(() => {
      show(k + 1);
      if (k >= last) { stop(); say.textContent = pc.lastChild.textContent; }
    }, 45);
  });
  hidBtn.addEventListener('click', () => {
    hiddenOn = !hiddenOn;
    hidBtn.setAttribute('aria-pressed', String(hiddenOn));
    draw();
    say.textContent = hiddenOn ? hid.lastChild.textContent : 'The hidden points are not shown.';
  });

  drawAll();
  return { destroy() { stop(); plot.destroy(); } };
}
