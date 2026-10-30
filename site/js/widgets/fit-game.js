// widgets/fit-game.js — the hand-fitting game of the lesson what-is-learning (brief §6 Part 0, §3 "Productive failure", R4).
// Data: data/whole-game.json (tools/trace/trace_whole_game.cpp): hours of practice and test scores of ten training students and
// thirty hidden students, and the run of nn::Linear(1, 1) + nn::MSELoss + optim::SGD that TorC made on the ten training points.
//
//   mountFitGame(host, D)   1. The learner moves a line with two white handles (drag a handle, or focus it and use the arrow
//                              keys). Each miss on a training point is drawn as a yellow square whose side is the miss, so the
//                              loss (the mean of the squares) is visible. A readout gives the loss and its square root, the
//                              typical miss.
//                           2. "Let the computer fit the line" shows TorC's run step by step (Step, Play, To the end). Play
//                              never starts by itself, and with reduced motion it goes to the last step at once.
//                           3. "Show the hidden points" opens only after the computer's last step. From then on the learner's
//                              line is fixed, so nobody can tune a line on the hidden points (they score both lines on new data).
// Only a drag that starts on a handle moves the line; any other touch scrolls the page. The handles are kept as elements, so
// keyboard focus stays on a handle while the plot redraws. The loss of the learner's line is computed here (mse, a port of
// nn::MSELoss for a line); site/test/what-is-learning.test.mjs compares it with TorC's probes[]. The computer's line and its losses
// are TorC's own numbers (runs[0].steps). Pure helpers are exported for the tests; the module touches the DOM only in mountFitGame.
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
// May the hidden points be shown? Only after the computer reached its last step (H1: the hand line is never tuned on them).
export const hiddenAllowed = (reachedEnd) => reachedEnd === true;

// The two handles sit at these hours; the start line is flat at score 5 (TorC scores the same line as the probe "flat-5").
export const HANDLE_X = [0.5, 4];
export const START = { w: 0, b: 5 };
const X = [0, 4.5], Y = [0, 10];
const clampY = (v) => Math.min(Y[1], Math.max(Y[0], v));

// ---- styles ---------------------------------------------------------------------------------------------------------------
const CSS = `
.fg{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.fg-plot{max-width:720px}
.fg-plot svg{user-select:none;-webkit-user-select:none}
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
.fg-sq{fill:rgba(227,179,65,.2);stroke:var(--sq);stroke-width:1.2}
.fg-sq.pc{fill:rgba(131,193,103,.16);stroke:var(--ok)}
.fg-you{stroke:var(--fg);stroke-width:2.5;fill:none}
.fg-you.back{stroke-dasharray:7 5;opacity:.75}
.fg-pc{stroke:var(--ok);stroke-width:3;fill:none}
.fg-pt{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.fg-ht{fill:none;stroke:var(--hid);stroke-width:2}
.fg-hg{cursor:ns-resize;touch-action:none;outline:none}
.fg-hg .hit{fill:transparent;stroke:none}
.fg-hg .dot{fill:var(--fg);stroke:var(--bg);stroke-width:2.5}
.fg-hg:focus-visible .dot,.fg-hg.drag .dot{stroke:var(--warn);stroke-width:4}
.fg-hg.frozen{cursor:default}
.fg-hg.frozen .dot{fill:var(--muted)}
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
  const run = G.runs[0], last = run.steps.length - 1, best = G.derived.best_hidden_line;
  const ys = [START.w * HANDLE_X[0] + START.b, START.w * HANDLE_X[1] + START.b];   // the heights of the two handles
  let phase = 'hand', k = 0, reachedEnd = false, hiddenOn = false, frozen = false, timer = null, dragging = -1;

  const plotHost = el('div', { class: 'fg-plot' });
  const key = el('p', { class: 'fg-key' });
  const you = el('p', { class: 'you' }), pc = el('p', { class: 'pc', hidden: true }), hid = el('p', { class: 'hid', hidden: true });
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  const goBtn = el('button', { type: 'button', class: 'btn primary' }, 'Let the computer fit the line');
  const stepBtn = el('button', { type: 'button', class: 'btn' }, 'Step');
  const playBtn = el('button', { type: 'button', class: 'btn' }, 'Play');
  const endBtn = el('button', { type: 'button', class: 'btn' }, 'To the end');
  const hidBtn = el('button', { type: 'button', class: 'btn', 'aria-pressed': 'false', disabled: true }, 'Show the hidden points');
  const runBtns = el('div', { class: 'fg-btns fg-run', role: 'group', 'aria-label': 'Steps of the computer', hidden: true }, stepBtn, playBtn, endBtn);
  host.replaceChildren(el('div', { class: 'fg' }, plotHost, key,
    el('div', { class: 'fg-btns' }, goBtn, runBtns, hidBtn), el('div', { class: 'fg-read' }, you, pc, hid), say));

  // a handle: a large invisible touch target (radius 20, so at least 40 px on a phone) around the visible white dot
  const handles = [0, 1].map((i) => svg('g', {
    class: 'fg-hg', tabindex: 0, role: 'slider', 'aria-valuemin': Y[0], 'aria-valuemax': Y[1],
    'aria-label': i === 0 ? 'Left handle of your line. Use the up and down arrow keys.' : 'Right handle of your line. Use the up and down arrow keys.',
  }, svg('circle', { class: 'hit', r: 20 }), svg('circle', { class: 'dot', r: 10 })));
  const plot = createPlot(plotHost, {
    width: 640, height: 380, x: X, y: Y, xLabel: 'hours of practice in a week', yLabel: 'test score, out of 10',
    ariaLabel: 'Test score against hours of practice, with your line and the square of each miss', onResize: () => drawAll(),
  });
  srTable(plotHost, {
    caption: 'The students: hours of practice and test score',
    columns: ['hours', 'score', 'set'],
    rows: G.data.points.map((p) => [p.x, p.y, p.set === 'train' ? 'training point' : 'hidden point']),
  });

  const yourLine = () => lineThrough(HANDLE_X[0], ys[0], HANDLE_X[1], ys[1]);
  const pcLine = () => ({ w: run.steps[k].w, b: run.steps[k].b });

  function drawLine(l, cls, label, above = true) {   // the label sits above the line, or below it when the other line is higher
    plot.curve([[X[0], l.w * X[0] + l.b], [X[1], l.w * X[1] + l.b]], { class: cls, layer: 'lines' });
    const lx = HANDLE_X[1] - 0.25, ly = Math.min(Y[1] - 0.8, Math.max(Y[0] + 0.8, l.w * lx + l.b));   // the label stays on the plot
    plot.text(lx, ly, label, { class: `fg-lbl${cls === 'fg-pc' ? ' pc' : ''}`, layer: 'labels', dy: above ? -10 : 20, anchor: 'end' });
  }
  function drawSquares(l, cls) {
    for (let i = 0; i < tx.length; i++) {
      const px = plot.sx(tx[i]), py = plot.sy(ty[i]), pp = plot.sy(l.w * tx[i] + l.b);
      const side = Math.abs(pp - py), left = tx[i] > 2.25;   // squares point toward the middle, so they stay on the plot
      plot.layer('sq').append(svg('rect', { class: cls, x: left ? px - side : px, y: Math.min(py, pp), width: side, height: side }));
    }
  }
  function placeHandles() {   // the same two elements every time, so a focused handle keeps its focus
    const g = plot.layer('handles'), had = handles.indexOf(document.activeElement);
    handles.forEach((h, i) => {
      h.setAttribute('transform', `translate(${plot.sx(HANDLE_X[i]).toFixed(1)} ${plot.sy(ys[i]).toFixed(1)})`);
      h.setAttribute('aria-valuenow', ys[i].toFixed(1));
      h.setAttribute('aria-valuetext', `score ${fmt2(ys[i])} at ${HANDLE_X[i]} hours`);
      h.classList.toggle('frozen', frozen);
      h.setAttribute('aria-disabled', String(frozen));
      if (h.parentNode !== g) g.append(h);
    });
    if (had >= 0 && document.activeElement !== handles[had]) handles[had].focus({ preventScroll: true });
  }
  function draw() {
    ['sq', 'lines', 'pts', 'labels'].forEach((n) => plot.clear(n));
    const mine = yourLine();
    if (phase === 'hand') drawSquares(mine, 'fg-sq');
    else drawSquares(pcLine(), 'fg-sq pc');
    const at = (l) => l.w * (HANDLE_X[1] - 0.25) + l.b, pcAbove = phase !== 'hand' && at(pcLine()) >= at(mine);
    drawLine(mine, phase === 'hand' ? 'fg-you' : 'fg-you back', 'your line', !pcAbove);
    if (phase !== 'hand') drawLine(pcLine(), 'fg-pc', 'computer', pcAbove);
    for (let i = 0; i < tx.length; i++) plot.dot(tx[i], ty[i], { class: 'fg-pt', r: 5, layer: 'pts' });
    if (hiddenOn) for (let i = 0; i < hx.length; i++) plot.dot(hx[i], hy[i], { class: 'fg-ht', r: 5, layer: 'pts' });
    placeHandles();
    text();
  }
  function drawAll() { plot.axes({ xTicks: [0, 1, 2, 3, 4], yTicks: [0, 2, 4, 6, 8, 10] }); draw(); }
  function text() {
    const mine = yourLine(), L = mse(mine.w, mine.b, tx, ty);
    key.textContent = phase === 'hand'
      ? 'Each blue dot is one training point. The side of each yellow square is the distance from a point to your line.'
      : 'Each blue dot is one training point. The side of each green square is the distance from a point to the computer’s line.';
    if (hiddenOn) key.textContent += ' Each purple ring is a hidden point. Your line is now locked.';
    else if (phase !== 'hand' && !reachedEnd) key.textContent += ' The hidden points open after the last step.';
    you.replaceChildren(el('b', {}, 'Your line'), `The line is ${equation(mine.w, mine.b)}. The loss is ${fmt2(L)}, and lower is better. The typical miss is ${fmt2(typicalMiss(L))} on a test out of 10.`);
    pc.hidden = phase === 'hand';
    if (phase !== 'hand') {
      const s = run.steps[k];
      pc.replaceChildren(el('b', {}, 'The computer’s line'), `After ${k} of ${last} steps, the line is ${equation(s.w, s.b)}. The loss is ${fmt2(s.train_loss)}, and the typical miss is ${fmt2(typicalMiss(s.train_loss))}.`);
    }
    hid.hidden = !hiddenOn;
    if (hiddenOn) {
      hid.replaceChildren(el('b', {}, 'On the hidden points'),
        `Your line misses a hidden point by about ${fmt2(typicalMiss(mse(mine.w, mine.b, hx, hy)))}. The computer’s line misses one by about ${fmt2(typicalMiss(run.steps[last].hidden_loss))}. `
        + `The best possible line for the hidden points misses one by about ${fmt2(typicalMiss(best.hidden_loss))}.`);
    }
  }
  const announce = () => { say.textContent = [...you.childNodes].slice(1).map((n) => n.textContent).join(' '); };

  // ---- the handles: a drag that starts on a handle, or the arrow keys ----
  const yAt = (clientY) => {
    const r = plot.root.getBoundingClientRect(), py = ((clientY - r.top) * plot.height) / r.height, a = plot.plotArea;
    return clampY(Y[0] + ((a.t + a.h - py) / a.h) * (Y[1] - Y[0]));
  };
  handles.forEach((h, i) => {
    h.addEventListener('pointerdown', (e) => {
      if (frozen) return;
      dragging = i;
      plot.root.setPointerCapture(e.pointerId);
      h.classList.add('drag');
      e.preventDefault();
    });
    h.addEventListener('keydown', (e) => {
      const d = { ArrowUp: 0.1, ArrowRight: 0.1, ArrowDown: -0.1, ArrowLeft: -0.1, PageUp: 1, PageDown: -1 }[e.key];
      if (d === undefined && e.key !== 'Home' && e.key !== 'End') return;
      e.preventDefault();
      if (frozen) { say.textContent = 'Your line is locked, because the hidden points are shown.'; return; }
      ys[i] = e.key === 'Home' ? Y[0] : e.key === 'End' ? Y[1] : clampY(Math.round((ys[i] + d) * 10) / 10);
      draw();
      announce();
    });
  });
  plot.root.addEventListener('pointermove', (e) => { if (dragging >= 0) { ys[dragging] = yAt(e.clientY); draw(); } });
  const endDrag = () => { if (dragging < 0) return; handles[dragging].classList.remove('drag'); dragging = -1; announce(); };
  plot.root.addEventListener('pointerup', endDrag);
  plot.root.addEventListener('pointercancel', endDrag);

  // ---- the computer's run ----
  const stop = () => { if (timer) { clearInterval(timer); timer = null; } playBtn.textContent = 'Play'; };
  const show = (to) => {
    k = Math.max(0, Math.min(last, to));
    if (k === last && !reachedEnd) { reachedEnd = true; hidBtn.disabled = !hiddenAllowed(reachedEnd); }
    draw();
  };
  const sayStep = () => { say.textContent = pc.lastChild.textContent + (k === last ? ' You can now show the hidden points.' : ''); };
  goBtn.addEventListener('click', () => {
    phase = 'computer';
    goBtn.hidden = true;
    runBtns.hidden = false;
    show(0);
    stepBtn.focus();
    say.textContent = `The computer starts with the line ${equation(run.steps[0].w, run.steps[0].b)}. Press Step to watch it learn.`;
  });
  stepBtn.addEventListener('click', () => { stop(); show(k + 1); sayStep(); });
  endBtn.addEventListener('click', () => { stop(); show(last); sayStep(); });
  playBtn.addEventListener('click', () => {
    if (timer) { stop(); sayStep(); return; }
    if (k >= last) show(0);
    if (reducedMotion()) { show(last); sayStep(); return; }
    playBtn.textContent = 'Pause';
    timer = setInterval(() => {
      show(k + 1);
      if (k >= last) { stop(); sayStep(); }
    }, 45);
  });
  hidBtn.addEventListener('click', () => {
    if (!hiddenAllowed(reachedEnd)) return;
    hiddenOn = !hiddenOn;
    frozen = true;   // once the hidden points were seen, the hand line stays where it is
    hidBtn.setAttribute('aria-pressed', String(hiddenOn));
    draw();
    say.textContent = hiddenOn ? `${hid.lastChild.textContent} Your line is now locked.` : 'The hidden points are not shown. Your line stays locked.';
  });

  drawAll();
  return { destroy() { stop(); plot.destroy(); } };
}
