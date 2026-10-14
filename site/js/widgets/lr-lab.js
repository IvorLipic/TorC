// widgets/lr-lab.js — "descend the bowl": the loss landscape (contour2d), a Step button with a growing trail and table, and — depending on
// the mode — a learning-rate slider with presets and the derived critical rate, or a momentum slider.
//
//   mountDescent(host, D, { mode: 'step' | 'lr' | 'momentum' })
//     step      lr fixed at 0.05, nothing planned in advance: press Step, watch the trail and the table grow
//     lr        learning-rate slider, number box, presets 0.05 / 0.1 / 0.15 / 0.2, the critical rate (derived) marked on the slider
//     momentum  lr fixed at 0.05, momentum slider and presets 0 / 0.5 / 0.9
//
// Everything is recomputed in the browser (optim.js sgdStep on the engine's model); when the settings equal one of the six runs TorC
// recorded (data/optim.json) the recorded losses are drawn as open circles so you can see the two agree.
// Pure helpers (describeRun, yRange) are exported for the tests. Importing this module does not touch the DOM.
import { el, svg } from '../core/dom.js';
import { math } from '../core/katex.js';
import { createPlot, tableToggle } from './plot.js';
import { mountContour2d } from './contour2d.js';
import { makeModel, simulate, hessian, eigenvalues2, analyze, validateSGD } from '../optim.js';
import { injectStyle, SHARED_CSS, badge, sliderField, approx, round3, fmt, fitWidth, onResize, enhanceScrollers } from './sgd-ui.js';
import { updateWindowData, updateTable } from './sgd-update-window.js';

export const STEPS = 10;
export const MU_MAX = 1;                         // the momentum box accepts values below this (mu >= 1 never settles)
export const LR_MAX = 0.3;                       // lab limit for the number box (the slider stops at 0.25)
// 'below' / 'above', or 'just below' / 'just above' when the two numbers look the same at the digits we show
export const cmpWord = (a, b) => (fmt(a) === fmt(b) ? (a < b ? 'just below' : 'just above') : (a < b ? 'below' : 'above'));
// The two number boxes' rules: TorC's own constructor checks first, then this lab's limits (messages are shown under the box).
export function checkLrBox(v) {
  validateSGD(v, 0);
  if (v > LR_MAX) throw new Error(`this lab stops at ${LR_MAX} so the numbers stay readable (TorC itself accepts any lr above 0)`);
}
export function checkMuBox(lr, v) {
  validateSGD(lr, v);
  if (!(v < MU_MAX)) throw new Error(`momentum ${MU_MAX} or more never settles (the velocity does not decay; a loop without zero_grad() amounts to μ = 1), so this lab stays below ${MU_MAX}. TorC itself accepts any value from 0 up`);
}
const fmtDecade = (d) => (d >= -3 && d <= 4 ? fmt(10 ** d) : `1e${d}`);
export const LR_PRESETS = [{ lr: 0.05, tag: 'steady' }, { lr: 0.1, tag: 'overshoots' }, { lr: 0.15, tag: 'zig-zags' }, { lr: 0.2, tag: 'diverges' }];
export const MU_PRESETS = [{ mu: 0, tag: 'plain' }, { mu: 0.5, tag: 'bounces' }, { mu: 0.9, tag: 'bounces hard' }];

// ---- pure helpers ----------------------------------------------------------------------------------------------------
// rows = states after 0..N updates ([{w, b, loss}, …]); returns what the run did, in terms a test can check.
//   swings    how many times w passed the minimum's w by more than 0.1 on one side and then on the other
//   reach     the largest w the run reached (the minimum is at optimum.w)
export function describeRun(rows, optimum, margin = 0.1) {
  const L0 = rows[0].loss, Ln = rows.at(-1).loss;
  const finite = rows.every((r) => Number.isFinite(r.loss) && Number.isFinite(r.w));
  const rose = [];                                                   // update numbers after which the loss went UP
  for (let i = 1; i < rows.length; i++) if (!(rows[i].loss <= rows[i - 1].loss)) rose.push(i);
  let swings = 0, side = 0;
  for (const r of rows) {
    const d = r.w - optimum.w;
    if (Math.abs(d) <= margin) continue;
    const sgn = Math.sign(d);
    if (side && sgn !== side) swings++;
    side = sgn;
  }
  return { finite, diverging: !finite || Ln > L0, rose, swings, reach: Math.max(...rows.map((r) => r.w)), start: L0, end: Ln };
}

// decade range [lo, hi] (powers of ten) covering every positive value, at least two decades tall
export function yRange(values) {
  const v = values.filter((x) => x > 0 && Number.isFinite(x));
  let lo = Math.floor(Math.log10(Math.min(...v))), hi = Math.ceil(Math.log10(Math.max(...v)));
  if (hi - lo < 2) hi = lo + 2;
  return [lo, hi];
}

const CSS = `
.sgd-desc .sgd-grid{display:grid;grid-template-columns:minmax(0,1.12fr) minmax(0,1fr);gap:10px 18px;align-items:start}
.sgd-desc h4.sub{margin:.4rem 0 .15rem;color:var(--muted);font-size:.85rem}
.sgd-desc .chart svg{display:block;width:100%;height:auto;max-height:240px}
.sgd-desc .chart text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sgd-desc .lc-main{fill:none;stroke:var(--fwd);stroke-width:2.5}
.sgd-desc .lc-ghost{fill:none;stroke:var(--muted);stroke-width:1.8;stroke-dasharray:5 4}
.sgd-desc .lc-dot{fill:var(--fwd);stroke:var(--bg);stroke-width:1.5}
.sgd-desc .lc-rec{fill:none;stroke:var(--warn);stroke-width:2}
.sgd-desc .lc-ref{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 4}
.sgd-desc .lc-grid{stroke:var(--border);stroke-width:.6}
.sgd-desc .lc-t{fill:var(--muted);font-size:10.5px}
.sgd-desc .lc-l{fill:#c4ccd5;font-size:11px}
.sgd-desc .lc-keyline{fill:none}
.sgd-desc .stab{margin-top:.5rem;padding:.5rem .8rem;border:1px solid var(--border);border-radius:10px;font-size:.9rem;color:#c4ccd5;background:var(--panel2)}
.sgd-desc .stab h4{margin:0 0 .2rem;color:var(--fg);font-size:.9rem}
.sgd-desc .stab p{margin:.2rem 0}
.sgd-desc .tag{display:inline-block;font-size:.72rem;border:1px solid var(--warn);color:var(--warn);border-radius:99px;padding:0 .5rem;margin-right:.3rem}
.sgd-desc .formula{margin:.1rem 0 .3rem;padding:.2rem .6rem;border-left:3px solid var(--ok);background:rgba(131,193,103,.07);border-radius:0 8px 8px 0}
.sgd-desc .formula .katex-display{margin:.3em 0}
.sgd-desc .fixed{margin:0;font-size:.92rem;color:#c4ccd5}
.sgd-desc .presets{display:flex;flex-wrap:wrap;gap:6px}
.sgd-desc .win.update{margin:0}
@media(max-width:900px){.sgd-desc .sgd-grid{grid-template-columns:minmax(0,1fr)}}
`;

export function mountDescent(host, D, { mode = 'step' } = {}) {
  injectStyle('sgd-ui-css', SHARED_CSS); injectStyle('sgd-desc-css', CSS);
  const O = D.files.optim;
  const model = makeModel(D.scenes.sgd);
  const lam = eigenvalues2(hessian(model.x));
  const optimum = O.data.optimum;
  const state = { lr: 0.05, mu: 0, k: mode === 'step' ? 0 : STEPS };
  let rows = [], ghost = null, rec = null, recRows = null, lastChartW = 0;

  // ---- skeleton ---------------------------------------------------------------------------------------------------------
  const root = el('div', { class: `sgd-w sgd-desc mode-${mode}` });
  const status = el('div', { class: 'sgd-status', role: 'status', 'aria-live': 'polite' });
  const ctl = el('div', { class: 'sgd-ctl' });
  const left = el('div'), right = el('div');
  const chartHost = el('div', { class: 'chart' }), chartNote = el('p', { class: 'hint', style: 'margin:.1rem 0 .2rem;font-size:.8rem;min-height:1.2em' });
  const updHost = el('div'), updTitle = el('h4', { class: 'sub' });
  const stab = el('div', { class: 'stab' });
  root.append(
    badge('browser', mode === 'step'
      ? 'Every position, loss and gradient below is computed live by the JavaScript port of TorC’s rules (tested against TorC’s output; TorC works in float32 and this port in double, so a last digit can differ). The recorded TorC run is drawn as open circles when the settings match it.'
      : 'The runs below are recomputed live by the JavaScript port of TorC’s rules. When your settings equal a run TorC recorded, TorC’s losses appear as open circles on the loss curve. The critical learning rate is derived mathematics, not something TorC reports.'),
    ctl, status, el('div', { class: 'sgd-grid' }, left, right));
  host.append(root);
  const contour = mountContour2d(left, { model, base: { w: [-1, 5], b: [-1.5, 3] }, optimum: { w: optimum.w, b: optimum.b }, samples: O.surface.points });
  right.append(el('h4', { class: 'sub' }, 'Loss after each update (log scale, gridlines are powers of ten)'), chartHost, chartNote);
  const chartTable = tableToggle(right, { columns: ['update', 'loss (recomputed)', 'loss recorded by TorC'], rows: [], caption: 'Loss after each update' });
  right.append(updTitle, updHost);
  if (mode !== 'step') right.append(stab);

  // ---- controls -----------------------------------------------------------------------------------------------------------
  const btn = (key, label, cls, onclick) => el('button', { type: 'button', class: `btn ${cls ?? ''}`.trim(), 'data-key': key, onclick }, label);
  const stepBtn = btn('step', 'Step ▶  (optimizer.step())', 'primary', () => { if (state.k < STEPS) state.k++; render(); });
  const backBtn = btn('back', '◀ Back', 'small', () => { if (state.k > 0) state.k--; render(); });
  const resetBtn = btn('reset', 'Reset', 'small', () => { state.k = 0; render(); });
  const allBtn = btn('all', `Run all ${STEPS}`, 'small', () => { state.k = STEPS; render(); });
  const kCount = el('span', { class: 'mono', style: 'font-size:.85rem;color:var(--muted)' });
  ctl.append(el('div', { class: 'row' }, stepBtn, backBtn, resetBtn, allBtn, kCount));

  let lrField = null, muField = null;
  const lrPresetBtns = [], muPresetBtns = [];
  const tried = new Set([`lr${state.lr}`, `mu${state.mu}`]);       // a preset's verdict is shown only after it has been tried (so it cannot spoil a prediction)
  const tagText = (key, tag) => (tried.has(key) ? tag : 'try it');
  if (mode === 'lr') {
    lrField = sliderField({ id: 'sgd-lr', label: 'Learning rate η (lr)', min: 0.01, max: 0.25, step: 0.005, value: state.lr, validate: checkLrBox, onChange: (v) => { state.lr = round3(v); render(); }, marker: { value: analyze(0.1, 0, lam).critical_lr, label: `critical ${fmt(analyze(0.1, 0, lam).critical_lr)}` } });
    ctl.append(lrField.root, el('div', { class: 'presets', role: 'group', 'aria-label': 'Learning-rate presets (runs TorC recorded)' },
      LR_PRESETS.map((p) => { const b = el('button', { type: 'button', class: 'btn small', 'data-key': 'lr' + p.lr, 'aria-pressed': 'false', onclick: () => { state.lr = p.lr; tried.add(`lr${p.lr}`); lrField.set(p.lr); render(); } }, `${p.lr}`, el('small', {}, tagText(`lr${p.lr}`, p.tag))); lrPresetBtns.push([p.lr, b, p.tag]); return b; })));
  } else {
    ctl.append(el('p', { class: 'fixed' }, 'Learning rate η = ', el('strong', { class: 'mono' }, String(state.lr)), mode === 'step' ? ' (fixed here; you will change it in the next section)' : ' (fixed here)'));
  }
  if (mode === 'momentum') {
    muField = sliderField({ id: 'sgd-mu', label: 'Momentum μ', min: 0, max: 0.95, step: 0.05, value: state.mu, validate: (v) => checkMuBox(state.lr, v), onChange: (v) => { state.mu = round3(v); render(); }, marker: null });
    ctl.append(muField.root, el('div', { class: 'presets', role: 'group', 'aria-label': 'Momentum presets (runs TorC recorded)' },
      MU_PRESETS.map((p) => { const b = el('button', { type: 'button', class: 'btn small', 'data-key': 'mu' + p.mu, 'aria-pressed': 'false', onclick: () => { state.mu = p.mu; tried.add(`mu${p.mu}`); muField.set(p.mu); render(); } }, `μ = ${p.mu}`, el('small', {}, tagText(`mu${p.mu}`, p.tag))); muPresetBtns.push([p.mu, b, p.tag]); return b; })));
    root.insertBefore(el('div', { class: 'formula' }, el('div', { class: 'math', html: math('v \\leftarrow \\mu\\,v + \\overline{p}\\qquad p \\leftarrow p - \\eta\\,v', true) }),
      el('p', { class: 'hint', style: 'margin:0' }, 'TorC’s SGD with momentum: v starts at 0, so the first update equals a plain one. With μ = 0 it takes the plain branch.')), status);
  }

  // ---- drawing --------------------------------------------------------------------------------------------------------------
  function compute() {
    const opt = { lr: state.lr, momentum: state.mu, steps: STEPS };
    const run = simulate(model, opt);
    rows = [...run.steps, run.final];
    ghost = mode === 'momentum' && state.mu !== 0 ? (() => { const g = simulate(model, { lr: state.lr, momentum: 0, steps: STEPS }); return [...g.steps, g.final]; })() : null;
    rec = O.runs.find((r) => approx(r.lr, state.lr) && approx(r.momentum, state.mu)) ?? null;
    recRows = rec ? [...rec.steps, rec.final] : null;
  }

  function drawChart() {
    const all = [...rows.map((r) => r.loss), ...(ghost ? ghost.map((r) => r.loss) : []), ...(recRows ? recRows.map((r) => r.loss) : [])];
    const [lo, hi] = yRange(all);
    chartHost.replaceChildren();
    const W = fitWidth(chartHost, 300, 460, 440), H = Math.round(W * 0.53), m = { l: 64, r: 12, t: 12, b: 38 };
    lastChartW = W;
    const p = createPlot(chartHost, { width: W, height: H, x: [0, STEPS], y: [lo, hi], xLabel: 'updates applied', yLabel: 'loss', margin: m, ariaLabel: 'Loss after each update, log scale' });
    p.axes({ xTicks: [0, 2, 4, 6, 8, 10], yTicks: [] });
    const s = p.root;
    const every = Math.max(1, Math.ceil((hi - lo) / 6));       // at most about six gridlines, even for a run that explodes
    for (let d = lo; d <= hi; d += every) {
      s.append(svg('line', { class: 'lc-grid', x1: m.l, x2: W - m.r, y1: p.sy(d), y2: p.sy(d) }), svg('text', { class: 'lc-t', x: m.l - 6, y: p.sy(d) + 3.5, 'text-anchor': 'end' }, fmtDecade(d)));
    }
    const ly = (v) => Math.log10(Math.max(v, 1e-12));
    const L0 = rows[0].loss;
    s.append(svg('line', { class: 'lc-ref', x1: m.l, x2: W - m.r, y1: p.sy(ly(L0)), y2: p.sy(ly(L0)) }), svg('text', { class: 'lc-t', x: W - m.r - 2, y: p.sy(ly(L0)) - 3, 'text-anchor': 'end' }, `start ${fmt(L0)}`));
    const shown = rows.slice(0, state.k + 1);
    if (ghost) p.curve(ghost.slice(0, state.k + 1).map((r, i) => [i, ly(r.loss)]), { class: 'lc-ghost' });
    if (shown.length > 1) p.curve(shown.map((r, i) => [i, ly(r.loss)]), { class: 'lc-main' });
    shown.forEach((r, i) => p.dot(i, ly(r.loss), { class: 'lc-dot', r: 3.4, title: `after ${i} updates: loss ${fmt(r.loss)}` }));
    if (recRows) shown.forEach((_, i) => p.dot(i, ly(recRows[i].loss), { class: 'lc-rec', r: 6.2, title: `TorC recorded: loss ${fmt(recRows[i].loss)}` }));
    const last = shown.at(-1), prevL = shown.length > 1 ? shown.at(-2).loss : null;
    const descending = prevL != null && last.loss < prevL;
    const py = p.sy(ly(last.loss)), bottom = H - m.b;
    const labelY = descending ? (py + 17 > bottom - 2 ? py - 12 : py + 17) : py - 10;     // never on top of the x tick labels
    s.append(svg('text', { class: 'lc-l', x: shown.length > 1 ? p.sx(shown.length - 1) - 9 : p.sx(0) + 10, y: labelY, 'text-anchor': shown.length > 1 ? 'end' : 'start' }, fmt(last.loss)));
    chartNote.textContent = ghost ? 'Solid line: the run with these settings. Dashed grey line: μ = 0 (no momentum), for comparison.' : '';
    chartHost.querySelector('svg').setAttribute('aria-label', `Loss after each update on a log scale. It starts at ${fmt(L0)} and is ${fmt(last.loss)} after ${state.k} updates.`);
    chartTable.update(shown.map((r, i) => [i, r.loss, recRows ? recRows[i].loss : '–']));
  }

  function drawUpdate() {
    const u = Math.max(1, state.k);
    const a = rows[u - 1], z = rows[u];
    const mom = state.mu !== 0;
    const data = updateWindowData({ lr: state.lr, momentum: state.mu, params: [
      { name: 'W', before: a.w, grad: a.gw, vAfter: a.vw, after: z.w },
      { name: 'b', before: a.b, grad: a.gb, vAfter: a.vb, after: z.b }] });
    updTitle.textContent = state.k === 0 ? 'The next update, worked out (not applied yet)' : `Update ${state.k}, just applied${mom ? ' (with momentum)' : ''}`;
    updHost.replaceChildren(updateTable(data.table));
  }

  function summary() {
    const d = describeRun(rows.slice(0, state.k + 1), optimum);
    const L = (i) => fmt(rows[i].loss);
    if (mode === 'step') {
      if (state.k === 0) return { text: `Nothing has happened yet: w = ${fmt(rows[0].w)}, b = ${fmt(rows[0].b)}, loss ${L(0)}. The dashed arrow is where one step() would take the parameters. Press Step.`, kind: '' };
      const a = rows[state.k - 1], z = rows[state.k];
      return { text: `Update ${state.k}: w ${fmt(a.w)} → ${fmt(z.w)}, b ${fmt(a.b)} → ${fmt(z.b)}, loss ${fmt(a.loss)} → ${fmt(z.loss)}.${state.k === STEPS ? ' That is all ${STEPS} recorded updates; Reset starts over.' : ''}`, kind: '' };
    }
    const first = `After ${state.k} update${state.k === 1 ? '' : 's'} the loss is ${fmt(rows[state.k].loss)} (it started at ${L(0)}).`;
    if (state.k === 0) return { text: 'Press Step, or Run all, to apply updates with these settings.', kind: '' };
    if (!d.finite) return { text: 'The numbers overflowed (they are no longer finite): these settings diverge.', kind: 'bad' };
    if (d.diverging) return { text: `${first} It is higher than where it started: these settings diverge.`, kind: 'bad' };
    if (fmt(d.end) === fmt(d.start)) return { text: `${first} At this learning rate the steps are so small that the loss barely moves.`, kind: 'warn' };
    const bits = [];
    if (d.swings === 1) bits.push(`w overshot the minimum once (it reached ${fmt(d.reach)}; the minimum is at ${fmt(optimum.w)})`);
    else if (d.swings > 1) bits.push(`w swung from one side of the minimum to the other ${d.swings} times`);
    if (d.rose.length) bits.push(`the loss went up at update${d.rose.length === 1 ? '' : 's'} ${d.rose.join(', ')}`);
    return { text: `${first}${bits.length ? ' ' + bits.join('; ') + ', yet the loss ended lower than it started.' : ' It fell at every update.'}`, kind: bits.length ? 'warn' : '' };
  }

  function drawStability() {
    const an = analyze(state.lr, state.mu, lam);
    const crit = an.critical_lr;
    const lines = [];
    const gloss = `<p><strong>λmax</strong> is how sharply the bowl curves along its steepest direction and <strong>λmin</strong> along its flattest. Here λmax = <strong class="mono">${fmt(lam[0])}</strong> and λmin = <strong class="mono">${fmt(lam[1])}</strong> (the eigenvalues of the Hessian, the matrix of second derivatives of the loss, computed here from your x values: mathematics, not a TorC output).</p>`;
    if (mode === 'lr') {
      lines.push(`<h4><span class="tag">derived</span>Is this learning rate safe? η &lt; 2 / λmax</h4>`, gloss);
      lines.push(`<p>One update multiplies the error along the steepest direction by 1 − η·λmax = 1 − ${fmt(state.lr)}·${fmt(lam[0])} = <strong class="mono">${fmt(an.factors[0])}</strong>, and along the flattest by <strong class="mono">${fmt(an.factors[1])}</strong>. It shrinks only if both have size below 1, i.e. η &lt; 2/λmax = <strong class="mono">${fmt(crit)}</strong>.</p>`);
    } else {
      lines.push(`<h4><span class="tag">derived</span>How momentum moves the safe limit: η &lt; 2 (1 + μ) / λmax</h4>`, gloss);
      lines.push(`<p>At μ = ${fmt(state.mu)} the safe limit for the learning rate is 2(1+${fmt(state.mu)})/λmax = <strong class="mono">${fmt(crit)}</strong>${state.mu === 0 ? ' (plain SGD)' : ', higher than the plain limit ' + fmt(analyze(state.lr, 0, lam).critical_lr)}. The slowest error shrink factor per update is <strong class="mono">${fmt(an.spectral_radius)}</strong>${an.spectral_radius < 1 ? ' (below 1: it converges in the long run, possibly after bouncing)' : ' (1 or more: it does not converge)'}.</p>`);
    }
    const verdict = an.stable_predicted ? `η = ${fmt(state.lr)} is ${cmpWord(state.lr, crit)} ${fmt(crit)}: predicted to converge.` : `η = ${fmt(state.lr)} is ${cmpWord(state.lr, crit)} ${fmt(crit)}: predicted to diverge.`;
    lines.push(`<p><strong>${verdict}</strong>${state.k ? ` Observed here: the loss went from ${fmt(rows[0].loss)} to ${fmt(rows[state.k].loss)} in ${state.k} updates.` : ''}</p>`);
    stab.innerHTML = lines.join('');
  }

  function render() {
    compute();
    const trail = rows.slice(0, state.k + 1);
    contour.draw({ trail, next: state.k < STEPS ? rows[state.k + 1] : null, caption: `Learning rate ${state.lr}${state.mu ? `, momentum ${state.mu}` : ''}.` });
    drawChart();
    drawUpdate();
    if (mode !== 'step') drawStability();
    const sum = summary();
    status.textContent = sum.text;
    status.className = 'sgd-status ' + sum.kind;
    kCount.textContent = `updates applied: ${state.k} of ${STEPS}`;
    stepBtn.setAttribute('aria-disabled', String(state.k >= STEPS));
    backBtn.setAttribute('aria-disabled', String(state.k <= 0));
    resetBtn.setAttribute('aria-disabled', String(state.k <= 0));
    allBtn.setAttribute('aria-disabled', String(state.k >= STEPS));
    // typing a value equal to a preset counts as trying it
    if (mode === 'lr') for (const p of LR_PRESETS) if (approx(p.lr, state.lr)) tried.add(`lr${p.lr}`);
    if (mode === 'momentum') for (const p of MU_PRESETS) if (approx(p.mu, state.mu)) tried.add(`mu${p.mu}`);
    for (const [v, b, tag] of lrPresetBtns) { b.setAttribute('aria-pressed', String(approx(v, state.lr))); b.querySelector('small').textContent = tagText(`lr${v}`, tag); }
    for (const [v, b, tag] of muPresetBtns) { b.setAttribute('aria-pressed', String(approx(v, state.mu))); b.querySelector('small').textContent = tagText(`mu${v}`, tag); }
    enhanceScrollers(root);
    if (lrField) {
      const an = analyze(state.lr, 0, lam);
      lrField.describe(`${state.lr}; ${cmpWord(state.lr, an.critical_lr)} the derived critical learning rate ${fmt(an.critical_lr)}`);
    }
    if (muField) muField.describe(`momentum ${state.mu}`);
  }

  const ac = new AbortController();
  onResize(() => { if (Math.abs(fitWidth(chartHost, 300, 460, 440) - lastChartW) > 8) drawChart(); }, ac.signal);
  render();
  return { destroy() { ac.abort(); root.remove(); }, state };
}
