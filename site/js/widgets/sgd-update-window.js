// widgets/sgd-update-window.js — "the update on paper": the SGD rule, the same rule with the real numbers, and the
// before | grad | step | after table. Same idea as explain.js' math windows (rule, then numbers), for an optimizer step.
//
//   updateWindowData({lr, momentum, params:[{name, before, grad, vBefore?, after}]})  -> {title, kind, blocks[], table}   (pure; LaTeX strings)
//   renderUpdateWindow(box, data)                                                       -> DOM (KaTeX through core/katex.js)
//   mountUpdateWindow(host, D)                                                          -> the lesson widget: pick update 1..4 of the recorded lr 0.05 run
// The numbers of mountUpdateWindow come from the TorC run in data/optim.json; the "after" column is recomputed with optim.js sgdStep and
// the tests assert it equals the next recorded state.
import { el } from '../core/dom.js';
import { math } from '../core/katex.js';
import { tex, bar, tn, par } from '../ops/texutil.js';
import { sgdStep, initialState } from '../optim.js';
import { injectStyle, SHARED_CSS, badge, fmt } from './sgd-ui.js';

// ---- pure: the window contents ----------------------------------------------------------------------------------------
export function updateWindowData({ lr, momentum = 0, params }) {
  const mom = momentum !== 0;
  const rule = mom
    ? [`v \\leftarrow \\mu\\,v + \\overline{p}`, `p \\leftarrow p - \\eta\\,v`]
    : [`p \\leftarrow p - \\eta\\,\\overline{p}`];
  const blocks = [{
    heading: 'The rule (what optimizer.step() does to every parameter p)', latex: rule,
    html: mom
      ? `<code>p</code> is a parameter, <code>p̄</code> is the gradient <code>backward()</code> left in <code>p.grad()</code>, η is the learning rate and μ the momentum you gave <code>SGD(params, lr, momentum)</code>; <code>v</code> is the parameter’s velocity, zero before the first step.`
      : `<code>p</code> is a parameter, <code>p̄</code> is the gradient <code>backward()</code> left in <code>p.grad()</code>, and η is the learning rate you gave <code>SGD(params, lr)</code>. Read it as: move against the gradient by η times its size.`,
  }];
  const lines = [];
  for (const q of params) {
    const dir = mom ? q.vAfter : q.grad;
    if (mom) lines.push(`v \\leftarrow ${tn(momentum)}\\cdot ${par(q.vBefore ?? 0)} + ${par(q.grad)} = ${tn(q.vAfter)}`);
    lines.push(`${tex(q.name)} \\leftarrow ${tn(q.before)} - ${tn(lr)}\\cdot ${par(dir)} = ${tn(q.before)} ${lr * dir < 0 ? '+' : '-'} ${tn(Math.abs(lr * dir))} = ${tn(q.after)}`);
  }
  blocks.push({ heading: `With our numbers: ${params.map((q) => q.name).join(' and ')}`, latex: lines, html: '' });
  const columns = mom ? ['parameter', 'before', '.grad', 'velocity v', 'step = lr × v', 'after'] : ['parameter', 'before', '.grad', 'step = lr × .grad', 'after'];
  const rows = params.map((q) => (mom
    ? [q.name, q.before, q.grad, q.vAfter, lr * q.vAfter, q.after]
    : [q.name, q.before, q.grad, lr * q.grad, q.after]));
  return { title: 'One update, on paper', kind: 'update', blocks, table: { columns, rows } };
}

// ---- DOM ------------------------------------------------------------------------------------------------------------------
export function updateTable(table) {
  return el('div', { class: 'sgd-scroll' }, el('table', { class: 'sgd-tab' },
    el('caption', {}, 'before − step = after, per parameter (numbers rounded to four digits)'),
    el('thead', {}, el('tr', {}, table.columns.map((c) => el('th', { scope: 'col' }, c)))),
    el('tbody', {}, table.rows.map((r) => el('tr', {}, r.map((v, i) => (i === 0 ? el('th', { scope: 'row' }, el('code', {}, v)) : el('td', {}, fmt(v)))))))));
}

export function renderUpdateWindow(box, data) {
  box.className = 'win update';
  box.replaceChildren(
    el('div', { class: 'win-head' }, el('h3', {}, data.title)),
    el('div', { class: 'win-body' }, data.blocks.map((b) => el('section', { class: 'blk' },
      el('h4', {}, b.heading), b.html ? el('p', { html: b.html }) : null, ...(b.latex ?? []).map((l) => el('div', { class: 'math', html: math(l) }))))),
    updateTable(data.table));
}

const CSS = `.win.update{border-left-color:var(--ok);margin:8px 0 0}
.sgd-upd .win-body{grid-template-columns:minmax(0,.7fr) minmax(0,1.3fr)}
.sgd-upd .math .katex{font-size:1.05em}
@media(max-width:900px){.sgd-upd .win-body{grid-template-columns:minmax(0,1fr)}}
.sgd-upd .pick{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:.2rem 0 .4rem}
.sgd-upd .pick .lbl{min-width:11ch;text-align:center;font-weight:600}
.sgd-upd .loss{margin:.5rem 0 0;font-size:.95rem}`;

// The recorded SGD run with lr 0.05: update u (1-based) goes from runs[0].steps[u-1] to the next recorded state.
export function mountUpdateWindow(host, D) {
  injectStyle('sgd-ui-css', SHARED_CSS); injectStyle('sgd-upd-css', CSS);
  const run = D.files.optim.runs.find((r) => r.id === 'sgd-lr0.05');
  const rows = [...run.steps, run.final];
  const COUNT = 4;
  let u = 1;
  const root = el('div', { class: 'sgd-w sgd-upd' });
  const label = el('span', { class: 'lbl', 'aria-live': 'polite' });
  const prev = el('button', { type: 'button', class: 'btn small', 'data-key': 'upd-prev', onclick: () => go(u - 1) }, '◀ Previous update');
  const next = el('button', { type: 'button', class: 'btn small primary', 'data-key': 'upd-next', onclick: () => go(u + 1) }, 'Next update ▶');
  const box = el('div', { class: 'win update' });
  const loss = el('p', { class: 'loss' });
  root.append(badge('torc', 'The “before” numbers and gradients are TorC’s; the “after” column is recomputed here with the same formula and equals TorC’s next state.'),
    el('div', { class: 'pick' }, prev, label, next), box, loss);
  host.append(root);
  function go(k) {
    u = Math.max(1, Math.min(COUNT, k));
    const a = rows[u - 1], z = rows[u];
    const st = initialState([[a.w], [a.b]]);
    const out = sgdStep(st, [[a.w], [a.b]], [[a.gw], [a.gb]], { lr: run.lr, momentum: run.momentum });
    renderUpdateWindow(box, updateWindowData({ lr: run.lr, params: [
      { name: 'W', before: a.w, grad: a.gw, after: out.params[0][0] },
      { name: 'b', before: a.b, grad: a.gb, after: out.params[1][0] }] }));
    label.textContent = `Update ${u} of ${COUNT}`;
    loss.innerHTML = `Loss before this update: <strong class="mono">${fmt(a.loss)}</strong> → after it: <strong class="mono">${fmt(z.loss)}</strong> (TorC’s forward pass at the new parameters).`;
    prev.setAttribute('aria-disabled', String(u === 1)); next.setAttribute('aria-disabled', String(u === COUNT));
  }
  go(1);
  return { destroy() { root.remove(); } };
}
