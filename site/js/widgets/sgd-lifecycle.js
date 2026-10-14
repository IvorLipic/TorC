// widgets/sgd-lifecycle.js — "who touches what": a sandbox with the three calls of a training iteration (forward + backward(), step(),
// zero_grad()) acting on the model's parameters W and b, next to the real TorC source of the function you just called.
//
//   mountLifecycle(host, D)
// Gradients come from the engine (the JS port of TorC's autograd), the update from optim.js sgdStep; the first states equal TorC's recorded
// ones (tests). Pure helper: lifecycleReduce(state, action, model, lr) -> {state, note, highlight, tab} (used by the tests).
import { el } from '../core/dom.js';
import { makeModel, sgdStep } from '../optim.js';
import { injectStyle, SHARED_CSS, badge, fmt } from './sgd-ui.js';
import { codePanel } from './sgd-code.js';

export const START = { w: 0, b: 0, gw: null, gb: null, calls: [] };

const par = (x) => (x < 0 ? `(${fmt(x)})` : fmt(x));

// One call. state = {w, b, gw, gb (null = has_grad() false), calls[]}. Never mutates its input.
export function lifecycleReduce(state, action, model, lr = 0.05) {
  const s = { ...state, calls: [...state.calls] };
  const lines = {};
  if (action === 'backward') {
    const e = model.eval(s.w, s.b);
    if (s.gw == null) {
      s.gw = e.gw; s.gb = e.gb;
      s.calls.push('forward + backward()');
      return { state: s, tab: 'backward', highlight: [76, 77, 78, 79, 80], note: `<code>has_grad()</code> was false, so <code>accumulate_grad</code> <strong>copied</strong> the new gradient in: <code>W.grad</code> = ${fmt(s.gw)}, <code>b.grad</code> = ${fmt(s.gb)}. (Loss at W = ${fmt(s.w)}, b = ${fmt(s.b)} is ${fmt(e.loss)}.) <code>.data</code> did not change.` };
    }
    const before = [s.gw, s.gb];
    s.gw += e.gw; s.gb += e.gb;
    s.calls.push('forward + backward()');
    return { state: s, tab: 'backward', highlight: [82, 83], note: `<code>.grad</code> was not empty, so <code>accumulate_grad</code> <strong>added</strong> to it: <code>W.grad</code> = ${par(before[0])} + ${par(e.gw)} = ${fmt(s.gw)}, <code>b.grad</code> = ${par(before[1])} + ${par(e.gb)} = ${fmt(s.gb)}. Nothing was overwritten.` };
  }
  if (action === 'step') {
    s.calls.push('step()');
    if (s.gw == null) return { state: s, tab: 'step', highlight: [20], note: `No parameter has a gradient, so <code>if (!param-&gt;has_grad()) continue;</code> skipped both. <code>W</code> = ${fmt(s.w)}, <code>b</code> = ${fmt(s.b)}: <strong>nothing changed, and TorC raised no error.</strong>` };
    const out = sgdStep(null, [[s.w], [s.b]], [[s.gw], [s.gb]], { lr });
    const [w0, b0] = [s.w, s.b];
    s.w = out.params[0][0]; s.b = out.params[1][0];
    return { state: s, tab: 'step', highlight: [22, 27], note: `<code>step()</code> <strong>read</strong> <code>W.grad</code> = ${fmt(s.gw)} and <code>b.grad</code> = ${fmt(s.gb)} and <strong>rewrote</strong> <code>.data</code>: <code>W</code> ${fmt(w0)} → ${fmt(s.w)}, <code>b</code> ${fmt(b0)} → ${fmt(s.b)}. <code>.grad</code> is untouched: it still holds the old gradient.` };
  }
  if (action === 'zero_grad') {
    s.calls.push('zero_grad()');
    const had = s.gw != null;
    s.gw = null; s.gb = null;
    return { state: s, tab: 'zero', highlight: [34, 111], note: had ? '<code>zero_grad()</code> set <code>has_grad_ = false</code> for every parameter: <code>.grad</code> is empty again. <code>.data</code> did not change.' : '<code>.grad</code> was already empty; <code>zero_grad()</code> had nothing to clear.' };
  }
  throw new Error(`unknown action ${action}`);
}

const CSS = `
.sgd-life .grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px 18px;align-items:start}
.sgd-life .calls{display:flex;flex-wrap:wrap;gap:8px;margin:.2rem 0 .6rem}
.sgd-life .calls .btn{text-align:left}
.sgd-life .calls .btn code{background:none;padding:0;color:inherit}
.sgd-life .calls .btn small{color:var(--muted)}
.sgd-life .state td.empty{color:var(--muted);font-style:italic}
.sgd-life .state td.fresh{color:var(--warn)}
.sgd-life .history{margin:.5rem 0 0;font-size:.85rem;color:var(--muted)}
.sgd-life .history code{background:var(--panel2)}
.sgd-life .tabs2{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 .3rem}
.sgd-life .tabs2 .btn{font-size:.82rem;padding:.15rem .7rem}
.sgd-life .note{min-height:3.4em}
@media(max-width:900px){.sgd-life .grid2{grid-template-columns:minmax(0,1fr)}}
`;

export function mountLifecycle(host, D) {
  injectStyle('sgd-ui-css', SHARED_CSS); injectStyle('sgd-life-css', CSS);
  const O = D.files.optim;
  const model = makeModel(D.scenes.sgd);
  const lr = O.no_zero_grad.training_loop.lr;
  let st = { ...START, calls: [] };
  let view = { tab: 'step', highlight: [], note: '' };

  const root = el('div', { class: 'sgd-w sgd-life' });
  const stateHost = el('div', { class: 'sgd-scroll' }), codeHost = el('div'), history = el('p', { class: 'history' });
  const status = el('div', { class: 'sgd-status note', role: 'status', 'aria-live': 'polite' });
  const act = (action) => { const r = lifecycleReduce(st, action, model, lr); st = r.state; view = { tab: r.tab, highlight: r.highlight, note: r.note, fresh: true }; render(); };
  const callBtn = (key, code, small, action, cls = '') => el('button', { type: 'button', class: `btn ${cls}`, 'data-key': key, onclick: () => act(action) }, el('code', {}, code), el('small', {}, small));
  const calls = el('div', { class: 'calls' },
    callBtn('bw', 'loss.backward()', 'after a fresh forward pass', 'backward', 'primary'),
    callBtn('st', 'optimizer.step()', `lr = ${lr}`, 'step'),
    callBtn('zg', 'optimizer.zero_grad()', 'clears .grad', 'zero_grad'),
    el('button', { type: 'button', class: 'btn small', 'data-key': 'reset', onclick: () => { st = { ...START, calls: [] }; view = { tab: 'step', highlight: [], note: 'Back to W = 0, b = 0 with no gradients.' }; render(); } }, 'Reset'));
  const tabs = el('div', { class: 'tabs2', role: 'group', 'aria-label': 'Which TorC function to show' });
  const tabDefs = [['backward', 'backward() → accumulate_grad'], ['step', 'SGD::step()'], ['zero', 'SGD::zero_grad()']];
  const tabBtns = tabDefs.map(([id, label]) => { const b = el('button', { type: 'button', class: 'btn small', 'data-key': 'tab-' + id, onclick: () => { view = { ...view, tab: id, highlight: [] }; drawCode(); markTabs(); } }, label); return [id, b]; });
  tabs.append(...tabBtns.map(([, b]) => b));
  const left = el('div'), right = el('div');
  left.append(calls, el('h4', { style: 'margin:.3rem 0 .15rem;color:var(--muted);font-size:.85rem' }, 'The tensors, after your calls'), stateHost, history);
  right.append(tabs, codeHost);
  root.append(badge('browser', 'You are driving a sandbox that uses the JavaScript port of TorC’s rules; the code on the right is the real library source (release ' + O.library.ref + '), and the first states match what TorC printed.'),
    status, el('div', { class: 'grid2' }, left, right));
  host.append(root);

  const markTabs = () => tabBtns.forEach(([id, b]) => b.setAttribute('aria-pressed', String(view.tab === id)));
  function drawCode() {
    const ref = O.library.ref;
    if (view.tab === 'backward') codeHost.replaceChildren(codePanel({ key: 'accumulate', ref, highlight: view.highlight }));
    else if (view.tab === 'step') codeHost.replaceChildren(codePanel({ key: 'step', ref, highlight: view.highlight }));
    else codeHost.replaceChildren(codePanel({ key: 'zero', ref, highlight: view.highlight }), codePanel({ key: 'varzero', ref, highlight: view.highlight }));
  }
  function drawState() {
    const cell = (v) => (v == null ? el('td', { class: 'empty' }, 'empty') : el('td', {}, fmt(v)));
    const tx = (a) => '[' + a.map(fmt).join(', ') + ']';
    stateHost.replaceChildren(el('table', { class: 'sgd-tab state' },
      el('caption', {}, 'every tensor of the graph that you could look at'),
      el('thead', {}, el('tr', {}, ['tensor', '.data', '.grad (has_grad)'].map((c) => el('th', { scope: 'col' }, c)))),
      el('tbody', {},
        el('tr', {}, el('th', { scope: 'row' }, el('code', {}, 'W')), el('td', {}, fmt(st.w)), cell(st.gw)),
        el('tr', {}, el('th', { scope: 'row' }, el('code', {}, 'b')), el('td', {}, fmt(st.b)), cell(st.gb)),
        el('tr', {}, el('th', { scope: 'row' }, el('code', {}, 'x')), el('td', {}, tx(model.x)), el('td', { class: 'empty' }, 'none: requires_grad is false')),
        el('tr', {}, el('th', { scope: 'row' }, el('code', {}, 't')), el('td', {}, tx(model.t)), el('td', { class: 'empty' }, 'none: requires_grad is false')))));
    history.replaceChildren('Calls so far: ', ...(st.calls.length ? st.calls.flatMap((c, i) => [i ? ' → ' : '', el('code', {}, c)]) : ['none']));
  }
  function render() {
    status.innerHTML = view.note || 'Press the calls in any order. Try <code>step()</code> first, then <code>backward()</code> twice, then <code>step()</code>.';
    drawState(); drawCode(); markTabs();
  }
  render();
  return { destroy() { root.remove(); }, get state() { return st; } };
}
