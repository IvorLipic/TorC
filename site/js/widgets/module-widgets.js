// widgets/module-widgets.js — the two widgets of the lesson modules. Both show only what TorC recorded
// (data/modules.json, tools/trace/trace_modules.cpp):
//
//   mountModelViewer(host, D)   four Sequential models: each module's parameters (name, shape, size) and output shape, or the
//                               error TorC throws, and the total number of parameters
//   mountCallOrder(host, D)     three orders of forward and backward calls on one model, and what TorC does in each
//
// The module touches the DOM only in mount*.
import { el } from '../core/dom.js';

const shapeText = (s) => `(${s.join(', ')})`;
const count = (n) => n.toLocaleString('en-US');
const MODELS = [['mlp', '784 → 32 → 10'], ['mnist', '784 → 32 → 32 → 10'], ['no-relu', 'no ReLU'], ['mismatch', 'wrong sizes']];
const ORDERS = [['in-order', 'backward, then forward'], ['early-second-forward', 'second forward first'], ['latest-backward', 'backward of the second loss']];

const CSS = `
.mw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.mw-presets{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 .7rem}
.mw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.mw-stack{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.mw-layer{display:grid;grid-template-columns:minmax(9rem,auto) 1fr auto;gap:.2rem 1rem;align-items:baseline;background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.45rem .7rem}
.mw-layer.bad{border-color:var(--bwd)}
.mw-layer.off{opacity:.55}
.mw-name{font:600 .95rem ui-monospace,monospace}
.mw-params{font:.85rem ui-monospace,monospace;color:#c4ccd5}
.mw-out{font:.85rem ui-monospace,monospace;color:var(--muted);text-align:right}
.mw-io{font:.85rem ui-monospace,monospace;color:var(--muted);margin:.2rem 0}
.mw-err{grid-column:1/-1;font:.85rem ui-monospace,monospace;color:#ffb3ab;overflow-wrap:anywhere}
.mw-total{margin:.6rem 0 0;font-size:.95rem}
.mw-code{margin:0;padding:.6rem .8rem;background:#0b0f14;border:1px solid var(--border);border-radius:10px;overflow-x:auto;font-size:.85rem;line-height:1.55}
.mw-code code{background:none;padding:0}
.mw-result{margin:.6rem 0 0;padding:.45rem .7rem;border-radius:8px;font:.9rem ui-monospace,monospace;overflow-wrap:anywhere}
.mw-result.ok{border:1px solid var(--ok);color:var(--ok)}
.mw-result.bad{border:1px solid var(--bwd);color:#ffb3ab;background:#2b1513}
@media(max-width:560px){.mw-layer{grid-template-columns:1fr}.mw-out{text-align:left}}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('mw-styles')) return;
  document.head.append(el('style', { id: 'mw-styles' }, CSS));
}
function presets(label, items, pick) {
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return { root: el('div', { class: 'mw-presets', role: 'group', 'aria-label': label }, ...btns), set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))) };
}

// ======================================================================================================================
export function mountModelViewer(host, D) {
  injectStyles();
  const M = D.files.modules;
  let id = 'mlp';
  const pr = presets('Model', MODELS, (k) => { id = k; draw(); });
  const body = el('div', { 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'mw' }, pr.root, body));

  function draw() {
    pr.set(id);
    const m = M.models.find((x) => x.id === id);
    const items = m.layers.map((l) => {
      const params = l.params.length ? l.params.map((p) => `${p.name} ${shapeText(p.shape)}`).join(', ') : 'no parameters';
      const out = l.out_shape ? `→ ${shapeText(l.out_shape)}` : (l.what ? 'error' : 'not reached');
      return el('li', { class: `mw-layer${l.what ? ' bad' : ''}${l.out === 'not reached' ? ' off' : ''}` },
        el('span', { class: 'mw-name' }, l.label), el('span', { class: 'mw-params' }, params), el('span', { class: 'mw-out' }, out),
        l.what ? el('span', { class: 'mw-err' }, `${l.exception}: ${l.what}`) : null);
    });
    body.replaceChildren(
      el('p', { class: 'mw-io' }, `input x ${shapeText(m.input_shape)}`),
      el('ol', { class: 'mw-stack', 'aria-label': 'The modules in order' }, ...items),
      el('p', { class: 'mw-total' }, `${count(m.total)} parameters in ${m.tensors} tensors.`));
  }
  draw();
  return { draw };
}

// ======================================================================================================================
export function mountCallOrder(host, D) {
  injectStyles();
  const M = D.files.modules;
  let id = 'in-order';
  const pr = presets('Order of calls', ORDERS, (k) => { id = k; draw(); });
  const code = el('pre', { class: 'mw-code', tabindex: '0', role: 'group', 'aria-label': 'The calls' });
  const result = el('p', { class: 'mw-result', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'mw' }, pr.root, code, result));

  function draw() {
    pr.set(id);
    const o = M.orders.find((x) => x.id === id);
    code.replaceChildren(el('code', {}, o.code));
    const ok = o.exception === 'none';
    result.className = `mw-result ${ok ? 'ok' : 'bad'}`;
    result.textContent = ok ? 'backward() runs, and every parameter gets its gradient.' : `${o.exception}: ${o.what}`;
  }
  draw();
  return { draw };
}
