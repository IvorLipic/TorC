// widgets/why-gears.js — "the chain rule as gears" for the lesson why-gradients.
//
//   mountGears(host, D)   nudge the weight w by delta and watch the change travel w -> p -> e -> L, each stage multiplying by
//                         its own local slope; the product of those slopes is what backward() puts in w.grad.
//
// All values come from landscape-model.js (engine.buildGraph + the ops' own backward rules), so they are the same code as the
// graph lab. At delta = the recorded nudge (data/why-gradients.json `nudge`) the widget says so and shows TorC's float32 numbers.
// The stage table is the text alternative of the picture. Defines functions only (no DOM at import time).
import { el, keepFocus } from '../core/dom.js';
import { fmt } from '../ops/texutil.js';
import { gearsTex } from './landscape-tex.js';
import { makeModel, sameNumber } from './landscape-model.js';
import { injectStyles, uid, slider, eqLine, badgeEl, BADGE } from './landscape1d.js';

const NAMES = { w: 'the knob', p: 'prediction', e: 'error', L: 'loss' };
const signed = (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + fmt(Math.abs(v));

export function mountGears(host, D) {
  const files = D.files['why-gradients'], m = makeModel(D.scenes.why);
  injectStyles();
  injectGearStyles();
  host.classList.add('ls-host');
  const key = uid('g');
  let delta = files.nudge.delta;

  const sl = slider({
    key: `${key}:d`, label: 'Nudge the knob by <code>δ</code> (how far we turn it)', min: -0.5, max: 0.5, step: 0.01, value: delta,
    show: (v) => `δ = ${signed(v)}`, valueText: (v) => `delta ${fmt(v)}: the loss changes by ${fmt(m.nudge(m.w0, v).actualDL)}`,
    onInput: (v) => { delta = v; update(); }, hint: 'Start tiny. Then push δ out and watch the prediction drift from what really happens.',
  });
  const strip = el('div', { class: 'wg-strip', role: 'group', 'aria-label': 'The change travelling from w to L' });
  const eqs = el('div', { class: 'ls-eqs' });
  const tableWrap = el('div', { class: 'ls-tablewrap', tabindex: 0, role: 'region', 'aria-label': 'Stage table (scrollable)' });
  const badge = badgeEl();
  const foot = el('p', { class: 'ls-foot' }, 'Changes and slopes are recomputed in your browser by the engine behind the graph lab (a JavaScript port of TorC’s rules, tested against the real run).');
  host.replaceChildren(el('div', { class: 'wg' }, el('div', { class: 'ls-panel' }, sl.row, strip), el('div', { class: 'ls-panel' }, eqs, tableWrap, badge)), foot);

  function update() {
    keepFocus(() => {
      const n = m.nudge(m.w0, delta);
      // picture: four cards joined by "x slope" connectors
      const cards = [{ name: 'w', before: m.w0, after: m.w0 + delta, change: delta }, ...n.stages.map((s) => ({ name: s.name, before: s.before, after: s.after, change: s.change }))];
      const parts = [];
      cards.forEach((c, i) => {
        if (i) parts.push(el('div', { class: 'wg-arrow' }, el('span', { class: 'wg-times' }, `× ${signed(n.stages[i - 1].slope).replace('+', '')}`), el('span', { class: 'wg-lab' }, 'local slope')));
        parts.push(el('div', { class: 'wg-card' }, el('div', { class: 'wg-name' }, el('b', {}, c.name), ' ', el('span', {}, NAMES[c.name])),
          el('div', { class: 'wg-vals' }, `${fmt(c.before)} → ${fmt(c.after)}`), el('div', { class: 'wg-change' }, `change ${signed(c.change)}`)));
      });
      strip.replaceChildren(...parts);
      eqs.replaceChildren(...gearsTex(m, n).map(eqLine));
      tableWrap.replaceChildren(el('table', { class: 'ls-table' }, el('caption', {}, `Turning w by ${signed(delta)} from w = ${fmt(m.w0)}. predicted = local slope × the change in the stage before; actual = what the engine computes.`),
        el('thead', {}, el('tr', {}, ['stage', 'from', 'slope', 'predicted', 'actual'].map((c) => el('th', { scope: 'col' }, c)))),
        el('tbody', {}, n.stages.map((s) => el('tr', {}, [s.name, s.from, fmt(s.slope), signed(s.predicted), signed(s.change)].map((v) => el('td', {}, v)))))));
      const rec = files.nudge;
      const onRecord = Math.abs(delta - rec.delta) < 1e-9 && Math.abs(m.w0 - rec.w0) < 1e-9;
      badge.set(onRecord && sameNumber(n.actualDL, rec.actual_dL, 1e-4)
        ? BADGE.match(`δ = ${fmt(rec.delta)} is the nudge TorC recorded: its float32 run printed ΔL = ${fmt(rec.actual_dL)}, the browser agrees`)
        : BADGE.browser('Recomputed in your browser (no TorC recording for this δ)'));
    }, host);
  }
  update();
  return { update };
}

const GEARS_CSS = `
.wg{display:grid;gap:10px}
.wg-strip{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:6px;align-items:center;margin-top:.4rem}
.wg-card{border:1px solid var(--border);background:var(--bg);border-radius:12px;padding:.5rem .6rem;min-width:0;text-align:center}
.wg-name{font-size:.82rem;color:var(--muted)}.wg-name b{color:var(--fg);font-size:1.05rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.wg-vals{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92rem;margin:.15rem 0;overflow-wrap:anywhere}
.wg-change{font-size:.84rem;color:var(--fwd)}
.wg-arrow{display:flex;flex-direction:column;align-items:center;font-size:.8rem;color:var(--muted)}
.wg-arrow::after{content:"→";color:var(--bwd);font-size:1.2rem;line-height:1}
.wg-times{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1rem;color:var(--fg);font-weight:700}
.wg-lab{font-size:.7rem}
@media(max-width:700px){
  .wg-strip{grid-template-columns:1fr}
  .wg-arrow{flex-direction:row;gap:.5rem;justify-content:center}
  .wg-arrow::after{content:"↓"}
}
`;
function injectGearStyles() {
  if (typeof document === 'undefined' || document.getElementById('wg-styles')) return;
  const s = document.createElement('style');
  s.id = 'wg-styles';
  s.textContent = GEARS_CSS;
  document.head.append(s);
}
