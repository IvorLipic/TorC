// widgets/shape-explorer.js — "Shapes first": steppers for in, out and batch; a to-scale diagram of x · Wᵀ + b = y; the live
// parameter count in·out + out; and a "break it" panel that shows the REAL error message TorC printed for the recorded wrong calls.
//
//   mountShapeExplorer(host, D, ctx) -> { destroy(), state }
//
// Live (JavaScript arithmetic in the browser): shapes, parameter count, the predicted error. Recorded (precomputed from the real
// TorC run, data/linear.json): the exact what() texts and the layer parameter counts, shown with a badge whenever the learner's
// numbers coincide with a recorded call. The shape logic lives in linear-shapes.js (pure, unit-tested in Node).
import { el } from '../core/dom.js';
import { injectStyle, useBaseStyle, sourceBadge, stepperField } from './linear-ui.js';
import { LIMITS, fmtShape, linearShapes, paramCount, predictCall, findRecorded, recordedRuns, recordedLinearCounts, boxPx, sameShape } from './linear-shapes.js';

const CSS = `
.lin-se .se-cols{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:14px 22px}
@media(max-width:900px){.lin-se .se-cols{grid-template-columns:minmax(0,1fr)}}
.lin-se .se-eq{display:flex;flex-wrap:wrap;align-items:flex-start;gap:8px 10px;margin:.6rem 0 .3rem}
.lin-se .se-t{display:flex;flex-direction:column;align-items:center;gap:4px}
.lin-se .se-rect{display:flex;align-items:center;justify-content:center;border:2px solid var(--fwd);border-radius:6px;background:rgba(88,196,221,.09);font:600 .95rem ui-monospace,monospace;color:var(--fg);transition:width .15s,height .15s}
.lin-se .se-rect.w{border-color:var(--ok);background:rgba(131,193,103,.09)}
.lin-se .se-rect.y{border-color:var(--warn);background:rgba(227,179,65,.09)}
.lin-se .se-rect.bad{border-style:dashed;border-color:var(--bwd);background:rgba(252,98,85,.1)}
.lin-se .se-dims{display:flex;flex-direction:column;align-items:center;font:.74rem/1.25 ui-monospace,monospace;color:var(--muted);text-align:center}
.lin-se .se-dims .n{color:var(--fg)}
.lin-se .se-dims .n.bad{color:#ff9d93;font-weight:700}
.lin-se .se-slot{display:flex;align-items:center;justify-content:center;position:relative}
.lin-se .se-T{position:absolute;right:-14px;top:-4px;font-size:1.05rem;line-height:1;color:var(--ok);font-weight:700}
.lin-se .se-t:has(.se-T){margin-right:10px}
.lin-se .se-op{display:flex;align-items:center;margin-top:0}
.lin-se .se-inner{font-size:.9rem;margin:.2rem 0 .5rem}
.lin-se .se-inner.good::before{content:"\\2713  ";color:var(--ok)}
.lin-se .se-inner.bad::before{content:"\\2717  ";color:#ff9d93}
.lin-se table.se-table{width:100%;border-collapse:collapse;font-size:.86rem}
.lin-se .se-table caption{text-align:left;color:var(--muted);padding-bottom:.2rem}
.lin-se .se-table th,.lin-se .se-table td{border-bottom:1px solid var(--border);padding:.22rem .5rem;text-align:left}
.lin-se .se-table th{color:var(--muted);font-weight:600}
.lin-se .se-table td.m{font-family:ui-monospace,monospace;white-space:nowrap}
.lin-se .se-box{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:.65rem .9rem;margin-bottom:12px}
.lin-se .se-box h4{margin:0 0 .35rem;color:var(--fg);font-size:.95rem}
.lin-se .se-total{font:700 1.6rem ui-monospace,monospace;color:var(--fg)}
.lin-se .se-sum{font:.9rem ui-monospace,monospace;color:#c4ccd5;margin:.1rem 0}
.lin-se pre.se-msg{margin:.4rem 0;padding:.5rem .7rem;background:var(--bg);border:1px solid var(--bwd);border-radius:8px;white-space:pre-wrap;word-break:break-word;font-size:.86rem}
.lin-se pre.se-msg .exc{color:#ff9d93}
.lin-se .se-ok{margin:.4rem 0;padding:.5rem .7rem;background:var(--bg);border:1px solid var(--ok);border-radius:8px;font-size:.92rem}
.lin-se .btn.se-break[aria-pressed=true]{border-color:var(--bwd);color:#ff9d93}
.lin-se .btn[aria-pressed=true]:not(.se-break){border-color:var(--fwd);color:var(--fwd)}
`;

export function mountShapeExplorer(host, D, ctx = {}) {
  useBaseStyle();
  injectStyle('lin-shape-explorer-style', CSS);
  const rec = D.files.linear;
  const runs = recordedRuns(rec.shape_errors);
  const counts = recordedLinearCounts(rec.param_counts);
  const st = { inF: 3, outF: 2, B: 2, broken: false, width: 2 };
  const xShape = () => [st.B, st.broken ? st.width : st.inF];

  // ---- controls (built once; never re-rendered, so keyboard focus stays where it is) -------------------------------
  const fields = {
    in: stepperField({ label: '<b>in</b> · features per sample', min: LIMITS.in.min, max: LIMITS.in.max, value: st.inF, key: 'se-in', onChange: (v) => { st.inF = v; render(); } }),
    out: stepperField({ label: '<b>out</b> · neurons', min: LIMITS.out.min, max: LIMITS.out.max, value: st.outF, key: 'se-out', onChange: (v) => { st.outF = v; render(); } }),
    B: stepperField({ label: '<b>B</b> · batch size', min: LIMITS.batch.min, max: LIMITS.batch.max, value: st.B, key: 'se-B', onChange: (v) => { st.B = v; render(); } }),
  };
  const wrongField = stepperField({ label: 'width of the <b>x</b> you feed in', min: LIMITS.in.min, max: LIMITS.in.max, value: st.width, key: 'se-w', onChange: (v) => { st.width = v; render(); } });
  const presetDefs = [
    { label: 'Lesson example · Linear(3, 2)', in: 3, out: 2, B: 2 },
    { label: 'MNIST layer 1 · Linear(784, 32)', in: 784, out: 32, B: 8 },
    { label: 'MNIST layer 2 · Linear(32, 32)', in: 32, out: 32, B: 8 },
    { label: 'MNIST layer 3 · Linear(32, 10)', in: 32, out: 10, B: 8 },
  ];
  const setAll = (inF, outF, B, broken, width) => {
    Object.assign(st, { inF, outF, B, broken });
    if (width !== undefined) st.width = width;
    fields.in.set(inF); fields.out.set(outF); fields.B.set(B); wrongField.set(st.width);
    render();
  };
  const presetBtns = presetDefs.map((p) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', 'data-key': `se-p-${p.in}-${p.out}`, onclick: () => setAll(p.in, p.out, p.B, false) }, p.label));
  const breakBtn = el('button', { type: 'button', class: 'btn small se-break', 'aria-pressed': 'false', 'data-key': 'se-break', onclick: () => {
    st.broken = !st.broken;
    if (st.broken) { st.width = st.inF > 1 ? st.inF - 1 : st.inF + 1; wrongField.set(st.width); }
    render();
  } });
  const runBtns = runs.map((r) => el('button', { type: 'button', class: 'btn small chip-on', 'aria-pressed': 'false', 'data-key': `se-r-${r.id}`,
    title: r.fails ? `Recorded: TorC threw an exception for this call.` : 'Recorded: this call works.',
    onclick: () => setAll(r.in, r.out, r.B, r.fails, r.width) }, `Linear(${r.in}, ${r.out}) on x ${fmtShape([r.B, r.width])} → ${r.fails ? 'error' : 'works'}`));

  // ---- static skeleton -------------------------------------------------------------------------------------------------
  const eq = el('div', { class: 'se-eq', role: 'img' });
  const inner = el('div', { class: 'se-inner' });
  const table = el('table', { class: 'se-table' });
  const paramBox = el('div', { class: 'se-box' });
  const resultBox = el('div', { 'aria-live': 'polite' });
  const status = el('div', { class: 'lin-sr', role: 'status' });
  const wrongWrap = el('div', { style: 'margin:.5rem 0' }, wrongField.root);
  const breakBox = el('div', { class: 'se-box' },
    el('h4', {}, 'Try to break it'),
    el('p', { class: 'lin-note', style: 'margin-top:0' }, 'Feed the layer an input of the wrong width. The messages below are the ones the real TorC printed.'),
    breakBtn, wrongWrap, resultBox,
    el('p', { class: 'lin-note dim', style: 'margin:.7rem 0 .3rem' }, 'Recorded calls (run on the real library; click to load one):'),
    el('div', { class: 'lin-btns' }, runBtns));
  host.replaceChildren(
    el('div', { class: 'lin-head' }, el('h3', {}, 'Shape explorer'), sourceBadge('live', 'Shapes and counts recomputed in your browser · error texts recorded from TorC')),
    el('div', { class: 'se-cols' },
      el('div', {},
        el('div', { class: 'lin-fields' }, fields.in.root, fields.out.root, fields.B.root),
        el('div', { class: 'lin-btns', role: 'group', 'aria-label': 'Preset layers' }, presetBtns),
        eq, inner, el('div', { style: 'overflow-x:auto' }, table)),
      el('div', {}, paramBox, breakBox)),
    status);
  host.classList.add('lin-se');

  // ---- drawing --------------------------------------------------------------------------------------------------------------
  // One tensor of the equation: a fixed-height slot (so the operators line up with the middle of the tallest box), the box, and its dims.
  let slotH = 40;
  function tensor(name, rows, cols, sym, num, cls = '', bad = false, h = null, sup = null) {
    return el('div', { class: 'se-t' },
      el('div', { class: 'se-slot', style: `height:${slotH}px` },
        el('div', { class: `se-rect ${cls}`, style: `width:${boxPx(cols)}px;height:${h ?? boxPx(rows)}px`, 'aria-hidden': 'true' }, name),
        sup ? el('span', { class: 'se-T', title: 'W is transposed inside forward()' }, sup) : null),
      el('div', { class: 'se-dims' }, el('span', {}, sym), el('span', { class: 'n' + (bad ? ' bad' : '') }, num)));
  }
  const op = (t) => el('span', { class: 'lin-op se-op', style: `height:${slotH}px`, 'aria-hidden': 'true' }, t);

  function render() {
    const { inF, outF, B } = st;
    const sh = linearShapes(inF, outF, B);
    const xs = xShape();
    const call = predictCall(inF, outF, xs);
    const recorded = findRecorded(rec.shape_errors, inF, outF, xs);
    const mismatch = call.ok === false;

    // diagram
    slotH = Math.max(boxPx(xs[0]), boxPx(outF), boxPx(B), 22);
    const x = tensor('x', xs[0], xs[1], st.broken ? `(B, ${xs[1] === inF ? 'in' : 'width'})` : '(B, in)', fmtShape(xs) + (mismatch ? ' ✗' : ''), mismatch ? 'bad' : '', mismatch);
    const W = tensor('W', outF, inF, '(out, in)', fmtShape(sh.W), 'w', false, null, 'ᵀ');
    const b = tensor('b', 1, outF, '(out)', fmtShape(sh.b), 'w', false, 22);
    const y = mismatch ? tensor('y', B, outF, '(B, out)', 'not computed', 'bad', true) : tensor('y', B, outF, '(B, out)', fmtShape(sh.y), 'y');
    eq.replaceChildren(x, op('·'), W, op('+'), b, op('='), y);
    eq.setAttribute('aria-label', `x ${fmtShape(xs)} times W transposed plus b ${fmtShape(sh.b)} gives y ${mismatch ? 'not computed (error)' : fmtShape(sh.y)}`);
    if (xs[1] === inF) { inner.className = 'se-inner good'; inner.textContent = `Inner dimension matches: x has ${xs[1]} columns, Wᵀ has ${inF} rows.`; }
    else { inner.className = 'se-inner bad'; inner.textContent = `Inner dimension differs: x has ${xs[1]} columns but Wᵀ has ${inF} rows.`; }

    // shape table
    const rows = [
      ['x', 'input batch', fmtShape(xs), xs[0] * xs[1]],
      ['W', 'weight, one row per neuron', fmtShape(sh.W), outF * inF],
      ['Wᵀ', 'W transposed (used by matmul)', fmtShape(sh.Wt), outF * inF],
      ['b', 'bias, added to every row', fmtShape(sh.b), outF],
      ['y', 'output', mismatch ? '–' : fmtShape(sh.y), mismatch ? '–' : B * outF],
    ];
    table.replaceChildren(
      el('caption', {}, 'All shapes for the current numbers (TorC prints shapes as (rows, columns))'),
      el('thead', {}, el('tr', {}, ['tensor', 'role', 'shape', 'numbers stored'].map((h) => el('th', { scope: 'col' }, h)))),
      el('tbody', {}, rows.map((r) => el('tr', {}, el('th', { scope: 'row' }, r[0]), el('td', {}, r[1]), el('td', { class: 'm' }, r[2]), el('td', { class: 'm' }, String(r[3]))))));

    // parameters
    const p = paramCount(inF, outF);
    const known = counts.get(`${inF},${outF}`);
    paramBox.replaceChildren(
      el('h4', {}, 'Parameters (what training can change)'),
      el('div', { class: 'se-sum' }, `weight: out × in = ${outF} × ${inF} = ${p.weight}`),
      el('div', { class: 'se-sum' }, `bias: out = ${p.bias}`),
      el('div', { class: 'se-total' }, `${p.total}`, el('span', { class: 'lin-note dim', style: 'font:.85rem system-ui;margin-left:.6rem' }, `= in·out + out, in ${p.tensors} tensors`)),
      el('div', { class: 'lin-note dim', style: 'margin-bottom:0' }, `The batch size B = ${B} does not appear: the same ${p.total} numbers serve any batch.`),
      ...(known ? [el('div', { style: 'margin-top:.4rem' }, el('span', { class: 'lin-badge ok' }, `✓ TorC counted ${known.count} parameters for Linear(${inF}, ${outF}) (${known.source}, recorded)`))] : []));

    // result of the call
    breakBtn.setAttribute('aria-pressed', String(st.broken));
    breakBtn.textContent = st.broken ? 'Fix it: feed the right width' : 'Break it: feed the wrong width';
    wrongWrap.hidden = !st.broken;
    resultBox.replaceChildren(...resultNodes(call, recorded, xs, sh));
    for (const [i, pr] of presetDefs.entries()) presetBtns[i].setAttribute('aria-pressed', String(!st.broken && pr.in === inF && pr.out === outF && pr.B === B));
    for (const [i, r] of runs.entries()) runBtns[i].setAttribute('aria-pressed', String(r.in === inF && r.out === outF && sameShape([r.B, r.width], xs)));
    status.textContent = mismatch ? `Linear(${inF}, ${outF}) on x ${fmtShape(xs)}: error, ${call.what}.` : `Linear(${inF}, ${outF}) on x ${fmtShape(xs)}: y has shape ${fmtShape(sh.y)}; ${p.total} parameters.`;
  }

  function resultNodes(call, recorded, xs, sh) {
    const { inF, outF } = st;
    const src = recorded
      ? el('span', { class: 'lin-badge ok' }, '✓ Recorded: this exact call was run on the real TorC')
      : el('span', { class: 'lin-badge warn', title: 'The message template was checked against every recorded call; this particular call was not run.' }, 'Predicted with the recorded message template (this exact call was not run)');
    if (call.ok === true) {
      return [el('div', { class: 'se-ok' }, `Linear(${inF}, ${outF}) on x ${fmtShape(xs)} works: y has shape `, el('b', {}, fmtShape(call.y)), '.'),
        recorded ? el('span', { class: 'lin-badge ok' }, `✓ Recorded: TorC returned y of shape ${fmtShape(recorded.y_shape)} for this exact call`) : el('span', { class: 'lin-badge warn' }, 'Predicted from the shape rule')];
    }
    if (call.ok === null) return [el('p', { class: 'lin-note' }, 'Rank-3 inputs are not covered by this lesson.')];
    const lines = [
      el('div', { class: 'lin-note', style: 'margin:0' }, `Linear(${inF}, ${outF}) on x ${fmtShape(xs)} throws a `, el('code', {}, call.exception), ' whose message is:'),
      el('pre', { class: 'se-msg' }, call.what),
      src];
    if (call.exception === 'ShapeError') {
      lines.push(el('p', { class: 'lin-note' }, `Read the second shape: ${fmtShape([inF, outF])} is `, el('b', {}, 'Wᵀ'), ` = (in, out), not the weight itself, which is stored as ${fmtShape(sh.W)}. matmul sees x ${fmtShape(xs)} and Wᵀ ${fmtShape(sh.Wt)}; ${xs[1]} ≠ ${inF}.`));
    }
    return lines;
  }

  setAll(3, 2, 2, false);
  return { destroy() {}, state: st };
}
