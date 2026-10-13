// widgets/linear-backward.js — "Where do gW, gb and dX come from?" For the gradients TorC computed in the two trace scenes
// (`linear`: loss = mse; `linear-dx`: loss = sum with x trainable), pick one output entry and see which entries of dY, x or W it is built from:
//   gW[k,j] = sum_i dY[i,k] x[i,j]     gb[k] = sum_i dY[i,k]     dX[i,j] = sum_k dY[i,k] W[k,j]
//
//   mountLinearBackward(host, D, ctx) -> { destroy(), state }
//
// The arrays dY, x, W and the library's gradients are PRECOMPUTED by the real TorC run; the sums shown are plain arithmetic done in the
// browser, and a badge compares each one with the number TorC stored in `.grad`.
import { el } from '../core/dom.js';
import { math } from '../core/katex.js';
import { fmt } from '../ops/texutil.js';
import { useBaseStyle, injectStyle, sourceBadge, segmented, matrixEl, clearMarks, markRow, markCol, markCell } from './linear-ui.js';
import { rowsOf, gradW, gradB, gradX, gradWTex, gradBTex, gradXTex } from './linear-math.js';

const CSS = `
.lin-bw .bw-row{display:flex;flex-wrap:wrap;gap:12px 22px;align-items:flex-start;margin:.5rem 0}
.lin-bw .bw-label{flex:0 0 100%;font-size:.8rem;color:var(--muted);margin:.4rem 0 -.2rem;text-transform:uppercase;letter-spacing:.05em}
.lin-bw .bw-story{font-size:.95rem;margin:.3rem 0;min-height:2.9em}
.lin-bw button.lin-out{background:var(--panel2)}
.lin-bw .lin-cell.off{display:flex;align-items:center;justify-content:center;font-size:.75rem;line-height:1.1}
`;

const node = (scene, name) => scene.nodes.find((n) => n.name === name);
const grid = (n) => rowsOf(n.value.values, n.value.shape[0], n.value.shape[1] ?? 1);
const gridG = (n) => rowsOf(n.grad.values, n.grad.shape[0], n.grad.shape[1] ?? 1);

const MODES = {
  linear: { scene: 'linear', label: 'loss = mse(y, t)', note: 'Scene <code>linear</code>: <code>x</code> is data, so only <code>W</code> and <code>b</code> get gradients.' },
  'linear-dx': { scene: 'linear-dx', label: 'loss = sum(y), x trainable', note: 'Scene <code>linear-dx</code>: every output has gradient 1 and <code>x</code> is trainable, so <code>x</code> gets a gradient too.' },
};

export function mountLinearBackward(host, D, ctx = {}) {
  useBaseStyle();
  injectStyle('lin-backward-style', CSS);
  host.classList.add('lin-bw');
  const st = { mode: 'linear', sel: { kind: 'gW', a: 0, b: 2 } };
  const seg = segmented({ label: 'Show gradients for', name: 'bw-mode', value: st.mode, options: Object.entries(MODES).map(([value, m]) => ({ value, label: m.label })), onChange: (v) => { st.mode = v; fixSel(); render(); } });
  const story = el('div', { class: 'bw-story', 'aria-live': 'polite' });
  const paper = el('div', { class: 'lin-paper' });
  const verdict = el('div', { class: 'lin-note', style: 'margin:.2rem 0' });
  const body = el('div');
  const modeNote = el('p', { class: 'lin-note dim', style: 'margin:.2rem 0' });
  host.replaceChildren(
    el('div', { class: 'lin-head' }, el('h3', {}, 'Where the gradients come from'), sourceBadge('pre', 'Gradients precomputed by the real TorC · sums recomputed in your browser')),
    seg.root, modeNote, body, story, paper, verdict);

  function data() {
    const scene = D.scenes[MODES[st.mode].scene];
    const y = node(scene, 'y'), x = node(scene, 'x'), W = node(scene, 'W'), b = node(scene, 'b');
    return { dY: gridG(y), x: grid(x), W: grid(W), libW: gridG(W), libB: b.grad.values.slice(), libX: x.grad ? gridG(x) : null };
  }
  function fixSel() { if (st.sel.kind === 'dX' && !data().libX) st.sel = { kind: 'gW', a: 0, b: 2 }; }

  function render() {
    const d = data();
    const B = d.dY.length, nOut = d.dY[0].length, nIn = d.x[0].length;
    const fW = gradW(d.dY, d.x), fB = gradB(d.dY), fX = gradX(d.dY, d.W);
    const hasX = !!d.libX;
    modeNote.innerHTML = MODES[st.mode].note;
    const num = (m, i, j) => el('span', { class: 'lin-cell' }, fmt(m[i][j]));
    const mDY = matrixEl({ caption: '<code>dY</code> · gradient arriving at <code>y</code> (TorC’s <code>y.grad</code>)', rows: B, cols: nOut, rowTag: (i) => `i=${i + 1}`, colTag: (k) => `k=${k + 1}`, cell: (i, k) => num(d.dY, i, k) });
    const mX = matrixEl({ caption: '<code>x</code> · input batch', rows: B, cols: nIn, rowTag: (i) => `i=${i + 1}`, colTag: (j) => `j=${j + 1}`, cell: (i, j) => num(d.x, i, j) });
    const mW = matrixEl({ caption: '<code>W</code> · weights', rows: nOut, cols: nIn, rowTag: (k) => `k=${k + 1}`, colTag: (j) => `j=${j + 1}`, cell: (k, j) => num(d.W, k, j) });
    const pick = (kind, a, b2) => () => { st.sel = { kind, a, b: b2 }; renderSel(); };
    const outCell = (kind, a, b2, text, label) => {
      const c = el('button', { type: 'button', class: 'lin-cell lin-out', 'data-key': `bw-${kind}-${a}-${b2}`, 'aria-label': label }, text);
      c.addEventListener('click', pick(kind, a, b2));
      c.addEventListener('focus', pick(kind, a, b2));
      c.addEventListener('mouseenter', () => { hover = { kind, a, b: b2 }; renderSel(); });
      c.addEventListener('mouseleave', () => { hover = null; renderSel(); });
      return c;
    };
    const gW = matrixEl({ caption: '<code>W.grad</code> = <code>dYᵀ x</code>', rows: nOut, cols: nIn, rowTag: (k) => `k=${k + 1}`, colTag: (j) => `j=${j + 1}`, cell: (k, j) => outCell('gW', k, j, fmt(fW[k][j]), `W gradient, neuron ${k + 1}, weight ${j + 1}: ${fmt(fW[k][j])}`) });
    const gB = matrixEl({ caption: '<code>b.grad</code> = column sums of <code>dY</code>', rows: 1, cols: nOut, colTag: (k) => `k=${k + 1}`, cell: (_, k) => outCell('gb', k, 0, fmt(fB[k]), `b gradient, neuron ${k + 1}: ${fmt(fB[k])}`) });
    const gX = matrixEl({ caption: hasX ? '<code>x.grad</code> = <code>dY W</code>' : '<code>x.grad</code> · not computed: <code>x</code> is data', rows: B, cols: nIn, rowTag: (i) => `i=${i + 1}`, colTag: (j) => `j=${j + 1}`,
      cell: (i, j) => (hasX ? outCell('dX', i, j, fmt(fX[i][j]), `x gradient, sample ${i + 1}, feature ${j + 1}: ${fmt(fX[i][j])}`) : el('span', { class: 'lin-cell off', title: 'x was created with requires_grad = false' }, 'none')) });
    body.replaceChildren(
      el('div', { class: 'bw-row' }, el('div', { class: 'bw-label' }, 'Given (from the forward and backward pass)'), mDY.root, mX.root, mW.root),
      el('div', { class: 'bw-row' }, el('div', { class: 'bw-label' }, 'Gradients to explain (pick an entry)'), gW.root, gB.root, gX.root));
    ctxRefs = { mDY, mX, mW, gW, gB, gX, d, fW, fB, fX };
    renderSel();
  }

  let ctxRefs = null, hover = null;
  function renderSel() {
    const { mDY, mX, mW, gW, gB, gX, d, fW, fB, fX } = ctxRefs;
    const s = hover ?? st.sel;
    for (const m of [mDY, mX, mW, gW, gB, gX]) clearMarks(m);
    let tex, text, lib, mine, name;
    if (s.kind === 'gW') {
      markCol(mDY, s.a); markCol(mX, s.b); markCell(gW, s.a, s.b, 'hl2'); gW.rowTags[s.a].classList.add('hl'); gW.colTags[s.b].classList.add('hl');
      tex = gradWTex(d.dY, d.x, s.a, s.b); mine = fW[s.a][s.b]; lib = d.libW[s.a][s.b]; name = `W.grad[${s.a + 1},${s.b + 1}]`;
      text = `Weight <code>W[${s.a + 1},${s.b + 1}]</code> belongs to neuron ${s.a + 1} and reads input feature ${s.b + 1}. Its gradient adds up, over the whole batch, (gradient arriving at output ${s.a + 1}) × (the feature it was multiplied with): column ${s.a + 1} of <code>dY</code> against column ${s.b + 1} of <code>x</code>.`;
    } else if (s.kind === 'gb') {
      markCol(mDY, s.a); markCell(gB, 0, s.a, 'hl2'); gB.colTags[s.a].classList.add('hl');
      tex = gradBTex(d.dY, s.a); mine = fB[s.a]; lib = d.libB[s.a]; name = `b.grad[${s.a + 1}]`;
      text = `Bias <code>b[${s.a + 1}]</code> was copied to every row of the batch in the forward pass, so every row sends a gradient back. They are <b>added</b>: the column sum of column ${s.a + 1} of <code>dY</code>. The bias has no feature to multiply with.`;
    } else {
      markRow(mDY, s.a); markCol(mW, s.b); markCell(gX, s.a, s.b, 'hl2'); gX.rowTags[s.a].classList.add('hl'); gX.colTags[s.b].classList.add('hl');
      tex = gradXTex(d.dY, d.W, s.a, s.b); mine = fX[s.a][s.b]; lib = d.libX[s.a][s.b]; name = `x.grad[${s.a + 1},${s.b + 1}]`;
      text = `Input entry <code>x[${s.a + 1},${s.b + 1}]</code> (sample ${s.a + 1}, feature ${s.b + 1}) feeds every neuron through the weights in column ${s.b + 1} of <code>W</code>. Its gradient adds up, over all neurons, (gradient arriving at that neuron's output for this sample) × (the weight that carries the feature into that neuron): row ${s.a + 1} of <code>dY</code> against column ${s.b + 1} of <code>W</code>.`;
    }
    story.innerHTML = text;
    paper.innerHTML = math(tex, true);
    const ok = Math.abs(mine - lib) <= 1e-5 * (1 + Math.abs(lib));
    verdict.innerHTML = '';
    verdict.append(el('span', { class: 'lin-badge ' + (ok ? 'ok' : 'warn') }, ok ? `✓ equals TorC’s stored ${name} = ${fmt(lib)}` : `✗ differs from TorC’s ${name} = ${fmt(lib)}`));
  }

  fixSel();
  render();
  return { destroy() {}, state: st };
}
