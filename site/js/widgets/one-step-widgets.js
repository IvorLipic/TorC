// widgets/one-step-widgets.js — the widget of the lesson one-step: one real training step of the trained digit network on one
// training image, stage by stage, with only what TorC recorded (data/one-step.json, tools/trace/trace_one_step.cpp):
//
//   mountOneStep(host, D)   Back and Next walk through ten stages: the image, the first layer, the second layer, the ten scores,
//                           the probabilities, the gradients of the scores, the gradients of four named parameters, the step, and
//                           the probabilities of the same image after the step. The image stays on the left with the brightest
//                           pixel ringed; the bars on the right show the values of the current stage.
//
// stages(D) is pure and exported for the tests: every readout sentence and every bar comes from the trace. The module touches the
// DOM only in mountOneStep.
import { el, svg } from '../core/dom.js';

const MINUS = '−';
// 3 significant figures, with the minus sign U+2212 (readouts use 2-3 significant figures)
const sig = (n) => (v) => { const s = String(Number(v.toPrecision(n))); return s.startsWith('-') ? MINUS + s.slice(1) : s; };
export const s3 = sig(3);
// the step changes a weight in its third or fourth digit, so the before and after values get 4 significant figures
export const s4 = sig(4);
const DIGIT = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];

// [{title, values, highlight[], compare?, unit, text}] for the ten stages
export function stages(D) {
  const O = D.files['one-step'], f = O.forward, b = O.backward, w = O.weights, a = O.after;
  const [first, outTrue, outRival, bias] = w;
  const label = O.image.label, rival = f.rival, px = first.path;
  const off1 = f.hidden1.post.filter((v) => v === 0).length, off2 = f.hidden2.post.filter((v) => v === 0).length;
  const n1 = f.hidden1.post.length;
  return [
    { title: 'The image', values: null, highlight: [],
      text: `A training image of ${DIGIT[label] === 'eight' ? 'an' : 'a'} ${DIGIT[label]}. The ringed pixel is the brightest. Its grey value is ${px.pixel_255}, so its input is ${s3(px.pixel_value)}.` },
    { title: 'The first layer', values: f.hidden1.post, highlight: [px.unit], unit: 'value after ReLU',
      text: `Each of the ${n1} neurons adds its weighted pixels and its bias. ReLU sets the negative sums to zero, so ${off1} neurons are off. Neuron ${px.unit} has the value ${s3(px.unit_post)}.` },
    { title: 'The second layer', values: f.hidden2.post, highlight: [outTrue.path.unit], unit: 'value after ReLU',
      text: `The second layer does the same with the values of the first layer. Here ${off2} neurons are off. Neuron ${outTrue.path.unit} is the most active, with ${s3(outTrue.path.unit_post)}.` },
    { title: 'The ten scores', values: f.scores, highlight: [label, rival], labels: true, unit: 'score',
      text: `The last layer gives one score for each digit. The ${DIGIT[label]} gets ${s3(f.scores[label])}, and the ${DIGIT[rival]} gets ${s3(f.scores[rival])}.` },
    { title: 'The probabilities', values: f.probabilities, highlight: [label, rival], labels: true, unit: 'probability',
      text: `Softmax turns the scores into probabilities. The ${DIGIT[label]} gets ${s3(f.p_true)}, and the ${DIGIT[rival]} gets ${s3(f.p_rival)}. The surprise is −ln(${s3(f.p_true)}).` },
    { title: 'Backward, the scores', values: b.scores_grad, highlight: [label, rival], labels: true, unit: 'gradient', grad: true,
      text: `The backward pass starts at the scores. The ${DIGIT[label]} gets its probability minus one, ${s3(b.scores_grad[label])}. Every other score gets its probability, so the ${DIGIT[rival]} gets ${s3(b.scores_grad[rival])}.` },
    { title: 'Backward, four parameters', values: w.map((x) => x.grad), highlight: [], unit: 'gradient', grad: true, names: ['first layer', `to the ${DIGIT[label]}`, `to the ${DIGIT[rival]}`, 'bias'],
      text: `The weight from neuron ${outTrue.path.unit} to the score of the ${DIGIT[label]} gets ${s3(outTrue.grad)}. Its weight to the ${DIGIT[rival]} gets ${s3(outRival.grad)}. A weight of the first layer gets ${s3(first.grad)}, and the bias of the ${DIGIT[label]} gets ${s3(bias.grad)}.` },
    { title: 'The step', values: w.map((x) => -x.lr_times_grad), highlight: [], unit: 'change', grad: true, names: ['first layer', `to the ${DIGIT[label]}`, `to the ${DIGIT[rival]}`, 'bias'],
      text: `Each parameter moves by minus the learning rate times its gradient. The weight to the ${DIGIT[label]} goes from ${s4(outTrue.before)} to ${s4(outTrue.after)}. The weight to the ${DIGIT[rival]} goes from ${s4(outRival.before)} to ${s4(outRival.after)}.` },
    { title: 'The same image again', values: a.probabilities, compare: f.probabilities, highlight: [label, rival], labels: true, unit: 'probability',
      text: `After the step, the ${DIGIT[label]} gets ${s3(a.p_true)} in place of ${s3(f.p_true)}. The ${DIGIT[rival]} gets ${s3(a.p_rival)} in place of ${s3(f.p_rival)}.` },
  ];
}

const CSS = `
.os{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.os-top{display:flex;flex-wrap:wrap;gap:.5rem 1rem;align-items:center;margin:0 0 .6rem}
.os-stage{font-weight:600}
.os-row{display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start}
.os-img{width:112px;height:112px;image-rendering:pixelated;border:1px solid var(--border);border-radius:4px;background:#000;flex:none}
.os-imgbox{position:relative;width:112px;height:112px;flex:none}
.os-ring{position:absolute;width:10px;height:10px;border:2px solid var(--warn);border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}
.os-bars{flex:1 1 16rem;min-width:0}
.os-bars svg{display:block;width:100%;max-width:560px;height:auto}
.os-bar{fill:var(--fwd)}
.os-bar.hi{fill:var(--warn)}
.os-bar.neg{fill:var(--bwd)}
.os-was{fill:none;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:3 2}
.os-axis{stroke:var(--border);stroke-width:1}
.os-lbl{fill:var(--muted);font-size:11px}
.os-lbl.hi{fill:var(--fg);font-weight:600}
.os-read{font:.92rem ui-monospace,monospace;margin:.6rem 0 0;max-width:42rem}
.os-key{font-size:.82rem;color:var(--muted);margin:.3rem 0 0}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('os-styles')) return;
  document.head.append(el('style', { id: 'os-styles' }, CSS));
}

function barsSvg(st) {
  const W = 360, H = 170, top = 12, bottom = 26, n = st.values.length;
  const all = [...st.values, ...(st.compare ?? [])];
  const max = Math.max(1e-12, ...all.map((v) => Math.max(0, v))), min = Math.min(0, ...all);
  const span = max - min, y = (v) => top + ((max - v) / span) * (H - top - bottom);
  const gap = n > 12 ? 2 : 6, bw = (W - 8 - gap * (n - 1)) / n;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${st.title}: ${st.unit} of each ${st.names ? 'parameter' : st.labels ? 'digit' : 'neuron'}` });
  root.append(svg('line', { class: 'os-axis', x1: 0, x2: W, y1: y(0), y2: y(0) }));
  st.values.forEach((v, i) => {
    const x = 4 + i * (bw + gap), hi = st.highlight.includes(i);
    const cls = `os-bar${hi && !st.grad ? ' hi' : ''}${st.grad && v < 0 ? ' neg' : ''}`;
    root.append(svg('rect', { class: cls, x, width: bw, y: Math.min(y(v), y(0)), height: Math.max(1, Math.abs(y(v) - y(0))) }));
    if (st.compare) { const c = st.compare[i]; root.append(svg('rect', { class: 'os-was', x, width: bw, y: Math.min(y(c), y(0)), height: Math.max(1, Math.abs(y(c) - y(0))) })); }
    const name = st.names ? st.names[i] : st.labels ? String(i) : hi ? String(i) : null;
    if (name) root.append(svg('text', { class: `os-lbl${hi ? ' hi' : ''}`, x: x + bw / 2, y: H - 8, 'text-anchor': 'middle' }, name));
  });
  return root;
}

export function mountOneStep(host, D) {
  injectStyles();
  const O = D.files['one-step'], S = stages(D);
  let k = 0;
  const img = el('canvas', { class: 'os-img', width: 28, height: 28, role: 'img', 'aria-label': 'The training image, a handwritten digit' });
  const ctx = img.getContext('2d'), im = ctx.createImageData(28, 28);
  O.image.pixels.forEach((p, i) => { im.data.set([p, p, p, 255], 4 * i); });
  ctx.putImageData(im, 0, 0);
  const px = O.weights[0].path;
  const ring = el('span', { class: 'os-ring', style: `left:${((px.pixel_col + 0.5) / 28) * 100}%;top:${((px.pixel_row + 0.5) / 28) * 100}%` });
  const back = el('button', { type: 'button', class: 'btn small' }, 'Back');
  const next = el('button', { type: 'button', class: 'btn small primary' }, 'Next');
  const stageLbl = el('span', { class: 'os-stage' });
  const bars = el('div', { class: 'os-bars' });
  const key = el('p', { class: 'os-key' });
  const read = el('p', { class: 'os-read', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'os' },
    el('div', { class: 'os-top' }, back, next, stageLbl),
    el('div', { class: 'os-row' }, el('div', { class: 'os-imgbox' }, img, ring), bars),
    key, read));
  function show() {
    const st = S[k];
    stageLbl.textContent = `Stage ${k + 1} of ${S.length}. ${st.title}.`;
    back.disabled = k === 0;
    next.disabled = k === S.length - 1;
    bars.replaceChildren(...(st.values ? [barsSvg(st)] : []));
    key.textContent = !st.values ? '' : st.compare ? 'Filled bars: after the step. Dashed outlines: before the step.'
      : st.grad ? 'Red bars are negative, blue bars are positive.' : st.labels ? 'Amber bars: the right digit and the most probable wrong digit.' : 'Each bar is one neuron, counted from zero. The amber bar is the neuron that the text names.';
    read.textContent = st.text;
  }
  back.addEventListener('click', () => { if (k > 0) { k -= 1; show(); } });
  next.addEventListener('click', () => { if (k < S.length - 1) { k += 1; show(); } });
  show();
  return { go: (i) => { k = i; show(); } };
}
