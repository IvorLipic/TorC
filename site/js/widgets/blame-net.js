// widgets/blame-net.js — the small network of the lesson backprop, forward and then backward, one stage per click (brief §6
// "blame flows back along the weights"). Two inputs, two hidden neurons with ReLU, one output, one sample, loss = miss squared.
// Every number is read from the real TorC run (data/autograd_trace.json, scene `net`): netValues() only reshapes it, and
// site/test/backprop.test.mjs checks the rules the captions state (a neuron's gradient is the output's gradient times its output
// weight, a weight's gradient is its neuron's gradient times its input) against the recorded gradients. The DOM is touched only in
// mountBlameNet.
import { el, svg } from '../core/dom.js';

const v = (scene, name) => scene.nodes.find((n) => n.name === name);
const vals = (scene, name) => v(scene, name).value.values;
const grads = (scene, name) => v(scene, name).grad?.values ?? null;

// The scene as the widget shows it. W1[i * 2 + j] is the weight from input i to neuron j (x @ W1).
export function netValues(scene) {
  return {
    x: vals(scene, 'x'), W1: vals(scene, 'W1'), b1: vals(scene, 'b1'), h: vals(scene, 'h'), a: vals(scene, 'a'),
    W2: vals(scene, 'W2'), b2: vals(scene, 'b2')[0], o: vals(scene, 'o')[0], t: vals(scene, 't')[0], miss: vals(scene, 'd')[0], L: vals(scene, 'L')[0],
    gO: grads(scene, 'o')[0], gH: grads(scene, 'h'), gW1: grads(scene, 'W1'), gB1: grads(scene, 'b1'), gW2: grads(scene, 'W2'), gB2: grads(scene, 'b2')[0],
  };
}

// 2 to 3 significant figures, with a true minus sign
export const fmt3 = (x) => { const s = String(Number(Number(x).toPrecision(3))); return s.startsWith('-') ? `−${s.slice(1)}` : s; };

export const STAGES = [
  { title: 'The network', caption: 'Each arrow carries its input times the weight on the arrow. Each neuron adds what it gets and its bias, then applies ReLU.' },
  { title: 'Forward pass', caption: 'The forward pass computes both neurons, the prediction and the loss. The loss is the miss squared.' },
  { title: 'Backward pass, the output', caption: 'The backward pass starts at the loss. The gradient of the prediction is twice the miss.' },
  { title: 'Backward pass, the neurons', caption: 'Each neuron gets the gradient of the prediction times its output weight. ReLU passes it on, because both neurons are on.' },
  { title: 'Backward pass, the weights', caption: 'Each weight gets the gradient of its neuron times its input. The labels on the arrows now show these gradients.' },
];

const P = { in: [[42, 96], [42, 228]], hid: [[180, 96], [180, 228]], out: [306, 140] };
const R = 22;

let seq = 0;

export function mountBlameNet(host, D) {
  injectStyles();
  const N = netValues(D.scenes.net);
  const id = `bn${++seq}`;
  let k = 0;
  const pic = el('div', { class: 'bn-pic' });
  const stageTitle = el('p', { class: 'bn-title', id: `${id}-t` });
  const caption = el('p', { class: 'bn-cap', 'aria-live': 'polite' });
  const back = el('button', { type: 'button', class: 'btn', onclick: () => go(k - 1) }, '◀ Back');
  const next = el('button', { type: 'button', class: 'btn primary', onclick: () => go(k + 1) }, 'Next ▶');
  const restart = el('button', { type: 'button', class: 'btn', onclick: () => go(0) }, 'Start again');
  host.replaceChildren(el('div', { class: 'bn-ctl' }, back, next, restart, el('span', { class: 'bn-count', 'aria-hidden': 'true' })), stageTitle, pic, caption);

  function go(i) {
    k = Math.max(0, Math.min(STAGES.length - 1, i));
    draw();
  }

  function draw() {
    back.disabled = k === 0; next.disabled = k === STAGES.length - 1;
    host.querySelector('.bn-count').textContent = `${k + 1} of ${STAGES.length}`;
    stageTitle.textContent = STAGES[k].title;
    caption.textContent = STAGES[k].caption;
    const s = svg('svg', { viewBox: '0 0 380 314', class: 'bn-svg', role: 'img', 'aria-labelledby': `${id}-t`, 'aria-describedby': `${id}-d` });
    const maxG = Math.max(...N.gW1.map(Math.abs), ...N.gW2.map(Math.abs));
    const weightsShowGrad = k >= 4;
    const labels = [];   // drawn last, over the arrows, with a halo in the page colour
    // an arrow from node a to node b, labelled at the share t of its length, `side` px to the left (+) or right (-) of its direction
    const edge = (a, b, w, g, t, side) => {
      const width = weightsShowGrad ? 1 + (5 * Math.abs(g)) / maxG : 1.6;
      s.append(svg('line', { x1: a[0] + R, y1: a[1], x2: b[0] - R, y2: b[1], class: `bn-edge${weightsShowGrad ? ' grad' : ''}`, 'stroke-width': width.toFixed(2) }));
      const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
      const px = a[0] + t * dx + (side * dy) / len, py = a[1] + t * dy - (side * dx) / len + 4;
      labels.push(svg('text', { x: px.toFixed(1), y: py.toFixed(1), class: weightsShowGrad ? 'bn-lab g' : 'bn-lab w', 'text-anchor': 'middle' }, weightsShowGrad ? fmt3(g) : fmt3(w)));
    };
    // first layer: the straight arrows labelled on their outer side, the crossing ones near their start
    edge(P.in[0], P.hid[0], N.W1[0], N.gW1[0], 0.5, 10);
    edge(P.in[1], P.hid[1], N.W1[3], N.gW1[3], 0.5, -10);
    edge(P.in[0], P.hid[1], N.W1[1], N.gW1[1], 0.3, -12);
    edge(P.in[1], P.hid[0], N.W1[2], N.gW1[2], 0.3, 12);
    edge(P.hid[0], P.out, N.W2[0], N.gW2[0], 0.45, 12);
    edge(P.hid[1], P.out, N.W2[1], N.gW2[1], 0.45, -12);
    // a node with its name and extra lines on its outer side (dir -1 above, +1 below)
    const node = (p, label, inner, cls, dir, extra = []) => {
      s.append(svg('circle', { cx: p[0], cy: p[1], r: R, class: `bn-node ${cls}` }));
      if (inner !== null) s.append(svg('text', { x: p[0], y: p[1] + 4, class: 'bn-val', 'text-anchor': 'middle' }, inner));
      [[label, 'bn-name'], ...extra].forEach(([text, c], r) => {
        const y = dir < 0 ? p[1] - R - 7 - 14 * r : p[1] + R + 15 + 14 * r;
        s.append(svg('text', { x: p[0], y, class: c, 'text-anchor': 'middle' }, text));
      });
    };
    N.x.forEach((x, i) => node(P.in[i], i === 0 ? 'input 1' : 'input 2', fmt3(x), 'in', i === 0 ? -1 : 1));
    N.a.forEach((a, j) => {
      const extra = [[weightsShowGrad ? `bias grad ${fmt3(N.gB1[j])}` : `bias ${fmt3(N.b1[j])}`, weightsShowGrad ? 'bn-sub g' : 'bn-sub']];
      if (k >= 3) extra.push([`grad ${fmt3(N.gH[j])}`, 'bn-sub g']);
      node(P.hid[j], j === 0 ? 'neuron 1' : 'neuron 2', k >= 1 ? fmt3(a) : null, 'hid', j === 0 ? -1 : 1, extra);
    });
    node(P.out, 'prediction', k >= 1 ? fmt3(N.o) : null, 'out', -1);
    const outLines = [];
    if (k >= 2) outLines.push([`grad ${fmt3(N.gO)}`, 'bn-sub g']);
    outLines.push([`target ${fmt3(N.t)}`, 'bn-sub']);
    if (k >= 1) outLines.push([`miss ${fmt3(N.miss)}`, 'bn-sub'], [`loss ${fmt3(N.L)}`, 'bn-sub']);
    outLines.forEach(([text, c], r) => s.append(svg('text', { x: P.out[0], y: P.out[1] + R + 15 + 14 * r, class: c, 'text-anchor': 'middle' }, text)));
    s.append(...labels);
    pic.replaceChildren(s, el('p', { class: 'sr', id: `${id}-d` }, describe()));
  }

  function describe() {
    const w = N.W1.map(fmt3).join(', ');
    const parts = [`Inputs ${fmt3(N.x[0])} and ${fmt3(N.x[1])}. First-layer weights ${w}. Output weights ${fmt3(N.W2[0])} and ${fmt3(N.W2[1])}.`];
    if (k >= 1) parts.push(`Neurons ${fmt3(N.a[0])} and ${fmt3(N.a[1])}. Prediction ${fmt3(N.o)}, target ${fmt3(N.t)}, miss ${fmt3(N.miss)}, loss ${fmt3(N.L)}.`);
    if (k >= 2) parts.push(`Gradient of the prediction ${fmt3(N.gO)}.`);
    if (k >= 3) parts.push(`Gradients of the neurons ${fmt3(N.gH[0])} and ${fmt3(N.gH[1])}.`);
    if (k >= 4) parts.push(`Gradients of the first-layer weights ${N.gW1.map(fmt3).join(', ')}.`);
    return parts.join(' ');
  }

  draw();
  return { go };
}

const CSS = `
.bn-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.bn-count{color:var(--muted);font-size:.85rem;margin-left:.4rem}
.bn-title{font-weight:600;margin:.6rem 0 0}
.bn-pic{max-width:560px}
.bn-svg{display:block;width:100%;height:auto}
.bn-svg text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.bn-edge{stroke:#566172}
.bn-edge.grad{stroke:var(--bwd);opacity:.75}
.bn-node{fill:#10151c;stroke:#4a5565;stroke-width:1.5}
.bn-node.out{stroke:var(--fwd)}
.bn-val{font-size:12px;fill:var(--fwd)}
.bn-name{font-size:11px;fill:var(--muted);font-family:system-ui,sans-serif!important}
.bn-lab{font-size:11px;paint-order:stroke;stroke:var(--bg);stroke-width:4px;stroke-linejoin:round}
.bn-lab.w{fill:var(--fg)}
.bn-lab.g,.bn-sub.g{fill:var(--bwd)}
.bn-sub{font-size:11px;fill:var(--muted)}
.bn-cap{margin:.3rem 0 0;max-width:44rem}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('bn-styles')) return;
  document.head.append(el('style', { id: 'bn-styles' }, CSS));
}
