// widgets/digit-widgets.js — the widgets of the lessons mnist-mlp and mnist-cnn, with the weights TorC learned
// (data/mnist-mlp.json, tools/trace/trace_mnist_mlp.cpp; data/mnist-cnn.json, tools/trace/trace_mnist_cnn.cpp):
//
//   mountDigitPad(host, D, opts)  draw a digit (or pick a test image); the page runs the model's forward pass on it with TorC's
//                                 weights and shows the probability of each digit. opts = {file, forward} picks the model
//                                 (default: the MLP).
//   mountWeightsGallery(host, D)  the 32 rows of the MLP's first weight matrix, each drawn as a 28 x 28 image
//   mountCnnMaps(host, D)         the 8 kernels of the CNN's first convolution and their feature maps on a test image
//
// The forward passes are ports of nn::Linear (y = x·Wᵀ + b), conv2d (stride, zero padding, no kernel flip), ReLU and Flatten;
// the tests compare them with the logits TorC recorded for 20 test images, and with the CNN's first feature maps. A drawing goes through the same steps as the MNIST images: the ink is cropped, scaled into a 20 x 20 box, and placed so
// that its centre of mass is at the centre of the 28 x 28 image. Pure helpers are exported for the tests; the module touches the
// DOM only in mount*.
import { el } from '../core/dom.js';

// ---- pure helpers ---------------------------------------------------------------------------------------------------------
export function linear(layer, x) {   // layer = {weight_shape: [out, in], weight, bias}; x has `in` values
  const [out, inn] = layer.weight_shape, y = new Array(out);
  for (let o = 0; o < out; o++) {
    let s = layer.bias[o];
    for (let i = 0; i < inn; i++) s += x[i] * layer.weight[o * inn + i];
    y[o] = s;
  }
  return y;
}
const relu = (v) => v.map((a) => (a > 0 ? a : 0));
// Linear, ReLU, Linear, ReLU, Linear on pixels 0..255 (divided by 255, as data::MNISTDataset does)
export function forward(layers, pixels) {
  const x = Array.from(pixels, (p) => p / 255);
  return linear(layers[2], relu(linear(layers[1], relu(linear(layers[0], x)))));
}
// conv2d of one sample: x has C_in·H·W values, layer = {weight_shape: [C_out, C_in, K, K], weight, bias}; returns C_out·H_out·W_out
export function conv2d(x, cin, h, w, layer, stride, pad) {
  const [cout, , k] = layer.weight_shape, ho = Math.floor((h + 2 * pad - k) / stride) + 1, wo = Math.floor((w + 2 * pad - k) / stride) + 1;
  const y = new Array(cout * ho * wo);
  for (let co = 0; co < cout; co++) for (let i = 0; i < ho; i++) for (let j = 0; j < wo; j++) {
    let s = layer.bias[co];
    for (let ci = 0; ci < cin; ci++) for (let a = 0; a < k; a++) {
      const ii = i * stride + a - pad;
      if (ii < 0 || ii >= h) continue;
      for (let b = 0; b < k; b++) {
        const jj = j * stride + b - pad;
        if (jj < 0 || jj >= w) continue;
        s += x[(ci * h + ii) * w + jj] * layer.weight[((co * cin + ci) * k + a) * k + b];
      }
    }
    y[(co * ho + i) * wo + j] = s;
  }
  return { y, ho, wo };
}
// the CNN of examples/mnist_cnn: Conv2d(1, 8, 3, 2, 1), ReLU, Conv2d(8, 16, 3, 2, 1), ReLU, Flatten, Linear(784, 10)
export function cnnFirstMaps(layers, pixels) {
  const x = Array.from(pixels, (p) => p / 255);
  const c = conv2d(x, 1, 28, 28, layers[0], 2, 1);
  return { maps: relu(c.y), h: c.ho, w: c.wo };
}
export function forwardCnn(layers, pixels) {
  const a = cnnFirstMaps(layers, pixels);
  const b = conv2d(a.maps, layers[0].weight_shape[0], a.h, a.w, layers[1], 2, 1);
  return linear(layers[2], relu(b.y));   // Flatten keeps the (C, H, W) order
}
export function softmax(z) {
  const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
export const argmax = (v) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);

// alpha: Float values 0..1 on an n x n grid (row-major). Returns 784 integers 0..255 laid out like an MNIST image, or null if
// there is no ink.
export function preprocess(alpha, n) {
  let x0 = n, y0 = n, x1 = -1, y1 = -1;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (alpha[y * n + x] > 0.05) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (x1 < 0) return null;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, scale = 20 / Math.max(bw, bh);
  const w = Math.max(1, Math.round(bw * scale)), h = Math.max(1, Math.round(bh * scale));
  // area average of the box into a w x h grid
  const small = new Float64Array(w * h);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const sx0 = x0 + (i * bw) / w, sx1 = x0 + ((i + 1) * bw) / w, sy0 = y0 + (j * bh) / h, sy1 = y0 + ((j + 1) * bh) / h;
    let sum = 0, area = 0;
    for (let y = Math.floor(sy0); y < Math.ceil(sy1); y++) for (let x = Math.floor(sx0); x < Math.ceil(sx1); x++) {
      const a = (Math.min(x + 1, sx1) - Math.max(x, sx0)) * (Math.min(y + 1, sy1) - Math.max(y, sy0));
      if (a > 0) { sum += a * alpha[y * n + x]; area += a; }
    }
    small[j * w + i] = area > 0 ? sum / area : 0;
  }
  // centre of mass of the small grid -> the centre of the 28 x 28 image
  let m = 0, cx = 0, cy = 0;
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) { const v = small[j * w + i]; m += v; cx += v * (i + 0.5); cy += v * (j + 0.5); }
  const ox = Math.round(14 - cx / m), oy = Math.round(14 - cy / m);
  const out = new Array(784).fill(0);
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const X = i + ox, Y = j + oy;
    if (X >= 0 && X < 28 && Y >= 0 && Y < 28) out[Y * 28 + X] = Math.min(255, Math.round(255 * small[j * w + i]));
  }
  return out;
}

// ---- styles -------------------------------------------------------------------------------------------------------------------
const CSS = `
.dw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.dw-row{display:flex;flex-wrap:wrap;gap:1rem 1.6rem;align-items:flex-start}
.dw-pad{width:min(280px,100%);aspect-ratio:1;touch-action:none;background:#000;border:1px solid var(--border);border-radius:8px;cursor:crosshair;display:block}
.dw-side{display:flex;flex-direction:column;gap:.5rem;min-width:12rem;flex:1}
.dw-seen{width:84px;height:84px;image-rendering:pixelated;border:1px solid var(--border);border-radius:4px;background:#000}
.dw-lbl{color:var(--muted);font-size:.85rem}
.dw-bars{display:grid;grid-template-columns:auto 1fr auto;gap:2px .5rem;align-items:center;font:.85rem ui-monospace,monospace;max-width:22rem}
.dw-bar{height:.6rem;border-radius:3px;background:var(--fwd);min-width:1px}
.dw-bar.top{background:var(--ok)}
.dw-read{font:.92rem ui-monospace,monospace;margin:.3rem 0 0}
.dw-btns{display:flex;flex-wrap:wrap;gap:6px;margin:.6rem 0 0}
.dw-btns .btn{font-family:ui-monospace,monospace}
.dw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.dw-gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(64px,1fr));gap:6px;max-width:44rem}
.dw-row>.dw-gal{flex:1 1 14rem;min-width:0}
.dw-tile{display:flex;flex-direction:column;align-items:center;gap:2px;font:.72rem ui-monospace,monospace;color:var(--muted)}
.dw-tile canvas{width:100%;max-width:84px;aspect-ratio:1;image-rendering:pixelated;border:1px solid var(--border);border-radius:4px}
.dw-key{display:flex;flex-wrap:wrap;gap:.2rem 1.2rem;font-size:.82rem;color:var(--muted);margin:.5rem 0 0}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('dw-styles')) return;
  document.head.append(el('style', { id: 'dw-styles' }, CSS));
}
function paintGray(canvas, pixels) {   // 784 values 0..255 onto a 28 x 28 canvas, white ink on black
  const ctx = canvas.getContext('2d'), img = ctx.createImageData(28, 28);
  pixels.forEach((p, k) => { img.data[4 * k] = img.data[4 * k + 1] = img.data[4 * k + 2] = p; img.data[4 * k + 3] = 255; });
  ctx.putImageData(img, 0, 0);
}

// ======================================================================================================================
export function mountDigitPad(host, D, { file = 'mnist-mlp', forward: fwd = forward } = {}) {
  injectStyles();
  const M = D.files[file], layers = M.layers, N = 280;
  const pad = el('canvas', { class: 'dw-pad', width: N, height: N, role: 'img', 'aria-label': 'Drawing area. Draw a digit with a mouse, a pen or a finger.' });
  const seen = el('canvas', { class: 'dw-seen', width: 28, height: 28, role: 'img', 'aria-label': 'The 28 by 28 image the model reads' });
  const bars = el('div', { class: 'dw-bars', 'aria-hidden': 'true' });
  const read = el('p', { class: 'dw-read' });                  // follows the drawing: not a live region
  const say = el('p', { class: 'sr', 'aria-live': 'polite' }); // announced on a click and at the end of a stroke
  const clear = el('button', { type: 'button', class: 'btn small' }, 'Clear');
  const samples = M.shown.map((s) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', 'aria-label': `Test image ${s.index}, label ${s.label}` }, `#${s.index}`));
  host.replaceChildren(el('div', { class: 'dw' },
    el('div', { class: 'dw-btns', role: 'group', 'aria-label': 'Test images' }, clear, ...samples),
    el('div', { class: 'dw-row' }, pad,
      el('div', { class: 'dw-side' }, el('span', { class: 'dw-lbl' }, 'What the model reads'), seen, el('span', { class: 'dw-lbl' }, 'Probability of each digit'), bars, read, say))));
  const ctx = pad.getContext('2d');
  const reset = () => { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, N, N); };
  reset();

  function show(pixels, recorded, announce = true) {
    paintGray(seen, pixels ?? new Array(784).fill(0));
    if (!pixels) { bars.replaceChildren(); read.textContent = 'Draw a digit, or pick a test image.'; if (announce) say.textContent = read.textContent; return; }
    const p = softmax(fwd(layers, pixels)), top = argmax(p);
    bars.replaceChildren(...p.flatMap((v, d) => [el('span', {}, String(d)), el('div', { class: `dw-bar${d === top ? ' top' : ''}`, style: `width:${(100 * v).toFixed(1)}%` }), el('span', {}, v.toFixed(3))]));
    read.textContent = recorded
      ? `Test image ${recorded.index}, label ${recorded.label}. The model reads ${top}, with probability ${p[top].toFixed(3)}. TorC also reads ${argmax(recorded.logits)}.`
      : `The model reads ${top}, with probability ${p[top].toFixed(3)}.`;
    if (announce) say.textContent = read.textContent;
  }
  function fromPad(announce = true) {
    const data = ctx.getImageData(0, 0, N, N).data, alpha = new Float64Array(N * N);
    for (let k = 0; k < N * N; k++) alpha[k] = data[4 * k] / 255;
    show(preprocess(alpha, N), null, announce);
  }
  function loadSample(i) {
    const s = M.shown[i];
    samples.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)));
    reset();
    const tmp = el('canvas', { width: 28, height: 28 });
    paintGray(tmp, s.pixels);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, N, N);
    show(s.pixels, s);
  }
  samples.forEach((b, i) => b.addEventListener('click', () => loadSample(i)));
  clear.addEventListener('click', () => { samples.forEach((b) => b.setAttribute('aria-pressed', 'false')); reset(); show(null); });

  let drawing = false, last = null, pending = false;
  const at = (e) => { const r = pad.getBoundingClientRect(); return [((e.clientX - r.left) * N) / r.width, ((e.clientY - r.top) * N) / r.height]; };
  pad.addEventListener('pointerdown', (e) => {
    if (samples.some((b) => b.getAttribute('aria-pressed') === 'true')) { samples.forEach((b) => b.setAttribute('aria-pressed', 'false')); reset(); }
    drawing = true; last = at(e); pad.setPointerCapture(e.pointerId);
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(last[0], last[1], 11, 0, 2 * Math.PI); ctx.fill();
  });
  pad.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const p = at(e);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 22; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(last[0], last[1]); ctx.lineTo(p[0], p[1]); ctx.stroke();
    last = p;
    if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; fromPad(false); }); }
  });
  const end = () => { if (drawing) { drawing = false; fromPad(); } };
  pad.addEventListener('pointerup', end);
  pad.addEventListener('pointercancel', end);
  loadSample(0);
  return { loadSample };
}

// ======================================================================================================================
export function mountWeightsGallery(host, D) {
  injectStyles();
  const W = D.files['mnist-mlp'].layers[0], [rows, cols] = W.weight_shape;
  const tiles = Array.from({ length: rows }, (_, r) => {
    const c = el('canvas', { width: 28, height: 28, 'aria-hidden': 'true' });
    const w = W.weight.slice(r * cols, (r + 1) * cols), m = Math.max(...w.map(Math.abs)) || 1;
    const ctx = c.getContext('2d'), img = ctx.createImageData(28, 28);
    w.forEach((v, k) => {
      const t = Math.min(1, Math.abs(v) / m);
      const [R, G, B] = v >= 0 ? [227, 179, 65] : [88, 196, 221];   // positive: amber, negative: blue
      img.data[4 * k] = Math.round(14 + t * (R - 14)); img.data[4 * k + 1] = Math.round(17 + t * (G - 17)); img.data[4 * k + 2] = Math.round(22 + t * (B - 22)); img.data[4 * k + 3] = 255;
    });
    ctx.putImageData(img, 0, 0);
    return el('div', { class: 'dw-tile' }, c, el('span', {}, `neuron ${r}`));
  });
  host.replaceChildren(el('div', { class: 'dw' },
    el('div', { class: 'dw-gal', role: 'img', 'aria-label': `The ${rows} rows of the first weight matrix, each drawn as a 28 by 28 image. Amber pixels have positive weights, blue pixels have negative weights.` }, ...tiles),
    el('div', { class: 'dw-key', 'aria-hidden': 'true' }, el('span', {}, 'amber: a positive weight, so ink here raises the neuron'), el('span', {}, 'blue: a negative weight, so ink here lowers it'), el('span', {}, 'dark: a weight near zero'))));
}

// ======================================================================================================================
export function mountCnnMaps(host, D) {
  injectStyles();
  const M = D.files['mnist-cnn'], L = M.layers[0], [cout, , k] = L.weight_shape;
  let pick = 0;
  const btns = M.shown.slice(0, 8).map((s, i) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', 'aria-label': `Test image ${s.index}, label ${s.label}`, onclick: () => { pick = i; draw(); } }, `#${s.index}`));
  const img = el('canvas', { class: 'dw-seen', width: 28, height: 28, role: 'img' });
  const tiles = Array.from({ length: cout }, (_, c) => {
    const kc = el('canvas', { width: k, height: k, 'aria-hidden': 'true' }), mc = el('canvas', { width: 14, height: 14, 'aria-hidden': 'true' });
    return { kc, mc, root: el('div', { class: 'dw-tile' }, kc, mc, el('span', {}, `channel ${c}`)) };
  });
  const gal = el('div', { class: 'dw-gal', role: 'img' }, ...tiles.map((t) => t.root));
  host.replaceChildren(el('div', { class: 'dw' },
    el('div', { class: 'dw-btns', role: 'group', 'aria-label': 'Test images' }, ...btns),
    el('div', { class: 'dw-row' }, el('div', { class: 'dw-side' }, el('span', { class: 'dw-lbl' }, 'input'), img), gal),
    el('div', { class: 'dw-key', 'aria-hidden': 'true' }, el('span', {}, 'top of each pair: the 3 × 3 kernel (amber positive, blue negative)'), el('span', {}, 'bottom: its 14 × 14 feature map after relu (brighter is larger, and each map has its own scale)'))));
  // the kernels
  tiles.forEach((t, c) => {
    const w = L.weight.slice(c * k * k, (c + 1) * k * k), m = Math.max(...w.map(Math.abs)) || 1;
    const ctx = t.kc.getContext('2d'), im = ctx.createImageData(k, k);
    w.forEach((v, q) => {
      const s = Math.min(1, Math.abs(v) / m), [R, G, B] = v >= 0 ? [227, 179, 65] : [88, 196, 221];
      im.data.set([14 + s * (R - 14), 17 + s * (G - 17), 22 + s * (B - 22), 255].map(Math.round), 4 * q);
    });
    ctx.putImageData(im, 0, 0);
  });
  function draw() {
    btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === pick)));
    const s = M.shown[pick];
    paintGray(img, s.pixels);
    img.setAttribute('aria-label', `Test image ${s.index}, a handwritten ${s.label}`);
    const { maps, h, w } = cnnFirstMaps(M.layers, s.pixels);
    tiles.forEach((t, c) => {
      const ctx = t.mc.getContext('2d'), im = ctx.createImageData(w, h), top = Math.max(...maps.slice(c * h * w, (c + 1) * h * w)) || 1;
      for (let q = 0; q < h * w; q++) { const g = Math.round(255 * Math.min(1, maps[c * h * w + q] / top)); im.data.set([g, g, g, 255], 4 * q); }
      ctx.putImageData(im, 0, 0);
    });
    gal.setAttribute('aria-label', `The ${cout} kernels of the first convolution and their feature maps for test image ${s.index}. Each map is ${h} by ${w}, half the size of the image, because the stride is two.`);
  }
  draw();
  return { draw };
}
