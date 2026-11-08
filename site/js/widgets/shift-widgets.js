// widgets/shift-widgets.js — the two widgets of the lesson convolution about a digit that moves (brief C8). Both use test images
// that the page moves itself, by whole pixels with zero fill, exactly as tools/trace/trace_shift.cpp does. They never go through
// the digit pad, which would put a drawing back in the centre and undo the move.
//
//   mountShiftMaps(host, D)       test image 0 moved right by 0 to 4 pixels, and one feature map of the trained CNN's first
//                                 convolution (after relu) for the original and for the moved image, drawn cell by cell. The
//                                 readout says whether all eight maps equal the original maps moved by whole cells.
//   mountShiftAccuracy(host, D)   how many percentage points of correctly read test images the MLP and the CNN lose after a move,
//                                 compared with no move, for one direction (data/shift.json, measured by TorC on all 10,000 test
//                                 images). Only the loss is drawn, so the page does not show which model is better without a move
//                                 (the lesson mnist-cnn asks that as a prediction).
//
// The maps are computed here with the conv2d port of digit-widgets.js and the weights of data/mnist-cnn.json; the tests compare
// them with the maps TorC computed (shift.json feature_maps.original, feature_maps.shifted and not_a_multiple). Pure helpers are
// exported for the tests; the module touches the DOM only in mount*.
import { el } from '../core/dom.js';
import { cnnFirstMaps } from './digit-widgets.js';
import { createPlot, srTable } from './plot.js';

// ---- pure helpers ---------------------------------------------------------------------------------------------------------
// a 28 x 28 image moved by (dx, dy) whole pixels (right and down are positive); pixels that leave are dropped, the border is 0
export function shiftPixels(pixels, dx, dy, side = 28) {
  const out = new Array(side * side).fill(0);
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
    const X = x + dx, Y = y + dy;
    if (X >= 0 && X < side && Y >= 0 && Y < side) out[Y * side + X] = pixels[y * side + x];
  }
  return out;
}
// the largest difference between map b and map a moved right by `cells` cells, over the cells that exist in both
// (c channels of h x w, row-major): b[ch][i][j + cells] against a[ch][i][j], as trace_shift.cpp compares them
export function movedDiff(a, b, c, h, w, cells) {
  let d = 0, compared = 0;
  for (let ch = 0; ch < c; ch++) for (let i = 0; i < h; i++) for (let j = 0; j + cells < w; j++) {
    d = Math.max(d, Math.abs(b[(ch * h + i) * w + j + cells] - a[(ch * h + i) * w + j]));
    compared++;
  }
  return { d, compared };
}
// the first-layer maps of image `pixels` moved right by `px` pixels, and how they compare with the maps of the original
export function shiftReport(layers, pixels, px, stride) {
  const base = cnnFirstMaps(layers, pixels), moved = cnnFirstMaps(layers, shiftPixels(pixels, px, 0));
  const c = layers[0].weight_shape[0], { h, w } = base;
  const whole = px % stride === 0, cells = whole ? px / stride : null;
  const still = movedDiff(base.maps, moved.maps, c, h, w, 0).d;
  const oneCell = movedDiff(base.maps, moved.maps, c, h, w, 1).d;
  const exact = whole ? movedDiff(base.maps, moved.maps, c, h, w, cells) : null;
  return { base: base.maps, moved: moved.maps, c, h, w, whole, cells, still, oneCell, exact };
}
// two significant figures, for a readout
const sig2 = (v) => (v === 0 ? '0' : String(Number(v.toPrecision(2))));

const CSS = `
.sw{background:var(--panel);border:1px solid var(--border);border-radius:var(--r);padding:.8rem 1rem 1rem;margin:14px 0}
.sw-ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 .6rem}
.sw-lbl{color:var(--muted);font-size:.9rem;margin-right:.2rem}
.sw .btn[aria-pressed=true]{border-color:var(--fwd);color:var(--fwd);background:rgba(88,196,221,.1)}
.sw-row{display:flex;flex-wrap:wrap;gap:.8rem 1.4rem;align-items:flex-start}
.sw-fig{display:flex;flex-direction:column;gap:.25rem;margin:0}
.sw-fig figcaption{font-size:.82rem;color:var(--muted);max-width:150px}
.sw-img{width:112px;height:112px;image-rendering:pixelated;border:1px solid var(--border);border-radius:4px;background:#000}
.sw-map{width:150px;height:150px;border:1px solid var(--border);border-radius:4px;background:#000}
.sw-read{font-size:.95rem;margin:.7rem 0 0;max-width:42rem}
.sw-plot{max-width:620px}
.sw-lab-mlp{fill:var(--warn)}
.sw-lab-cnn{fill:var(--fwd)}
.sw-line-mlp{stroke:var(--warn)!important}
.sw-line-cnn{stroke:var(--fwd)!important}
.sw-dot-mlp{fill:var(--warn)}
.sw-dot-cnn{fill:var(--fwd)}
@media(max-width:560px){.sw-img{width:96px;height:96px}.sw-map{width:128px;height:128px}}
`;
function injectStyles() {
  if (typeof document === 'undefined' || document.getElementById('sw-styles')) return;
  document.head.append(el('style', { id: 'sw-styles' }, CSS));
}
function presets(label, items, pick) {
  const id = `swl${++seq}`;
  const btns = items.map(([key, text]) => el('button', { type: 'button', class: 'btn small', 'aria-pressed': 'false', onclick: () => pick(key) }, text));
  return {
    root: el('div', { class: 'sw-ctl', role: 'group', 'aria-labelledby': id }, el('span', { class: 'sw-lbl', id }, label), ...btns),
    set: (key) => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(items[i][0] === key))),
    btns,
  };
}
let seq = 0;
function paintGray(canvas, pixels) {
  const ctx = canvas.getContext('2d'), img = ctx.createImageData(28, 28);
  pixels.forEach((p, k) => { img.data[4 * k] = img.data[4 * k + 1] = img.data[4 * k + 2] = p; img.data[4 * k + 3] = 255; });
  ctx.putImageData(img, 0, 0);
}
// one h x w map, cell by cell with a dark gap between the cells, so that a move by one cell can be counted
function paintCells(canvas, values, h, w, top) {
  const ctx = canvas.getContext('2d'), cw = canvas.width / w, ch = canvas.height / h;
  ctx.fillStyle = '#0b0e12';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    const g = Math.round(255 * Math.min(1, values[i * w + j] / top));
    ctx.fillStyle = `rgb(${g},${g},${g})`;
    ctx.fillRect(j * cw + 1, i * ch + 1, cw - 2, ch - 2);
  }
}

// ======================================================================================================================
export function mountShiftMaps(host, D) {
  injectStyles();
  const S = D.files.shift.feature_maps, layers = D.files['mnist-cnn'].layers, stride = S.stride;
  const pixels = D.files['mnist-cnn'].shown.find((s) => s.index === S.index).pixels;
  const shifts = D.files.shift.setup.shifts;
  let px = shifts[0], channel = 0;
  const reports = Object.fromEntries(shifts.map((s) => [s, shiftReport(layers, pixels, s, stride)]));
  const R0 = reports[shifts[0]];
  // start on the channel whose map is brightest, so the pattern is easy to see
  let best = -1;
  for (let c = 0; c < R0.c; c++) { const m = Math.max(...R0.base.slice(c * R0.h * R0.w, (c + 1) * R0.h * R0.w)); if (m > best) { best = m; channel = c; } }
  const pm = presets('Move right by', shifts.map((s) => [s, s === 1 ? '1 pixel' : `${s} pixels`]), (s) => { px = s; draw(true); });
  const pc = presets('Kernel', Array.from({ length: R0.c }, (_, c) => [c, String(c)]), (c) => { channel = c; draw(true); });
  const imgA = el('canvas', { class: 'sw-img', width: 28, height: 28, role: 'img', 'aria-label': `Test image ${S.index}, a handwritten ${S.label}` });
  const imgB = el('canvas', { class: 'sw-img', width: 28, height: 28, role: 'img' });
  const mapA = el('canvas', { class: 'sw-map', width: 14 * 10, height: 14 * 10, role: 'img' });
  const mapB = el('canvas', { class: 'sw-map', width: 14 * 10, height: 14 * 10, role: 'img' });
  const capB = el('figcaption', {});
  const read = el('p', { class: 'sw-read' });
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  const fig = (canvas, cap) => el('figure', { class: 'sw-fig' }, canvas, typeof cap === 'string' ? el('figcaption', {}, cap) : cap);
  host.replaceChildren(el('div', { class: 'sw' }, pm.root, pc.root,
    el('div', { class: 'sw-row' }, fig(imgA, 'the original image'), fig(imgB, capB), fig(mapA, 'its feature map, after the bias and ReLU'), fig(mapB, 'the feature map of the moved image')),
    read, say));
  paintGray(imgA, pixels);
  function draw(announce = false) {
    pm.set(px); pc.set(channel);
    const R = reports[px], n = R.h * R.w, a = R.base.slice(channel * n, (channel + 1) * n), b = R.moved.slice(channel * n, (channel + 1) * n);
    const top = Math.max(...a, ...b) || 1;
    paintGray(imgB, shiftPixels(pixels, px, 0));
    capB.textContent = px === 0 ? 'the same image, not moved' : `the image moved ${px === 1 ? '1 pixel' : `${px} pixels`} to the right`;
    imgB.setAttribute('aria-label', capB.textContent);
    paintCells(mapA, a, R.h, R.w, top);
    paintCells(mapB, b, R.h, R.w, top);
    mapA.setAttribute('aria-label', `The feature map of kernel ${channel} for the original image, ${R.h} by ${R.w} cells`);
    mapB.setAttribute('aria-label', `The feature map of kernel ${channel} for the moved image, ${R.h} by ${R.w} cells`);
    if (px === 0) read.textContent = 'The image is not moved, so the two maps are the same. Choose a move.';
    else if (R.whole && R.exact.d === 0) read.textContent = `A move of ${px} pixels is ${R.cells === 1 ? 'one stride' : `${R.cells} strides`}. Each of the ${R.c} maps equals the original map, moved ${R.cells === 1 ? 'one cell' : `${R.cells} cells`} to the right.`;
    else if (R.whole) read.textContent = `A move of ${px} pixels is ${R.cells} strides. The maps move ${R.cells} cells, but they differ by up to ${sig2(R.exact.d)} where ink left the image.`;
    else read.textContent = `A move of ${px === 1 ? '1 pixel' : `${px} pixels`} is not a whole number of strides. The maps change. They differ from the original by up to ${sig2(R.still)}, and from the original moved one cell by up to ${sig2(R.oneCell)}.`;
    if (announce) say.textContent = read.textContent;
  }
  draw();
  return { pick: (s) => { px = s; draw(true); } };
}

// ======================================================================================================================
const DIR_WORD = { left: 'to the left', right: 'to the right', up: 'up', down: 'down' };
export function shiftSeries(shift, dir) {
  const A = shift.accuracy;
  return ['mlp', 'cnn'].map((m) => ({ id: m, label: m.toUpperCase(), points: A[m][dir].map((e) => [e.shift, 100 * (A[m][dir][0].accuracy - e.accuracy)]) }));
}
export function mountShiftAccuracy(host, D) {
  injectStyles();
  const S = D.files.shift, dirs = S.setup.directions, shifts = S.setup.shifts;
  let dir = 'right', px = shifts[Math.min(3, shifts.length - 1)];
  const pd = presets('Direction', dirs.map((d) => [d, d]), (d) => { dir = d; draw(true); });
  const pm = presets('Move by', shifts.map((s) => [s, s === 1 ? '1 pixel' : `${s} pixels`]), (s) => { px = s; draw(true); });
  const box = el('div', { class: 'sw-plot' });
  const read = el('p', { class: 'sw-read' });
  const say = el('p', { class: 'sr', 'aria-live': 'polite' });
  host.replaceChildren(el('div', { class: 'sw' }, pd.root, pm.root, box,
    el('p', { class: 'hint' }, `Each line shows how many percentage points a network loses after the move, on all ${S.setup.test_samples.toLocaleString('en-US')} test images. A fall from 90% to 80% correct is a loss of 10 points. Lower is better.`), read, say));
  const table = srTable(host, { columns: ['move in pixels', 'MLP points lost', 'CNN points lost'], rows: [], caption: 'Percentage points of correctly read test images lost after a move' });
  const p = createPlot(box, { width: 600, height: 300, x: [-0.2, shifts[shifts.length - 1] + 0.3], y: [0, 90], xLabel: 'move in pixels', yLabel: 'points lost', ariaLabel: 'Two lines, for the MLP and the CNN. They show how many percentage points of correctly read test images each network loses against the size of the move.', margin: { l: 56, r: 12 }, onResize: () => draw() });
  function draw(announce = false) {
    pd.set(dir); pm.set(px);
    const series = shiftSeries(S, dir);
    p.clear('lines');
    p.root.querySelectorAll('.plot-axes').forEach((g) => g.remove());
    p.axes({ xTicks: shifts, yTicks: [0, 20, 40, 60, 80] });
    // the label of the higher line goes above its end, the lower one below
    const end = series.map((s) => s.points[s.points.length - 1][1]), hiIdx = end[0] >= end[1] ? 0 : 1;
    series.forEach((s, k) => {
      p.curve(s.points, { class: `plot-curve sw-line-${s.id}`, layer: 'lines' });
      for (const [x, y] of s.points) p.dot(x, y, { class: `sw-dot-${s.id}`, r: x === px ? 6 : 3.5, layer: 'lines' });
      const [lx, ly] = s.points[s.points.length - 1];
      p.text(lx, ly, s.label, { class: `plot-text sw-lab-${s.id}`, anchor: 'end', dx: -8, dy: k === hiIdx ? -10 : 18, layer: 'lines' });
    });
    table.update(series[0].points.map(([x, y], i) => [x, Math.round(y * 10) / 10, Math.round(series[1].points[i][1] * 10) / 10]));
    const at = (k) => Math.round(series[k].points[shifts.indexOf(px)][1]);
    read.textContent = px === 0
      ? 'Without a move, neither network loses anything. Choose a move.'
      : `Moved ${px === 1 ? '1 pixel' : `${px} pixels`} ${DIR_WORD[dir]}, the MLP loses ${at(0)} points and the CNN loses ${at(1)} points.`;
    if (announce) say.textContent = read.textContent;
  }
  draw();
  return { destroy: () => p.destroy() };
}
