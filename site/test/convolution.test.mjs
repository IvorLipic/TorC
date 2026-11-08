// Lesson convolution (template 3): every recorded output recomputed from its input and kernel, the size rule and the parameter
// count step by step (data/conv.json derived), the port of the moved feature maps against TorC's maps (data/shift.json), every
// claim of the prose checked against the data, and no hand-typed digit in the prose (brief C8, R2, D14).
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { readJson, course } from './helpers.mjs';

const data = readJson('data/conv.json'), shift = readJson('data/shift.json'), cnn = readJson('data/mnist-cnn.json'), mlp = readJson('data/mnist-mlp.json');
const D = buildD({ conv: data, shift, 'mnist-cnn': cnn }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const lesson = (await import('../js/lessons/convolution.js')).default;
const { padded, outSize, windowAt } = await import('../js/widgets/conv-widgets.js');
const { shiftPixels, movedDiff, shiftReport, shiftSeries } = await import('../js/widgets/shift-widgets.js');
const { conv2d, cnnFirstMaps } = await import('../js/widgets/digit-widgets.js');
const html = () => renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n');

test('the small example: each recorded output is the sum of the nine products in its window (no kernel flip)', () => {
  const E = data.example, n = E.input.shape[2], k = E.kernel.shape[2];
  assert.equal(E.runs.length, 3);
  for (const R of E.runs) {
    const out = outSize(n, k, R.stride, R.padding);
    assert.deepEqual(R.output.shape, [1, 1, out, out], `stride ${R.stride} padding ${R.padding}`);
    for (let i = 0; i < out; i++) for (let j = 0; j < out; j++)
      assert.equal(windowAt(E.input.values, n, E.kernel.values, k, R.stride, R.padding, i, j).sum, R.output.values[i * out + j], `(${i}, ${j})`);
  }
  assert.deepEqual(padded([1, 2, 3, 4], 2, 1), [0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0]);
});

test('the prose about the small example: dark on the left, bright on the right, one brighter pixel; the kernel adds left and subtracts right', () => {
  const v = data.example.input.values, n = 5;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    if (x < 2) assert.equal(v[y * n + x], 0);
    else if (!(x === 2 && y === 2)) assert.equal(v[y * n + x], 4);
  }
  assert.equal(v[12], 2 * v[4], 'one pixel in the middle is twice as bright');
  const k = data.example.kernel.values;
  for (let r = 0; r < 3; r++) { assert.ok(k[r * 3] > 0); assert.equal(k[r * 3 + 1], 0); assert.ok(k[r * 3 + 2] < 0); }
  assert.equal(k[3], 2 * k[0], 'the middle row counts twice');
  // "zero where the image is even": a window inside the dark part or the bright part (away from the brighter pixel) gives 0
  assert.equal(windowAt(v, n, k, 3, 1, 1, 0, 0).sum, 0, 'all dark (and padding)');
  // task 1: the top row of the first setting: the first two large and negative, the last small
  const top = data.example.runs[0].output.values.slice(0, 3);
  assert.ok(top[0] < 0 && top[1] < 0 && Math.abs(top[2]) * 3 < Math.min(-top[0], -top[1]), JSON.stringify(top));
  // task 2: stride 2, padding 1 gives the 3 x 3 map that the task names, with a second edge (far from zero) in its right column
  // where the zero border meets the bright side; the second setting (stride 1, padding 1) gives 5 x 5
  assert.deepEqual([data.example.runs[2].stride, data.example.runs[2].padding], [2, 1]);
  const r2 = data.example.runs[2].output.values;
  for (let i = 0; i < 3; i++) assert.ok(r2[i * 3 + 2] >= 12, `second edge, row ${i}`);
  assert.deepEqual([data.example.runs[1].stride, data.example.runs[1].padding, data.example.runs[1].output.shape[2]], [1, 1, 5]);
  assert.equal(data.example.runs[0].stride, 1, 'the first setting moves one pixel at a time');
});

test('the size rule and the parameter count, step by step (derived in trace_conv.cpp), agree with the library', () => {
  const [a, b] = data.derived.layers;
  for (const [d, c] of [[a, data.cnn[0]], [b, data.cnn[2]]]) {
    assert.equal(d.side_in, c.in_shape[2]);
    assert.equal(d.twice_padding, 2 * d.padding);
    assert.equal(d.numerator, d.side_in + 2 * d.padding - d.kernel);
    assert.equal(d.quotient, d.numerator / d.stride);
    assert.equal(d.rounded_down, Math.floor(d.quotient));
    assert.equal(d.side_out, d.rounded_down + 1);
    assert.equal(d.side_out, c.out_shape[2]);
    assert.equal(d.side_out, outSize(d.side_in, d.kernel, d.stride, d.padding));
    assert.equal(d.kernel_area, d.kernel * d.kernel);
    assert.equal(d.weights_per_kernel, d.in_channels * d.kernel_area);
    assert.equal(d.weights, d.out_channels * d.weights_per_kernel);
    assert.equal(d.parameters_per_kernel, d.weights_per_kernel + 1);
    assert.equal(d.parameters, d.out_channels * d.parameters_per_kernel);
    assert.equal(d.parameters, c.parameters);
    assert.deepEqual([d.out_channels, d.in_channels, d.kernel, d.kernel], c.params.find((p) => p.name === 'weight').shape);
  }
  assert.ok(!Number.isInteger(a.quotient) && !Number.isInteger(b.quotient), 'both worked examples really round down');
  // the predict's slip (forgetting the bias) and the learner's step (stride 1, padding 1 keeps the size)
  assert.equal(a.weights, data.cnn[0].parameters - a.out_channels);
  const so = data.derived.stride_one;
  assert.deepEqual([so.side_in, so.kernel, so.stride, so.padding], [14, 3, 1, 1]);
  assert.equal(so.numerator, so.side_in + 2 * so.padding - so.kernel);
  assert.equal(so.quotient, so.numerator / so.stride);
  assert.equal(so.side_out, outSize(so.side_in, so.kernel, so.stride, so.padding));
  assert.equal(so.side_out, so.side_in, 'the reveal: the map keeps the size of its input');
  assert.equal(a.parameters_per_kernel, 10, 'The answer says ten parameters per kernel');
  assert.equal(b.side_in, a.side_out, 'the second layer reads the maps of the first');
  assert.equal(b.in_channels, a.out_channels, 'the next layer reads the maps like an image with that many channels');
  // the CNN in total, and "about a third" of the MLP
  assert.equal(data.cnn_parameters, data.cnn.reduce((s, c) => s + c.parameters, 0));
  assert.equal(data.cnn_parameters, cnn.setup.parameters);
  const r = data.cnn_parameters / mlp.setup.parameters;
  assert.ok(r > 0.3 && r < 0.37, `about a third: ${r}`);
  // the predict: the linear layer from the same image to as many neurons as there are kernels
  assert.equal(data.dense_compare.parameters, 784 * a.out_channels + a.out_channels);
  assert.equal(a.side_in * a.side_in, 784);
});

test('errors: the channel mismatch of the In TorC panel', () => {
  const e = Object.fromEntries(data.errors.map((x) => [x.id, x]));
  assert.equal(e.channels.what, 'conv2d input channels 3 must match weight channels 1');
  assert.equal(e.channels.exception, 'ShapeError');
  // the page no longer quotes the error (word budget); docs/lessons/convolution.md keeps it
});

test("the moved feature maps: the port gives TorC's maps for the original and the moved image (C8)", () => {
  const S = shift.feature_maps, layers = cnn.layers;
  assert.deepEqual(S.pixels, cnn.shown.find((s) => s.index === S.index).pixels, 'the widget uses the same test image');
  const base = cnnFirstMaps(layers, S.pixels).maps, moved = cnnFirstMaps(layers, shiftPixels(S.pixels, S.shift_px, 0)).maps;
  base.forEach((v, q) => assert.ok(Math.abs(v - S.original[q]) < 1e-5, `original ${q}`));
  moved.forEach((v, q) => assert.ok(Math.abs(v - S.shifted[q]) < 1e-5, `shifted ${q}`));
  const [, c, h, w] = S.shape;
  const two = movedDiff(base, moved, c, h, w, S.cell_shift);
  assert.equal(two.compared, S.check.compared_cells);
  assert.ok(two.d < 1e-6, `moved by the stride: equal (TorC: ${S.check.max_abs_diff})`);
  assert.equal(S.check.equal, true);
  assert.equal(S.check.max_abs_diff, 0);
  assert.equal(S.cell_shift, S.shift_px / S.stride);
  // a move of 1 px: TorC's two differences, and the port's
  const one = shiftReport(layers, S.pixels, S.not_a_multiple.shift_px, S.stride);
  assert.ok(Math.abs(one.still - S.not_a_multiple.max_abs_diff_unmoved) < 1e-4, `${one.still}`);
  assert.ok(Math.abs(one.oneCell - S.not_a_multiple.max_abs_diff_moved_one_cell) < 1e-4, `${one.oneCell}`);
  assert.ok(one.still > 0.5 && one.oneCell > 0.5, 'the map changes, it does not move');
  // the widget's readouts for every move it offers: whole strides move exactly, the others change
  for (const px of shift.setup.shifts) {
    const R = shiftReport(layers, S.pixels, px, S.stride);
    if (px % S.stride === 0) assert.ok(R.exact.d < 1e-6, `${px} px moves by ${px / S.stride} cells`);
    else assert.ok(R.still > 0.5 && R.oneCell > 0.5, `${px} px changes the maps`);
  }
  // task 1: the default kernel (the brightest map); its leftmost bright column moves one column right at a move of 2 px
  const R0 = shiftReport(layers, S.pixels, 0, S.stride);
  let best = -1, ch = 0;
  for (let k = 0; k < c; k++) { const m = Math.max(...R0.base.slice(k * h * w, (k + 1) * h * w)); if (m > best) { best = m; ch = k; } }
  const leftmost = (maps) => { const m = maps.slice(ch * h * w, (ch + 1) * h * w), top = Math.max(...m); for (let j = 0; j < w; j++) for (let i = 0; i < h; i++) if (m[i * w + j] > 0.05 * top) return j; return -1; };
  const R2 = shiftReport(layers, S.pixels, 2, S.stride);
  assert.equal(leftmost(R2.moved), leftmost(R2.base) + 1, 'one more dark column');
  // shiftPixels moves right and down with zero fill
  assert.deepEqual(shiftPixels([1, 2, 3, 4], 1, 0, 2), [0, 1, 0, 3]);
  assert.deepEqual(shiftPixels([1, 2, 3, 4], 0, -1, 2), [3, 4, 0, 0]);
});

test('the transfer quiz (expected mathematics): with the stride one, a move of one pixel moves the map by one cell', () => {
  const S = shift.feature_maps, L = cnn.layers[0], c = L.weight_shape[0];
  const x = S.pixels.map((p) => p / 255);
  const a = conv2d(x, 1, 28, 28, L, 1, 1), b = conv2d(shiftPixels(S.pixels, 1, 0).map((p) => p / 255), 1, 28, 28, L, 1, 1);
  const relu = (v) => v.map((z) => Math.max(0, z));
  assert.ok(movedDiff(relu(a.y), relu(b.y), c, a.ho, a.wo, 1).d < 1e-6);
});

test('the accuracy on moved test images (shift.json, all 10,000 test images): the claims of section 5 and of the widget tasks', () => {
  const A = shift.accuracy;
  assert.equal(shift.setup.test_samples, 10000);
  assert.equal(A.cnn.right[0].correct, cnn.final.test_correct, 'no move is the recorded final run');
  assert.equal(A.mlp.right[0].correct, mlp.final.test_correct);
  for (const d of shift.setup.directions) for (let s = 1; s < shift.setup.shifts.length; s++) {
    assert.ok(A.cnn[d][s].accuracy < A.cnn[d][0].accuracy, `the CNN loses digits when moved ${s} px ${d}: not shift-invariant`);
    assert.ok(A.cnn[d][s].accuracy > A.mlp[d][s].accuracy, `the CNN copes better, ${s} px ${d}`);
  }
  // section 5 (H1): moves of one and two strides, at which the first maps move exactly (tested above), still cost the CNN
  assert.deepEqual([shift.setup.shifts[2], shift.setup.shifts[4]], [2, 4]);
  assert.equal(T.human('files.shift.accuracy.cnn.right.2.accuracy', 'pct').replace(/<[^>]+>/g, ''), '91%');
  assert.equal(T.human('files.shift.accuracy.cnn.right.4.accuracy', 'pct').replace(/<[^>]+>/g, ''), '54%');
  // the widget draws percentage points lost against no move (H2: no absolute comparison of the two models on this page)
  const sr = shiftSeries(shift, 'right'), sd = shiftSeries(shift, 'down');
  assert.deepEqual(sr.map((x) => x.label), ['MLP', 'CNN']);
  assert.deepEqual(sr[1].points.map(([x]) => x), shift.setup.shifts);
  assert.ok(Math.abs(sr[1].points[3][1] - 100 * (A.cnn.right[0].accuracy - A.cnn.right[3].accuracy)) < 1e-9);
  for (const k of [0, 1]) assert.equal(sr[k].points[0][1], 0);
  for (let i = 1; i < shift.setup.shifts.length; i++) {
    assert.ok(sr[0].points[i][1] > sr[1].points[i][1], `"the MLP loses even more" and "the CNN line stays lower", ${i} px right`);
    for (const k of [0, 1]) assert.ok(sd[k].points[i][1] > sr[k].points[i][1], `"both lines climb higher" for down, ${i} px`);
  }
  for (const k of [0, 1]) assert.ok(sr[k].points[2][1] > 0 && sr[k].points[4][1] > 0, 'both lose points at both moves');
});

test('the page states C8 honestly: the map moves exactly for whole strides, and the CNN is not shift-invariant', () => {
  const h = html();
  assert.match(h, /This holds exactly only for whole strides\./);
  assert.match(h, /So the maps moved exactly, but the answer still fell\./);
  assert.doesNotMatch(h, /the CNN reads <span[^>]*>98%/, 'no absolute comparison with the MLP (H2)');
  assert.match(h, /TorC does not flip it, so each weight sits on the pixel that it multiplies\./);
  assert.doesNotMatch(h, /does not care where/i);
  assert.match(h, /Where it breaks\.<\/strong> A stencil only shows yes or no\. A kernel gives a sum with a sign\./);
  // the In TorC code is the course.json code, which the snippet check compiles and runs
  const entry = course().parts.flatMap((p) => p.lessons).find((l) => l.id === 'convolution');
  const esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  assert.ok(h.includes(`<code>${esc(entry.code)}</code>`), 'the panel shows the checked snippet');
  // D14: the code block has its comments on their own lines, and the batch is B
  const code = h.match(/<pre class="cv-code"[\s\S]*?<\/pre>/)[0];
  for (const line of code.replace(/<[^>]+>/g, '').split('\n')) assert.ok(line.trim().startsWith('//') || !line.includes('//'), line);
  assert.doesNotMatch(h, /\(N,/);
  // the picture (the sliding-window widget) comes before the floor rule
  const parts = renderLesson(lesson, D, { course: course() });
  assert.ok(lesson.sections.findIndex((x) => x.id === 'window') < parts.findIndex((p) => p.html.includes('lfloor')));
});

test('no hand-typed digit in the prose (headings, code, formulas and bound numbers aside)', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<div class="cv-eq"[\s\S]*?<\/div>/g, '')
    .replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget modules import in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/shift-widgets.js');
  assert.equal(typeof m.mountShiftMaps, 'function');
  assert.equal(typeof m.mountShiftAccuracy, 'function');
  assert.equal(typeof (await import('../js/widgets/conv-widgets.js')).mountSlidingWindow, 'function');
  assert.equal(typeof document, 'undefined');
});
