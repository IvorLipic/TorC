// Lesson convolution: every recorded output recomputed from its input and kernel, the output-size formula, the parameter counts,
// the errors (data/conv.json), every number bound to the trace, no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/conv.json');
const D = buildD({ conv: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/convolution.js')).default;
const { padded, outSize, windowAt } = await import('../js/widgets/conv-widgets.js');

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
  // the page: negative where the image turns from dark to bright (left to right), positive where it turns from bright to dark
  const p1 = E.runs.find((r) => r.stride === 1 && r.padding === 1).output.values;
  for (let i = 0; i < 5; i++) { assert.ok(p1[i * 5 + 1] < 0, `dark to bright, row ${i}`); assert.ok(p1[i * 5 + 4] > 0, `bright to dark at the padded edge, row ${i}`); }
});

test('the digit: the feature maps are conv2d with stride 1 and padding 1, and relu is max(0, x)', () => {
  const G = data.digit, n = G.shape[2];
  assert.equal(G.label, 7);
  assert.equal(G.pixels.length, n * n);
  G.pixels.forEach((p) => assert.ok(p >= 0 && p <= 1, 'MNISTDataset divides by 255'));
  for (const M of G.maps) {
    assert.deepEqual(M.shape, [1, 1, n, n]);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++)
      assert.ok(Math.abs(windowAt(G.pixels, n, M.kernel, 3, 1, 1, i, j).sum - M.output[i * n + j]) < 1e-5, `${M.id} (${i}, ${j})`);
    M.output.forEach((v, q) => assert.equal(M.relu[q], Math.max(0, v)));
  }
});

test('the CNN: shapes follow (H + 2P - K) / S + 1, and the parameters are C_out·C_in·K·K + C_out per convolution', () => {
  const C = data.cnn;
  assert.deepEqual(C.map((c) => c.out_shape), [[2, 8, 14, 14], [2, 8, 14, 14], [2, 16, 7, 7], [2, 16, 7, 7], [2, 784], [2, 10]]);
  for (const c of C.filter((x) => x.kernel)) {
    assert.equal(c.out_shape[2], outSize(c.in_shape[2], c.kernel, c.stride, c.padding));
    const [co, ci, k] = c.params.find((p) => p.name === 'weight').shape;
    assert.equal(c.parameters, co * ci * k * k + co);
  }
  assert.equal(data.cnn_parameters, C.reduce((a, c) => a + c.parameters, 0));
  assert.equal(data.cnn_parameters, 9098);
  assert.equal(data.dense_compare.parameters, 784 * 8 + 8);
  assert.ok(C[0].parameters < data.dense_compare.parameters / 50);
});

test('errors: the channel mismatch and a kernel larger than the padded input', () => {
  const e = Object.fromEntries(data.errors.map((x) => [x.id, x]));
  assert.equal(e.channels.what, 'conv2d input channels 3 must match weight channels 1');
  assert.match(e['too-small'].what, /^conv2d output dims must be positive/);
  for (const x of data.errors) assert.equal(x.exception, 'ShapeError');
});

test('every number in the prose is bound to the trace, with no hand-typed digits', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const all = parts.map((p) => p.html).join('\n');
  assert.match(all, /ShapeError: conv2d input channels 3 must match weight channels 1/);
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<div class="cv-eq"[\s\S]*?<\/div>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>')
    .replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/conv-widgets.js');
  assert.equal(typeof m.mountSlidingWindow, 'function');
  assert.equal(typeof m.mountFeatureMaps, 'function');
  assert.equal(typeof document, 'undefined');
});
