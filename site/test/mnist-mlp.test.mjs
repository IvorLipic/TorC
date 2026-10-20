// Lesson mnist-mlp: the browser port of the forward pass against the logits TorC recorded, the preprocessing of a drawing, the
// training record (data/mnist-mlp.json), every number bound to the trace, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/mnist-mlp.json');
const D = buildD({ 'mnist-mlp': data });
const T = makeT(D);
const lesson = (await import('../js/lessons/mnist-mlp.js')).default;
const { forward, softmax, argmax, preprocess, linear } = await import('../js/widgets/digit-widgets.js');

test('the layers are Linear(784, 32), Linear(32, 32), Linear(32, 10), with the parameter count the page states', () => {
  assert.deepEqual(data.layers.map((l) => l.weight_shape), [[32, 784], [32, 32], [10, 32]]);
  const n = data.layers.reduce((a, l) => a + l.weight.length + l.bias.length, 0);
  assert.equal(n, data.setup.parameters);
  assert.equal(n, 26506);
  assert.equal(data.setup.image_side ** 2, data.layers[0].weight_shape[1]);
});

test('the browser forward pass gives the logits TorC computed, for all 20 recorded test images', () => {
  assert.equal(data.shown.length, 20);
  for (const s of data.shown) {
    const z = forward(data.layers, s.pixels);
    z.forEach((v, j) => assert.ok(Math.abs(v - s.logits[j]) < 1e-4, `image ${s.index} logit ${j}: ${v} vs ${s.logits[j]}`));
    assert.equal(argmax(z), argmax(s.logits));
  }
  const p = softmax([1, 2, 3]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.deepEqual(linear({ weight_shape: [1, 2], weight: [2, 3], bias: [1] }, [1, 1]), [6]);
});

test('the shown images: the first 16 test images, then 4 that TorC reads wrongly', () => {
  assert.deepEqual(data.shown.slice(0, 16).map((s) => s.index), Array.from({ length: 16 }, (_, i) => i));
  const wrong = data.shown.slice(16);
  assert.equal(wrong.length, 4);
  for (const s of wrong) assert.notEqual(argmax(s.logits), s.label, `image ${s.index}`);
  for (const s of data.shown) { assert.equal(s.pixels.length, 784); s.pixels.forEach((p) => assert.ok(Number.isInteger(p) && p >= 0 && p <= 255)); }
});

test('preprocess: a drawing is cropped, scaled into 20 x 20 and centred; upscaled test images keep their class', () => {
  const N = 280;
  assert.equal(preprocess(new Float64Array(N * N), N), null, 'no ink, no image');
  let same = 0;
  for (const s of data.shown) {
    const a = new Float64Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) a[y * N + x] = s.pixels[Math.floor(y / 10) * 28 + Math.floor(x / 10)] / 255;
    const p = preprocess(a, N);
    assert.equal(p.length, 784);
    let m = 0, cx = 0, cy = 0, x0 = 28, x1 = -1, y0 = 28, y1 = -1;
    p.forEach((v, k) => { const x = k % 28, y = Math.floor(k / 28); m += v; cx += v * (x + 0.5); cy += v * (y + 0.5); if (v > 0) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } });
    assert.ok(Math.abs(cx / m - 14) <= 1 && Math.abs(cy / m - 14) <= 1, `image ${s.index}: centre of mass`);
    assert.ok(Math.max(x1 - x0 + 1, y1 - y0 + 1) <= 21, `image ${s.index}: fits the 20 x 20 box`);
    if (argmax(forward(data.layers, p)) === argmax(s.logits)) same += 1;
  }
  assert.ok(same >= 16, `${same} of 20 keep their class`);
});

test('the training record: the final test count equals the per-class counts, and the accuracy rises over the epochs', () => {
  const E = data.epochs;
  assert.equal(E.length, data.setup.epochs + 1);
  assert.equal(E.at(-1).test_correct, data.test_per_class.correct.reduce((a, b) => a + b, 0));
  assert.equal(data.test_per_class.total.reduce((a, b) => a + b, 0), data.setup.test_samples);
  assert.ok(E.at(-1).test_correct > E[1].test_correct && E[1].test_correct > E[0].test_correct);
  E.slice(1).forEach((e, i) => assert.equal(e.steps, (i + 1) * Math.ceil(data.setup.train_samples / data.setup.batch_size)));
  assert.ok(E.at(-1).test_correct / data.setup.test_samples > 0.95);
  // "rises, with only small drops": every drop is under half a percent of the test images
  for (let e = 2; e < E.length; e++) assert.ok(E[e].test_correct > E[e - 1].test_correct - data.setup.test_samples * 0.005, `epoch ${e}`);
  assert.ok(E.at(-1).test_correct >= Math.max(...E.slice(1, 10).map((x) => x.test_correct)));
});

test('every number in the prose is bound to the trace, with no hand-typed digits', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/digit-widgets.js');
  assert.equal(typeof m.mountDigitPad, 'function');
  assert.equal(typeof m.mountWeightsGallery, 'function');
  assert.equal(typeof document, 'undefined');
});
