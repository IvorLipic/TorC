// Lesson mnist-cnn: the browser port of the CNN's forward pass against the logits and feature maps TorC recorded, the training
// record (data/mnist-cnn.json), the comparison with the MLP (data/mnist-mlp.json), every number bound to the trace, and the
// widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/mnist-cnn.json'), mlp = readJson('data/mnist-mlp.json');
const D = buildD({ 'mnist-cnn': data, 'mnist-mlp': mlp });
const T = makeT(D);
const lesson = (await import('../js/lessons/mnist-cnn.js')).default;
const { forwardCnn, cnnFirstMaps, conv2d, argmax } = await import('../js/widgets/digit-widgets.js');

test('the layers are Conv2d(1, 8, 3), Conv2d(8, 16, 3) and Linear(784, 10): 9098 parameters', () => {
  assert.deepEqual(data.layers.map((l) => l.weight_shape), [[8, 1, 3, 3], [16, 8, 3, 3], [10, 784]]);
  const n = data.layers.reduce((a, l) => a + l.weight.length + l.bias.length, 0);
  assert.equal(n, data.setup.parameters);
  assert.equal(n, 9098);
  assert.ok(n < mlp.setup.parameters / 2);
});

test("the browser forward pass gives TorC's logits for all 20 images, and TorC's first feature maps for the first", () => {
  for (const s of data.shown) {
    const z = forwardCnn(data.layers, s.pixels);
    z.forEach((v, j) => assert.ok(Math.abs(v - s.logits[j]) < 1e-4, `image ${s.index} logit ${j}: ${v} vs ${s.logits[j]}`));
    assert.equal(argmax(z), argmax(s.logits));
  }
  const { maps, h, w } = cnnFirstMaps(data.layers, data.shown[0].pixels);
  assert.deepEqual([1, data.layers[0].weight_shape[0], h, w], data.first_maps.shape);
  maps.forEach((v, q) => assert.ok(Math.abs(v - data.first_maps.values[q]) < 1e-5, `map value ${q}`));
  // conv2d against a hand-computed case: a 2 x 2 input, a 2 x 2 kernel of ones, padding 0, stride 1
  assert.deepEqual(conv2d([1, 2, 3, 4], 1, 2, 2, { weight_shape: [1, 1, 2, 2], weight: [1, 1, 1, 1], bias: [0.5] }, 1, 0).y, [10.5]);
});

test('the shown images: the first 16, then 4 that TorC reads wrongly; one of them the MLP also reads wrongly', () => {
  assert.deepEqual(data.shown.slice(0, 16).map((s) => s.index), Array.from({ length: 16 }, (_, i) => i));
  const wrong = data.shown.slice(16);
  for (const s of wrong) assert.notEqual(argmax(s.logits), s.label, `image ${s.index}`);
  const mlpWrong = new Set(mlp.shown.slice(16).map((s) => s.index));
  assert.equal(wrong.filter((s) => mlpWrong.has(s.index)).length, 1, 'the page says the MLP reads one of them wrongly too');
});

test('the training record, and the comparison the page makes: the CNN reads more test images correctly', () => {
  const E = data.epochs;
  assert.equal(E.length, data.setup.epochs + 1);
  assert.equal(E.at(-1).test_correct, data.test_per_class.correct.reduce((a, b) => a + b, 0));
  assert.ok(E.at(-1).test_correct > mlp.epochs.at(-1).test_correct);
  assert.ok(data.setup.epochs < mlp.setup.epochs);
  assert.deepEqual(data.first_maps.shape, [1, 8, 14, 14]);
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
  assert.equal(typeof m.mountCnnMaps, 'function');
  assert.equal(typeof document, 'undefined');
});
