// Lesson mnist-cnn (template 3): the browser port of the CNN against TorC's logits and feature maps, and every claim of the prose
// against the data: the mistake counts, the honest comparison (C9) and its spread over three random starts (C5,
// data/mnist-seeds.json), what the first kernels find, the confusion pairs, the pad's re-centring, the transfer quiz's arithmetic,
// and no hand-typed digit in the prose.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { readJson, course } from './helpers.mjs';

const data = readJson('data/mnist-cnn.json'), mlp = readJson('data/mnist-mlp.json'), seeds = readJson('data/mnist-seeds.json');
const D = buildD({ 'mnist-cnn': data, 'mnist-mlp': mlp, 'mnist-seeds': seeds }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const lesson = (await import('../js/lessons/mnist-cnn.js')).default;
const { forwardCnn, cnnFirstMaps, conv2d, argmax, preprocess } = await import('../js/widgets/digit-widgets.js');
const { outSize } = await import('../js/widgets/conv-widgets.js');
const html = () => renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n');
const shown = (s) => s.replace(/<[^>]+>/g, '');

test('the layers are Conv2d(1, 8, 3), Conv2d(8, 16, 3) and Linear(784, 10): 9,098 parameters, about a third of the MLP', () => {
  assert.deepEqual(data.layers.map((l) => l.weight_shape), [[8, 1, 3, 3], [16, 8, 3, 3], [10, 784]]);
  const n = data.layers.reduce((a, l) => a + l.weight.length + l.bias.length, 0);
  assert.equal(n, data.setup.parameters);
  assert.equal(n, 9098);
  const r = n / mlp.setup.parameters;
  assert.ok(r > 0.3 && r < 0.37, `about a third: ${r}`);
  assert.deepEqual(data.first_maps.shape, [1, 8, 14, 14]);
  assert.equal(data.layers[2].weight_shape[1], 16 * 7 * 7, 'the linear layer reads every cell of the second maps');
  assert.equal(data.setup.train_samples, mlp.setup.train_samples, 'the same training images');
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
  assert.deepEqual(conv2d([1, 2, 3, 4], 1, 2, 2, { weight_shape: [1, 1, 2, 2], weight: [1, 1, 1, 1], bias: [0.5] }, 1, 0).y, [10.5]);
});

test('the pad task: the CNN reads all of the last four test images wrongly', () => {
  assert.equal(data.shown.length, 20);
  for (const s of data.shown.slice(16)) assert.notEqual(argmax(s.logits), s.label, `image ${s.index}`);
});

test('the mistakes: the worked example, the learner step and "about three / two in every hundred"', () => {
  for (const f of [mlp, data]) assert.equal(f.final.test_samples - f.final.test_correct, f.final.test_mistakes);
  assert.equal(Math.round(mlp.final.test_mistakes / 100), 3);
  assert.equal(Math.round(data.final.test_mistakes / 100), 2);
  assert.equal(shown(T.human('files.mnist-cnn.final.test_correct', 'per100', { of: 'files.mnist-cnn.final.test_samples' })), '2');
  assert.equal(shown(T.human('files.mnist-mlp.final.test_correct', 'per100', { of: 'files.mnist-mlp.final.test_samples' })), '3');
  assert.equal(data.final.test_correct, data.epochs.at(-1).test_correct);
});

test('C9: the comparison is not perfectly fair (epochs and learning rates differ), and the CNN still wins', () => {
  const h = html();
  assert.match(h, /This comparison is not perfectly fair\. The two runs use different numbers of <a[^>]*>epochs<\/a> and different <a[^>]*>learning rates<\/a>\./);
  assert.match(h, /The CNN still wins, with about a third of the parameters\./);
  assert.doesNotMatch(h, /settings of the two/);
  assert.notEqual(data.settings.epochs, mlp.settings.epochs);
  assert.notEqual(data.settings.lr, mlp.settings.lr);
  assert.ok(data.settings.epochs < mlp.settings.epochs, 'the CNN wins with fewer epochs');
  assert.notEqual(data.settings.example.epochs, data.settings.epochs, 'the In TorC panel: the example trains longer than the run');
  assert.ok(data.final.test_correct > mlp.final.test_correct);
});

test('C5: three random starts each; the best MLP run is behind the worst CNN run', () => {
  assert.equal(seeds.setup.seed_sets.length, 3);
  assert.equal(seeds.mlp.runs.length, 3);
  assert.equal(seeds.cnn.runs.length, 3);
  assert.equal(seeds.mlp.runs[0].test_correct, mlp.final.test_correct, 'start 1 is the run of mnist-mlp.json');
  assert.equal(seeds.cnn.runs[0].test_correct, data.final.test_correct, 'start 1 is the run of mnist-cnn.json');
  assert.equal(seeds.mlp.max_accuracy, Math.max(...seeds.mlp.runs.map((r) => r.test_accuracy)));
  assert.equal(seeds.cnn.min_accuracy, Math.min(...seeds.cnn.runs.map((r) => r.test_accuracy)));
  assert.ok(seeds.cnn.min_accuracy > seeds.mlp.max_accuracy, 'the CNN was ahead with every start');
  assert.equal(seeds.derived.cnn_always_ahead, true);
  // the brief's ranges: MLP 96.51-96.66 %, CNN 97.53-97.94 %
  assert.deepEqual([seeds.mlp.min_accuracy, seeds.mlp.max_accuracy, seeds.cnn.min_accuracy, seeds.cnn.max_accuracy], [0.9651, 0.9666, 0.9753, 0.9794]);
  assert.equal(shown(T.human('files.mnist-seeds.mlp.max_accuracy', 'per100')), '3');
  assert.equal(shown(T.human('files.mnist-seeds.cnn.min_accuracy', 'per100')), '2');
  assert.deepEqual([seeds.mlp.settings.epochs, seeds.cnn.settings.epochs], [mlp.settings.epochs, data.settings.epochs], 'each model keeps its own settings');
});

// how strongly each first-layer map follows one kind of stroke edge, over the first 16 test images (Pearson correlation)
function edgeCorr(k, edge) {
  const px = (x, r, c) => (r < 0 || r > 27 || c < 0 || c > 27 ? 0 : x[r * 28 + c] / 255);
  const f = { top: (x, r, c) => px(x, r + 1, c) - px(x, r - 1, c), right: (x, r, c) => px(x, r, c - 1) - px(x, r, c + 1) }[edge];
  const A = [], B = [];
  for (const s of data.shown.slice(0, 16)) {
    const { maps } = cnnFirstMaps(data.layers, s.pixels);
    for (let i = 0; i < 14; i++) for (let j = 0; j < 14; j++) { A.push(maps[k * 196 + i * 14 + j]); B.push(Math.max(0, f(s.pixels, 2 * i, 2 * j))); }
  }
  const m = (v) => v.reduce((a, b) => a + b, 0) / v.length, ma = m(A), mb = m(B);
  let sab = 0, sa = 0, sb = 0;
  A.forEach((a, i) => { sab += (a - ma) * (B[i] - mb); sa += (a - ma) ** 2; sb += (B[i] - mb) ** 2; });
  return sab / Math.sqrt(sa * sb);
}
test('the kernel tasks: blue top row and amber below finds top edges; amber left and blue right finds right edges', () => {
  const W = data.layers[0].weight, K = (k) => W.slice(9 * k, 9 * k + 9);
  const top = [], right = [];
  for (let k = 0; k < 8; k++) {
    const w = K(k);
    if (w.slice(0, 3).every((v) => v < 0) && w.slice(3).reduce((a, b) => a + b, 0) > 0) top.push(k);
    if ([w[0], w[3], w[6]].every((v) => v > 0) && [w[2], w[5], w[8]].every((v) => v < 0)) right.push(k);
  }
  assert.ok(top.length >= 1, 'a kernel that is blue in its top row and amber below');
  for (const k of top) assert.ok(edgeCorr(k, 'top') > 0.7, `kernel ${k} follows top edges: ${edgeCorr(k, 'top')}`);
  assert.equal(right.length, 1, '"the kernel that is amber on the left and blue on the right" is unique');
  assert.ok(edgeCorr(right[0], 'right') > 0.7, `kernel ${right[0]} follows right edges`);
  assert.equal(data.shown[1].index, 1, 'the task picks image #1');
});

test('the confusion pairs: the CNN reads fours as nines and nines as fours less often, and sevens as twos more often', () => {
  const a = mlp.confusion.matrix, b = data.confusion.matrix;
  for (const M of [a, b]) assert.equal(M.flat().reduce((s, v) => s + v, 0), 10000);
  b.forEach((row, i) => assert.equal(row[i], data.test_per_class.correct[i]));
  assert.ok(b[4][9] < a[4][9] && b[9][4] < a[9][4], 'less often in both directions');
  assert.ok(a[4][9] > 0 && b[4][9] > 0, 'both networks read some fours as nines');
  assert.ok(b[7][2] > a[7][2], 'the CNN does not win on every pair');
  assert.deepEqual([a[4][9], b[4][9], a[7][2], b[7][2]], [19, 14, 10, 20]);
});

test("the pad task: the pad moves a drawing near one side to the centre, as MNIST's images are", () => {
  const n = 56, alpha = new Float64Array(n * n);
  for (let y = 10; y < 40; y++) for (let x = 2; x < 10; x++) alpha[y * n + x] = 1;   // a bar at the left edge
  const out = preprocess(alpha, n);
  let m = 0, cx = 0;
  out.forEach((v, k) => { m += v; cx += v * ((k % 28) + 0.5); });
  assert.ok(Math.abs(cx / m - 14) < 1.5, `centre of mass ${cx / m}`);
});

test('the transfer quiz: on images twice as wide and tall, only the linear layer grows, four times', () => {
  const s1 = outSize(56, 3, 2, 1), s2 = outSize(s1, 3, 2, 1);
  assert.equal(s2 * s2 * 16, 4 * data.layers[2].weight_shape[1]);
  // the kernels' parameter counts depend only on (C_out, C_in, K)
  assert.deepEqual(data.layers.slice(0, 2).map((l) => l.weight.length), [8 * 1 * 9, 16 * 8 * 9]);
});

test('the review links every part, the answer and the bridge, and no hand-typed digit in the prose', () => {
  const h = html();
  for (const id of ['what-is-learning', 'why-gradients', 'nonlinearity', 'generalization', 'convolution']) assert.match(h, new RegExp(`href="\\.\\./${id}/"`));
  assert.match(h, /from digits to ChatGPT/);
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<pre[\s\S]*?<\/pre>/g, '').replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>')
    .replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/digit-widgets.js');
  assert.equal(typeof m.mountCnnMaps, 'function');
  assert.equal(typeof document, 'undefined');
});
