// Lesson mnist-mlp ("Reading handwritten digits with a network", template 3): every claim of the page checked from what TorC
// recorded (data/mnist-mlp.json, data/mnist-mistakes.json, data/mnist-seeds.json), the browser port of the forward pass against
// TorC's logits, the preprocessing of a drawing, the widget helpers, and no hand-typed digits. The v3 rules (roles, budgets,
// glossary, notation order) run on this lesson in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { humanValue } from '../js/core/numbers.js';
import { readJson, course } from './helpers.mjs';

const data = readJson('data/mnist-mlp.json');
const mistakes = readJson('data/mnist-mistakes.json');
const seeds = readJson('data/mnist-seeds.json');
const D = buildD({ 'mnist-mlp': data, 'mnist-mistakes': mistakes, 'mnist-seeds': seeds }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const mod = await import('../js/lessons/mnist-mlp.js');
const lesson = mod.default;
const { forward, softmax, argmax, preprocess, linear } = await import('../js/widgets/digit-widgets.js');
const W = await import('../js/widgets/mnist-widgets.js');
const per100 = (acc) => humanValue('per100', acc);
const render = () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
};
const html = render();
const V = data.validation_run;

test('the layers are Linear(784, 32), Linear(32, 32), Linear(32, 10), as the real line shows, with the parameter count the page states', () => {
  assert.deepEqual(data.layers.map((l) => l.weight_shape), [[32, 784], [32, 32], [10, 32]]);
  const n = data.layers.reduce((a, l) => a + l.weight.length + l.bias.length, 0);
  assert.equal(n, data.setup.parameters);
  assert.equal(n, D.anchors.mlp_parameters);
  assert.equal(data.setup.image_side ** 2, data.setup.pixels);
  assert.equal(data.setup.pixels, data.layers[0].weight_shape[1]);
  assert.equal(data.setup.classes, data.layers[2].weight_shape[0]);
  // every Linear(in, out) of the real line, in order, against the recorded shapes (out, in)
  const lines = [...mod.MODEL_CODE.matchAll(/nn::Linear>\((\d+), (\d+)\)/g)].map((m) => [Number(m[2]), Number(m[1])]);
  assert.deepEqual(lines, data.layers.map((l) => l.weight_shape));
  assert.equal((mod.MODEL_CODE.match(/nn::ReLU/g) ?? []).length, 2, 'ReLU after each of the first two layers');
  assert.equal(data.layers[0].weight_shape[0], data.layers[1].weight_shape[0], 'the first two layers have the same number of neurons');
  assert.match(data.settings.model, /^Linear\(784, 32\), ReLU, Linear\(32, 32\), ReLU, Linear\(32, 10\)$/);
  assert.deepEqual([data.settings.optimizer, data.settings.loss], ['optim::AdamW', 'nn::CrossEntropyLoss']);
  assert.deepEqual([data.setup.pixel_max, data.setup.batch_size, data.setup.epochs], [255, data.settings.batch_size, mod.LAST]);
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

test('the pad task: the last four buttons are test images that TorC reads wrongly', () => {
  assert.deepEqual(data.shown.slice(0, 16).map((s) => s.index), Array.from({ length: 16 }, (_, i) => i));
  const wrong = data.shown.slice(16);
  assert.equal(wrong.length, 4);
  for (const s of wrong) assert.notEqual(argmax(s.logits), s.label, `image ${s.index}`);
  for (const s of data.shown) { assert.equal(s.pixels.length, 784); s.pixels.forEach((q) => assert.ok(Number.isInteger(q) && q >= 0 && q <= 255)); }
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

test('the weight gallery tasks: neighbouring weights in the middle agree more than at the edge', () => {
  const Wt = data.layers[0].weight;
  let re = 0, ae = 0, rc = 0, ac = 0;
  for (let r = 0; r < 32; r++) for (let y = 0; y < 28; y++) for (let x = 0; x < 27; x++) {
    const k = y * 28 + x, a = Wt[r * 784 + k], b = Wt[r * 784 + k + 1];
    if (y < 4 || y > 23 || x < 4 || x > 22) { re += Math.abs(a - b); ae += Math.abs(a); } else if (y >= 7 && y < 21 && x >= 7 && x < 20) { rc += Math.abs(a - b); ac += Math.abs(a); }
  }
  // for independent random weights the ratio is about the square root of 2
  assert.ok(rc / ac < re / ae, `middle ${rc / ac} vs edge ${re / ae}`);
  assert.ok(rc / ac < 0.9 * Math.SQRT2, 'the middle forms patches, not noise');
});

test('the validation run (C2): held-out images, small gap, the predict reveal and the slider tasks', () => {
  assert.equal(V.train_samples + V.validation_samples, data.setup.train_samples);
  assert.equal(V.validation_first_index, V.train_samples, 'the last images are the validation set');
  assert.equal(V.epochs.length, mod.LAST + 1);
  const E = V.epochs, last = E[mod.LAST];
  assert.equal(last.validation.total, V.validation_samples);
  assert.equal(last.train_subset.total, V.train_subset_samples);
  // predict: on new images "a few, a little more than on its training images"
  const val = per100(last.validation.accuracy), tr = per100(last.train_subset.accuracy);
  assert.ok(val >= 1 && val <= 5, `a few: ${val}`);
  assert.ok(tr < val, 'a little more on new images');
  assert.ok(last.train_subset.accuracy > 0.98, 'almost all training images right');
  assert.ok(val - tr <= 2, 'the gap is small');
  // task 1: after the first pass the validation images have fewer mistakes than the training images
  assert.ok(E[1].validation.accuracy > E[1].train_subset.accuracy);
  // task 2: the training curve keeps falling, the validation curve falls more slowly, and a gap opens
  const C = W.trainValidationCurves(data);
  assert.equal(C.passes.length, mod.LAST);
  assert.ok(C.train.at(-1) < C.train[9] && C.train[9] < C.train[0]);
  assert.ok(C.train[9] - C.train.at(-1) > C.validation[9] - C.validation.at(-1));
  assert.ok(C.validation.at(-1) - C.train.at(-1) > 0.5, 'a gap opens');
  assert.ok(C.validation[0] < C.train[0], 'the validation curve starts lower');
  assert.ok(C.train[9] < C.validation[9], 'by the tenth epoch the training curve is lower');
  assert.ok(C.validation.at(-1) - C.train.at(-1) > C.validation[9] - C.train[9], 'the gap grows slowly');
  // "it flattens and even rises a little" over the last few epochs, "while the training curve keeps falling"
  assert.ok(C.validation.at(-1) > C.validation.at(-3), 'the validation curve rises from epoch 18 to 20');
  assert.ok(Math.abs(C.validation.at(-1) - C.validation.at(-6)) < 0.5, 'flat over the last five epochs');
  assert.ok(C.train.at(-1) < C.train.at(-6) - 0.5);
  // "as many as in the validation set"
  assert.equal(V.train_subset_samples, V.validation_samples);
  C.train.forEach((v, i) => assert.ok(Math.abs(v - 100 * (1 - V.epochs[i + 1].train_subset.accuracy)) < 1e-9));
});

test('the test set: the validation run reads it once, after training; the other runs recorded it per epoch (C2, C5)', () => {
  // "After training, this run reads the test set once": the validation run has no test value per epoch, only final_test
  assert.equal(V.final_test.total, data.setup.test_samples);
  assert.match(V.note, /The test set is measured once, after training\./);
  for (const e of V.epochs) assert.deepEqual(Object.keys(e).filter((k) => /test/.test(k)), [], `epoch ${e.epoch} has no test value`);
  assert.equal(per100(V.final_test.accuracy), 3);
  assert.match(html, /data-trace="files\.mnist-mlp\.validation_run\.final_test\.accuracy"/);
  // "These runs also recorded the test result after each epoch": the main run and the seed runs did
  assert.ok(data.epochs.every((e) => typeof e.test_accuracy === 'number'));
  assert.ok(seeds.mlp.runs.every((r) => r.test_correct_by_epoch.length === data.setup.epochs + 1));
  // "this course chooses nothing with those numbers": the page binds no per-epoch test value
  assert.doesNotMatch(html, /data-trace="files\.mnist-mlp\.epochs\./);
  assert.doesNotMatch(html, /test_correct_by_epoch/);
  // three random starts: the range the page shows, and "changes the result only a little"
  assert.equal(seeds.mlp.runs.length, 3);
  assert.equal(seeds.mlp.runs[0].test_correct, data.final.test_correct, 'seed set 1 is the network of the digit pad');
  assert.equal(seeds.mlp.min_accuracy, Math.min(...seeds.mlp.runs.map((r) => r.test_accuracy)));
  assert.equal(seeds.mlp.max_accuracy, Math.max(...seeds.mlp.runs.map((r) => r.test_accuracy)));
  assert.ok(seeds.mlp.max_accuracy - seeds.mlp.min_accuracy < 0.002);
  // the prose gives the range as mistakes per hundred: the best run makes the fewest
  assert.ok(Math.abs(seeds.mlp.min_mistakes_per100 - 100 * (1 - seeds.mlp.max_accuracy)) < 1e-9);
  assert.ok(Math.abs(seeds.mlp.max_mistakes_per100 - 100 * (1 - seeds.mlp.min_accuracy)) < 1e-9);
  assert.match(html, /data-trace="files\.mnist-seeds\.mlp\.min_mistakes_per100"/);
  assert.equal(data.final.test_correct, data.confusion.correct, 'the confusion matrix is the final network of the pad');
});

test('the confusion matrix: sums, diagonal, the most common mistake and the rare pairs the quiz feedback names', () => {
  const M = data.confusion.matrix;
  assert.equal(M.flat().reduce((a, b) => a + b, 0), data.setup.test_samples);
  M.forEach((row, t) => {
    assert.equal(row[t], data.test_per_class.correct[t]);
    assert.equal(row.reduce((a, b) => a + b, 0), data.test_per_class.total[t]);
    assert.ok(Math.abs(data.test_per_class.accuracy[t] - row[t] / data.test_per_class.total[t]) < 1e-12);
  });
  const top = W.confusions(M);
  assert.deepEqual([top[0].t, top[0].p], [4, 9], 'a four read as a nine is the most common mistake');
  assert.ok(M[9][4] < M[4][9], 'the opposite mistake is less common');
  // the most common mistakes of a nine: eights, then fours and sevens
  const nine = top.filter((c) => c.t === 9).slice(0, 3).map((c) => c.p).sort();
  assert.deepEqual(nine, [4, 7, 8]);
  assert.equal(W.cellText(M, 4, 9), `The network read ${M[4][9]} of the ${data.test_per_class.total[4]} test images of a four as a nine.`);
  assert.match(W.cellText(M, 8, 8), /of an eight correctly\.$/);
  assert.match(W.cellText(M, 9, 8), /of the 1,009 test images of a nine as an eight\.$/, 'a comma from 1,000 up (decision 22)');
  assert.match(W.cellText(M, 1, 7), /^The network never read a one as a seven\.$/);});

test('the worked row and the "your turn" quiz: the nine is the digit read worst', () => {
  const acc = data.test_per_class.accuracy;
  assert.equal(Math.min(...acc), acc[mod.MINE]);
  assert.ok(Math.abs(acc[mod.ROW] - data.confusion.matrix[mod.ROW][mod.ROW] / data.test_per_class.total[mod.ROW]) < 1e-12);
  assert.ok(Math.abs(T.num(`files.mnist-mlp.test_per_class.accuracy.${mod.MINE}`) - 958 / 1009) < 1e-12);
});

test('the grid of mistakes: the same images as mistakes.items, and the first is a four read as a nine', () => {
  assert.equal(mistakes.items.length, data.mistakes.items.length);
  mistakes.items.forEach((it, i) => {
    const r = data.mistakes.items[i];
    assert.deepEqual([it.index, it.label, it.predicted], [r.index, r.label, r.predicted]);
    assert.notEqual(it.label, it.predicted);
    assert.equal(it.pixels.length, 784);
    assert.ok(data.confusion.matrix[it.label][it.predicted] > 0);
  });
  assert.deepEqual([mistakes.items[0].label, mistakes.items[0].predicted], [4, 9]);
  assert.ok(mistakes.checks.predictions_match_mnist_mlp && mistakes.checks.weights_reproduce_mnist_mlp_logits);
  for (let i = 1; i < mistakes.items.length; i++) assert.ok(mistakes.items[i].index > mistakes.items[i - 1].index, 'in test-set order');
});

test('the cumulative quiz: one epoch is the batches of all training images, rounded up', () => {
  const s = data.settings;
  assert.ok(s.train_samples % s.batch_size !== 0, 'the division does not give a whole number');
  assert.equal(s.steps_per_epoch, Math.ceil(s.train_samples / s.batch_size));
  assert.equal(s.steps, s.steps_per_epoch * s.epochs);
  assert.equal(data.epochs.at(-1).steps, s.steps);
  assert.ok(Math.abs(s.train_samples / s.batch_size - s.steps_per_epoch) < 0.3, 'the tolerance accepts the plain division');
});

test('no hand-typed digits in the prose: every number is bound', () => {
  const prose = html.replace(/<style[\s\S]*?<\/style>/g, ' ').replace(/<pre[\s\S]*?<\/pre>/g, ' ').replace(/<svg[\s\S]*?<\/svg>/g, ' ')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, ' ').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, ' ').replace(/<h2[^>]*>\s*\d+ ·/g, '<h2>').replace(/<[^>]+>/g, ' ');
  assert.deepEqual(prose.match(/\S*\d\S*/g), null);
});

test('the widget modules import in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/digit-widgets.js');
  assert.equal(typeof m.mountDigitPad, 'function');
  assert.equal(typeof m.mountWeightsGallery, 'function');
  assert.equal(typeof W.mountConfusion, 'function');
  assert.equal(typeof W.mountTrainValidation, 'function');
  assert.equal(W.sig2(1.6), '1.6');
  assert.equal(W.sig2(12.6), '13');
  assert.equal(typeof document, 'undefined');
});
