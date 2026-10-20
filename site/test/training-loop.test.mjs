// Lesson training-loop: the claims of the page against what TorC recorded (data/training.json), every number bound to the
// trace, no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/training.json');
const D = buildD({ training: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/training-loop.js')).default;
const { loaderState } = await import('../js/widgets/training-widgets.js');
const N = data.setup.samples;
const run = (bs) => data.runs.find((r) => r.batch_size === bs);

test('the samples follow t = 3x + 1 with noise of standard deviation 0.1, and x is in [-1, 1)', () => {
  assert.equal(data.data.x.length, N);
  const res = data.data.x.map((x, i) => data.data.y[i] - (data.setup.true_weight * x + data.setup.true_bias));
  data.data.x.forEach((x) => assert.ok(x >= -1 && x < 1));
  const mse = res.reduce((a, r) => a + r * r, 0) / N;
  assert.ok(Math.abs(mse - data.true_line_loss) < 1e-6, `${mse} vs ${data.true_line_loss}`);
  assert.ok(Math.abs(Math.sqrt(mse) - data.setup.noise_std) < 0.02, 'the noise is about the configured size');
});

test('every epoch of every loader gives each sample exactly once, in batches of the batch size and a smaller last one', () => {
  assert.equal(data.loaders.length, 4);
  for (const l of data.loaders) {
    for (const [e, batches] of l.epochs.entries()) {
      assert.deepEqual(batches.flat().slice().sort((a, b) => a - b), Array.from({ length: N }, (_, i) => i), `${l.batch_size} ${l.shuffle} epoch ${e}`);
      assert.equal(batches.length, Math.ceil(N / l.batch_size));
      batches.slice(0, -1).forEach((b) => assert.equal(b.length, l.batch_size));
      assert.equal(batches.at(-1).length, N - l.batch_size * (batches.length - 1));
    }
    if (!l.shuffle) l.epochs.forEach((ep) => assert.deepEqual(ep.flat(), Array.from({ length: N }, (_, i) => i), 'no shuffle keeps the order'));
    else assert.notDeepEqual(l.epochs[0].flat(), l.epochs[1].flat(), 'reset() draws a new order');
  }
  assert.deepEqual(data.exhausted, { exception: 'TorcError', what: 'DataLoader::next_batch: no more batches in current epoch' });
});

test('loaderState follows next_batch() and has_next() through one epoch', () => {
  const l = data.loaders.find((x) => x.batch_size === 32 && x.shuffle);
  const s0 = loaderState(l, 0, 0);
  assert.equal(s0.batches, 4);
  assert.ok(s0.hasNext);
  assert.deepEqual(s0.current, []);
  const s2 = loaderState(l, 0, 2);
  assert.deepEqual(s2.current, l.epochs[0][1]);
  assert.deepEqual([...s2.used].sort((a, b) => a - b), l.epochs[0][0].slice().sort((a, b) => a - b));
  assert.equal(loaderState(l, 0, 4).hasNext, false);
});

test('runs: steps per epoch are ceil(100 / batch size); smaller batches reach a lower loss in the same epochs', () => {
  for (const bs of [100, 32, 10]) {
    const r = run(bs);
    assert.equal(r.epochs.length, data.setup.epochs + 1);
    r.epochs.forEach((e, i) => assert.equal(e.steps, i * Math.ceil(N / bs), `${bs} epoch ${i}`));
    assert.equal(r.epochs[0].weight, 0);
    assert.equal(r.epochs[0].bias, 0);
  }
  const at = (bs) => run(bs).epochs.at(-1).loss;
  assert.ok(at(10) < at(32) && at(32) < at(100), `${at(10)} ${at(32)} ${at(100)}`);
  const last = run(10).epochs.at(-1);
  assert.ok(Math.abs(last.weight - data.setup.true_weight) < 0.05 && Math.abs(last.bias - data.setup.true_bias) < 0.05);
  assert.ok(Math.abs(last.loss - data.true_line_loss) < 0.001, 'the loss ends close to the loss of the true line');
  // the epoch-0 loss is the mean of t² (weight and bias are zero)
  assert.ok(Math.abs(run(10).epochs[0].loss - data.data.y.reduce((a, y) => a + y * y, 0) / N) < 1e-5);
});

test('without reset() the loop stops after the first epoch, with no error', () => {
  assert.deepEqual(data.no_reset.map((e) => e.steps), [10, 0, 0]);
  assert.equal(data.no_reset[1].loss, data.no_reset[0].loss);
  assert.equal(data.no_reset[2].loss, data.no_reset[0].loss);
});

test('every number in the prose is bound to the trace, with no hand-typed digits', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const all = parts.map((p) => p.html).join('\n');
  assert.match(all, /TorcError: DataLoader::next_batch: no more batches in current epoch/);
  assert.match(all, /<code>32, 32, 32, 4<\/code>/);
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/training-widgets.js');
  assert.equal(typeof m.mountBatchViewer, 'function');
  assert.equal(typeof m.mountTrainingCurves, 'function');
  assert.equal(typeof document, 'undefined');
});
