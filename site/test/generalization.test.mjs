// Lesson generalization: the claims of the page against what TorC recorded (data/generalization.json), every number bound to
// the trace, no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/generalization.json');
const D = buildD({ generalization: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/generalization.js')).default;
const run = (wd) => data.runs.find((r) => r.weight_decay === wd);
const last = (wd) => run(wd).checkpoints.at(-1);
const best = (wd) => run(wd).checkpoints.reduce((b, c) => (c.test_loss < b.test_loss ? c : b));

test('the data: 10 evenly spaced training samples and 100 test samples of sin(pi x), noise within ±0.3', () => {
  const d = data.data;
  assert.equal(d.train_x.length, data.setup.train_samples);
  assert.equal(d.test_x.length, data.setup.test_samples);
  d.train_x.forEach((x, i) => assert.ok(Math.abs(x - (-1 + (2 * i) / (d.train_x.length - 1))) < 1e-6));
  const within = (xs, ts) => xs.forEach((x, i) => assert.ok(Math.abs(ts[i] - Math.sin(Math.PI * x)) <= data.setup.noise + 1e-5, `x = ${x}`));
  within(d.train_x, d.train_t);
  within(d.test_x, d.test_t);
  d.test_x.forEach((x) => assert.ok(x >= -1 && x < 1));
  assert.equal(data.derived.noise_variance, (data.setup.noise * data.setup.noise) / 3);
});

test('step 0: the weights are zero, so the model is the zero curve and each loss is the mean of t²', () => {
  for (const r of data.runs) {
    const c = r.checkpoints[0];
    assert.equal(c.step, 0);
    assert.equal(c.weight_sum_sq, 0);
    c.curve.forEach((y) => assert.equal(y, 0));
    const mean2 = (ts) => ts.reduce((a, t) => a + t * t, 0) / ts.length;
    assert.ok(Math.abs(c.train_loss - mean2(data.data.train_t)) < 1e-5);
    assert.ok(Math.abs(c.test_loss - mean2(data.data.test_t)) < 1e-5);
  }
});

test('without weight decay the model overfits: the training loss goes to almost zero, the test loss falls, then rises', () => {
  assert.ok(last(0).train_loss < 1e-3);
  assert.ok(best(0).step > 0 && best(0).step < data.setup.steps);
  assert.ok(last(0).test_loss > 2 * best(0).test_loss, `${last(0).test_loss} vs ${best(0).test_loss}`);
});

test('weight decay 0.1 keeps the weights small and the test loss low; weight decay 1 underfits', () => {
  assert.ok(last(0.1).weight_sum_sq < last(0).weight_sum_sq / 10);
  assert.ok(last(0.1).test_loss < last(0).test_loss / 2);
  const cs = run(0.1).checkpoints;
  assert.ok(Math.abs(cs.at(-1).test_loss / cs.at(-3).test_loss - 1) < 0.02, '"stays almost the same in the last steps"');
  assert.ok(last(1).train_loss > 10 * last(0.1).train_loss, 'too much decay: the training loss stays high');
  assert.ok(last(1).test_loss > last(0).test_loss, 'and the test loss is the worst of the three');
  for (const r of data.runs) for (const c of r.checkpoints) assert.ok(c.test_loss > data.derived.noise_variance, 'no run gets below the noise');
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

test('the widget module imports in Node without touching the DOM; the step axis is log10(step + 1)', async () => {
  const m = await import('../js/widgets/generalization-widgets.js');
  assert.equal(typeof m.mountGeneralization, 'function');
  assert.equal(m.stepX(0), 0);
  assert.equal(m.stepX(9), 1);
  assert.deepEqual(m.DECAYS.map(([v]) => v), data.runs.map((r) => r.weight_decay));
  assert.equal(typeof document, 'undefined');
});
