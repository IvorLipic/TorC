// Lesson losses: the claims of the page against what TorC recorded (data/losses.json), every number bound to the trace,
// no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/losses.json');
const D = buildD({ losses: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/losses.js')).default;
const close = (a, b, tol = 1e-5, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} ${a} vs ${b}`);
const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };

test('MSELoss: the mean of the squared differences, and the gradient 2 (y - t) / N', () => {
  const { prediction: y, target: t, loss, grad } = data.mse;
  const N = y.length;
  close(loss, y.reduce((a, v, i) => a + (v - t[i]) ** 2, 0) / N, 1e-6, 'loss');
  y.forEach((v, i) => close(grad[i], (2 * (v - t[i])) / N, 1e-6, `grad ${i}`));
  assert.equal(grad[2], 0, 'an exact prediction has a zero gradient');
});

test('CrossEntropyLoss: -log softmax of the correct class, and the gradient softmax minus one-hot', () => {
  assert.deepEqual(data.ce.map((r) => r.id), ['uniform', 'mild', 'confident']);
  for (const r of data.ce) {
    const p = softmax(r.logits);
    assert.equal(r.cases.length, 3);
    for (const c of r.cases) {
      c.softmax.forEach((v, j) => close(v, p[j], 1e-6, `${r.id} softmax ${j}`));
      close(c.loss, -Math.log(p[c.target]), 1e-5, `${r.id} t${c.target} loss`);
      c.grad.forEach((g, j) => close(g, p[j] - (j === c.target ? 1 : 0), 1e-6, `${r.id} t${c.target} grad ${j}`));
    }
  }
  const uni = data.ce[0];
  for (const c of uni.cases) close(c.loss, Math.log(3), 1e-6, 'uniform loss is log 3');
});

test('a batch reports the mean loss, and each gradient row is divided by the batch size', () => {
  const B = data.batch.targets.length;
  const mild = data.ce.find((r) => r.id === 'mild'), conf = data.ce.find((r) => r.id === 'confident');
  const rows = [mild.cases[data.batch.targets[0]], conf.cases[data.batch.targets[1]]];
  close(data.batch.loss, (rows[0].loss + rows[1].loss) / B, 1e-5, 'batch loss');
  rows.forEach((c, b) => c.grad.forEach((g, j) => close(data.batch.grad[b * 3 + j], g / B, 1e-6, `batch grad ${b},${j}`)));
});

test('curves: at a very wrong logit cross-entropy keeps a strong gradient, the squared error almost none', () => {
  const C = data.curves;
  assert.equal(C.z.length, 49);
  assert.equal(C.z[0], -6);
  assert.equal(C.z.at(-1), 6);
  C.z.forEach((z, i) => {
    const p = 1 / (1 + Math.exp(-z));
    close(C.ce_loss[i], -Math.log(p), 1e-5, `ce loss at ${z}`);
    close(C.ce_grad[i], p - 1, 1e-5, `ce grad at ${z}`);
    close(C.sq_loss[i], (1 - p) ** 2, 1e-5, `sq loss at ${z}`);
    close(C.sq_grad[i], -2 * (1 - p) ** 2 * p, 1e-5, `sq grad at ${z}`);
  });
  assert.ok(C.ce_grad[0] < -0.99);
  assert.ok(Math.abs(C.sq_grad[0]) < 0.01);
  for (let i = 1; i < C.z.length; i++) assert.ok(C.ce_grad[i] >= C.ce_grad[i - 1], 'the CE gradient rises toward zero');
});

test('bad labels and logits give the recorded errors', () => {
  const e = Object.fromEntries(data.errors.map((x) => [x.id, x]));
  assert.deepEqual(Object.keys(e), ['not-integer', 'out-of-range', 'rank-1-logits', 'length', 'column-targets']);
  for (const id of ['not-integer', 'out-of-range', 'rank-1-logits', 'length']) assert.equal(e[id].exception, 'ShapeError', id);
  assert.equal(e['not-integer'].what, 'CrossEntropyLoss target at index 0 must be an integer');
  assert.equal(e['out-of-range'].what, 'CrossEntropyLoss target 3 at index 0 is out of range [0, 3)');
  assert.equal(e['column-targets'].exception, 'none', 'labels of shape (B, 1) are accepted');
});

test('every number in the prose is bound to the trace, with no hand-typed digits', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const all = parts.map((p) => p.html).join('\n');
  assert.match(all, /ShapeError: CrossEntropyLoss target at index 0 must be an integer/);
  assert.match(all, /ShapeError: CrossEntropyLoss target 3 at index 0 is out of range \[0, 3\)/);
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<div class="ls2-eq"[\s\S]*?<\/div>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>')
    .replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/loss-widgets.js');
  assert.equal(typeof m.mountLossExplorer, 'function');
  assert.equal(typeof m.mountGradientCompare, 'function');
  assert.equal(typeof document, 'undefined');
});
