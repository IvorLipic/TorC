// Lesson nonlinearity: the claims of the page against what TorC recorded (data/nonlinearity.json), every number bound to the
// trace, no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/nonlinearity.json');
const D = buildD({ nonlinearity: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/nonlinearity.js')).default;
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * (1 + Math.abs(b));

test('relu: max(0, z); derivative 1 for z > 0 and 0 for z <= 0 (TorC uses 0 at exactly zero)', () => {
  const [relu] = data.activations;
  data.z.forEach((z, i) => {
    assert.equal(relu.value[i], Math.max(0, z));
    assert.equal(relu.grad[i], z > 0 ? 1 : 0, `z = ${z}`);
  });
  assert.equal(data.z[40], 0);
  assert.deepEqual([data.z[20], data.z[70], data.z[80]].map((v) => Math.round(v * 10) / 10), [-2, 3, 4], 'the z the prose and checks quote');
});

test('sigmoid: 1/(1 + e^-z), derivative s(1 - s), largest at zero (0.25), small at z = 4', () => {
  const sig = data.activations[1];
  data.z.forEach((z, i) => {
    assert.ok(near(sig.value[i], 1 / (1 + Math.exp(-z))), `value at ${z}`);
    assert.ok(near(sig.grad[i], sig.value[i] * (1 - sig.value[i]), 1e-5), `grad at ${z}`);
  });
  assert.equal(Math.max(...sig.grad), sig.grad[40]);
  assert.equal(sig.grad[40], 0.25);
  assert.ok(sig.grad[80] < 0.02);
});

test('softmax: rows add up to one; a constant shift and very large inputs give the same result; equal inputs share equally', () => {
  const [small, shifted, large, equal] = data.softmax;
  for (const r of data.softmax) assert.ok(near(r.output.reduce((a, b) => a + b, 0), 1));
  assert.deepEqual(shifted.output, small.output);
  assert.deepEqual(large.output, small.output);
  assert.ok(equal.output.every((v) => near(v, 1 / 3)));
});

test('fit: the linear model ends as the best straight line (flat at the mean), the relu model bends and fits far better', () => {
  const [lin, net] = data.models;
  assert.deepEqual(lin.checkpoints.map((c) => c.step), [0, 25, 100, data.setup.steps]);
  assert.ok(lin.derived.max_abs_diff_line_vs_model < 1e-4, 'two linear layers are one line');
  assert.ok(Math.abs(lin.derived.w_eff) < 1e-4, 'the line is flat');
  assert.ok(near(lin.derived.b_eff, data.derived.best_line_intercept, 1e-5), 'at the mean of the targets');
  const linEnd = lin.checkpoints.at(-1).loss, netEnd = net.checkpoints.at(-1).loss;
  assert.ok(netEnd * 100 < linEnd, `${netEnd} vs ${linEnd}`);
  assert.equal(data.setup.points, data.data.x.length);
  data.data.x.forEach((x, i) => assert.ok(near(data.data.t[i], x * x, 1e-5)));
  assert.deepEqual(net.checkpoints[0].prediction.length, data.data.grid.length);
  // the start is a tilted line, so the widget shows the linear model turning flat
  const start = lin.checkpoints[0].prediction;
  assert.ok(start.at(-1) - start[0] > 1, `start slope ${start[0]} .. ${start.at(-1)}`);
  // the prose quotes the last prediction at x = 0 (grid index 22) as the flat line
  assert.equal(data.data.grid[22], 0);
  assert.ok(near(lin.checkpoints.at(-1).prediction[22], data.derived.best_line_intercept, 1e-5));
});

test('the input grids are exact: built from integers, so 0 is exactly 0 on every machine', () => {
  data.z.forEach((z, k) => assert.equal(Math.fround(z), Math.fround((k - 40) / 10), `z[${k}]`));
  data.data.x.forEach((x, k) => assert.equal(Math.fround(x), Math.fround((k - 10) / 5), `x[${k}]`));
  data.data.grid.forEach((x, k) => assert.equal(Math.fround(x), Math.fround((k - 22) / 10), `grid[${k}]`));
});

test('the derivative of relu is drawn in two pieces, with no line across the jump at zero', async () => {
  const { pieces } = await import('../js/widgets/nonlinear-widgets.js');
  const [relu, sig] = data.activations;
  const parts = pieces(data.z, relu.grad);
  assert.equal(parts.length, 2);
  assert.deepEqual(parts.map((p) => [p[0][0], p.at(-1)[0]]), [[data.z[0], 0], [data.z[41], data.z[80]]]);
  assert.equal(pieces(data.z, sig.grad).length, 1, 'sigmoid has no jump');
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
  const m = await import('../js/widgets/nonlinear-widgets.js');
  assert.equal(typeof m.mountActivationPlot, 'function');
  assert.equal(typeof m.mountCurveFit, 'function');
  assert.equal(typeof document, 'undefined');
});
