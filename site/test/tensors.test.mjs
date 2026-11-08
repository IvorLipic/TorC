// Lesson tensors: the widget helpers against what TorC recorded (data/tensors.json), every number in the prose bound to the
// trace, no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { alignShapes, sourceIndex, reducedCells, reducedIndices } from '../js/widgets/tensor-widgets.js';
import { readJson } from './helpers.mjs';

const data = readJson('data/tensors.json');
const D = buildD({ tensors: data });
const T = makeT(D);
const lesson = (await import('../js/lessons/tensors.js')).default;
const A = data.example;

test('storage is row-major: a[i, j] sits at position 3i + j (recorded through Tensor::operator[])', () => {
  assert.deepEqual(A.shape, [2, 3]);
  for (const e of A.index) { assert.equal(e.flat, 3 * e.i + e.j); assert.equal(A.values[e.flat], e.value); }
  assert.equal(data.reshape.changing_the_copy_changes_a, false, 'reshape copies');
  assert.deepEqual(data.reshape.values, A.values);
  assert.equal(data.reshape.error.exception, 'ShapeError');
});

test('alignShapes and sourceIndex reproduce every recorded broadcast (result shape, values, or the error)', () => {
  for (const c of data.broadcast) {
    const al = alignShapes(A.shape, c.b.shape);
    assert.equal(al.ok, c.sum.exception === 'none', c.id);
    if (!al.ok) { assert.match(c.sum.what, /^Cannot broadcast shapes/); continue; }
    assert.deepEqual(al.out, c.sum.shape, c.id);
    const [R, C] = c.sum.shape;
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      assert.equal(A.values[sourceIndex(A.shape, [i, j])] + c.b.values[sourceIndex(c.b.shape, [i, j])], c.sum.values[i * C + j], `${c.id} [${i},${j}]`);
    }
  }
  for (const p of data.broadcast_shapes) {
    const al = alignShapes(p.x, p.y);
    assert.equal(al.ok, p.exception === 'none');
    if (al.ok) assert.deepEqual(al.out, p.result);
  }
  assert.deepEqual(data.broadcast_shapes[0].result, [4, 3], 'the check answer');
});

test('reducedCells reproduces every recorded reduction; the axis disappears from the shape', () => {
  const f = { sum: (v) => v.reduce((a, b) => a + b, 0), mean: (v) => v.reduce((a, b) => a + b, 0) / v.length, max: (v) => Math.max(...v) };
  for (const r of data.reductions) {
    assert.deepEqual(r.result.shape, [A.shape[1 - r.axis]]);
    r.result.values.forEach((v, k) => assert.equal(f[r.op](reducedCells(2, 3, r.axis, k).map(([i, j]) => A.values[i * 3 + j])), v, `${r.op}(${r.axis})[${k}]`));
  }
  assert.deepEqual(data.reduce_negative_axis.values, data.reductions.find((r) => r.op === 'sum' && r.axis === 1).result.values, 'axis -1 is the last axis');
  assert.match(data.reduce_error.what, /Invalid axis 2/);
});

test('matmul: (2, 3)(3, 2) gives (2, 2), the inner sizes must match, and batch dimensions broadcast', () => {
  const m = data.matmul;
  assert.deepEqual(m.a_at.shape, [2, 2]);
  assert.deepEqual(m.a_at.values, [14, 32, 32, 77]);
  assert.deepEqual(m.at_a.shape, [3, 3]);
  assert.match(m.a_a.what, /^matmul inner dimensions must match/);
  assert.deepEqual(m.batched.shape, [4, 2, 2]);
  // (4, 2, 3) holding 1..24 times aᵀ: batch k is rows 6k+1..6k+6 times aᵀ, computed here by hand
  const mm = (A, B, n, k, p) => Array.from({ length: n * p }, (_, ij) => { const i = Math.floor(ij / p), j = ij % p; let v = 0; for (let l = 0; l < k; l++) v += A[i * k + l] * B[l * p + j]; return v; });
  const at = m.transpose.values;
  for (let b = 0; b < 4; b++) {
    const A = Array.from({ length: 6 }, (_, q) => 6 * b + q + 1);
    assert.deepEqual(m.batched.values.slice(4 * b, 4 * b + 4), mm(A, at, 2, 3, 2), `batch ${b}`);
  }
  // (2, 1, 2, 3) times (3, 3, 2): both have batch dimensions, (2, 1) and (3) broadcast to (2, 3)
  assert.deepEqual(m.batched_both.shape, [2, 3, 2, 2]);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
    const A = Array.from({ length: 6 }, (_, q) => 6 * i + q + 1), B = Array.from({ length: 6 }, (_, q) => 6 * j + q + 1);
    assert.deepEqual(m.batched_both.values.slice(4 * (3 * i + j), 4 * (3 * i + j) + 4), mm(A, B, 2, 3, 2), `batch ${i},${j}`);
  }
});

test('a 3-D image: the position of image[c, i, j] is (c * 2 + i) * 2 + j, and the axis explorer reproduces every recorded reduction', () => {
  const I = data.image;
  assert.deepEqual(I.shape, [3, 2, 2]);
  assert.equal(I.numel, 12);
  for (const e of I.index) { assert.equal(e.flat, (e.c * 2 + e.i) * 2 + e.j); assert.equal(I.values[e.flat], e.value); }
  assert.deepEqual(I.index[10], { c: 2, i: 1, j: 0, flat: 10, value: 3 }, 'the example of the prose');
  const f = { sum: (v) => v.reduce((x, y) => x + y, 0), mean: (v) => v.reduce((x, y) => x + y, 0) / v.length, max: (v) => Math.max(...v) };
  for (const r of I.reductions) {
    assert.deepEqual(r.result.shape, I.shape.filter((_, k) => k !== r.axis), `${r.op}(${r.axis})`);
    r.result.values.forEach((v, k) => assert.equal(f[r.op](reducedIndices(I.shape, r.axis, k).map((q) => I.values[q])), v, `${r.op}(${r.axis})[${k}]`));
  }
  // the 2-D helper agrees with the n-d helper
  for (const axis of [0, 1]) for (let r = 0; r < A.shape[1 - axis]; r++) {
    assert.deepEqual(reducedIndices(A.shape, axis, r), reducedCells(2, 3, axis, r).map(([i, j]) => i * 3 + j));
  }
  assert.equal(I.sum, 60);
});

test('the worked 2 x 2 product and the entry of the "your turn" quiz', () => {
  const { p, q, c } = data.matmul_2x2;
  assert.deepEqual([p.shape, q.shape, c.shape], [[2, 2], [2, 2], [2, 2]]);
  const at = (m, i, j) => m.values[i * 2 + j];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) assert.equal(at(p, i, 0) * at(q, 0, j) + at(p, i, 1) * at(q, 1, j), at(c, i, j));
  assert.equal(c.values[0], 19, 'the worked entry');
  assert.equal(c.values[3], 50, 'the quiz answer');
  assert.deepEqual(data.index_errors.map((e) => e.exception), ['ShapeError', 'ShapeError', 'ShapeError']);
});

test('every claim of the prose against the data: shapes, results and messages', () => {
  assert.deepEqual(data.check_sum, { x_shape: [4, 5], sum0_shape: [5] });
  assert.deepEqual(data.broadcast.find((b) => b.id === 'column').b.shape, [2, 1]);
  assert.deepEqual(data.matmul.transpose.shape, [3, 2]);
  assert.deepEqual(data.matmul.batched.shape, [4, 2, 2], 'a (4, 2, 3) tensor times (3, 2)');
  assert.deepEqual(data.matmul.batched_both.shape, [2, 3, 2, 2]);
  assert.deepEqual(data.broadcast_shapes[1], { x: [4, 1, 3], y: [2, 1], result: [4, 2, 3], exception: 'none' });
  assert.equal(data.broadcast.find((b) => b.id === 'wrong-row').sum.exception, 'ShapeError', 'the predict quiz: (2) + (2, 3) fails');
  assert.deepEqual(data.reductions[0].result.shape, [3]);
  assert.deepEqual(data.reductions[3].result.shape, [2]);
  assert.deepEqual(data.image.reductions[0].result.shape, [2, 2], 'image.sum(0)');
  assert.match(data.reshape.error.what, /^Cannot reshape tensor of shape \(2, 3\) with 6 elements into shape \(4, 2\) with 8 elements$/);
  const all = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.match(all, /has the shape <code>\(4, 5\)<\/code>/);
  // H1: (4, 1, 3) + (2, 1): from the right 3 meets 1, 1 meets 2, and 4 meets a missing size
  assert.match(all, /From the right, <code>3<\/code> meets <code>1<\/code>, <code>1<\/code> meets <code>2<\/code>, and <code>4<\/code> meets a missing size/);
  const al = alignShapes([4, 1, 3], [2, 1]);
  assert.deepEqual([al.a, al.b], [[4, 1, 3], [null, 2, 1]]);
  assert.deepEqual(data.matmul.batched_a_shape, [4, 2, 3]);
  assert.match(all, /b\.reshape\(\{2, 1\}\)/);
  assert.match(all, /a\.matmul\(a\.transpose\(\{1, 0\}\)\)/);
  assert.match(all, /ShapeError: Cannot broadcast shapes \(2, 3\) and \(2\)/);
  assert.match(all, /ShapeError: matmul inner dimensions must match, got \(2, 3\) and \(2, 3\)/);
  assert.match(all, /ShapeError: Cannot reshape tensor of shape \(2, 3\) with 6 elements into shape \(4, 2\) with 8 elements/);
});

test('every number in the prose is bound to the trace', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const b of boundNums(html)) assert.equal(T.reformat(b), b.shown, b.path);
  }
  const all = parts.map((p) => p.html).join('\n');
  const shown = Object.fromEntries(boundNums(all).map(({ path, shown: v }) => [path, v]));
  assert.deepEqual([shown['files.tensors.example.numel'], shown['files.tensors.example.index.5.flat'], shown['files.tensors.whole.sum']], ['6', '5', '21']);
  assert.deepEqual([shown['files.tensors.image.index.10.flat'], shown['files.tensors.image.index.10.value']], ['10', '3']);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/tensor-widgets.js');
  assert.equal(typeof m.mountBroadcastGrid, 'function');
  assert.equal(typeof m.mountAxisExplorer, 'function');
  assert.equal(typeof m.reducedIndices, 'function');
  assert.equal(typeof document, 'undefined');
});
