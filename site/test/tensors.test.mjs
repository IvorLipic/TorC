// Lesson tensors: the widget helpers against what TorC recorded (data/tensors.json), every number in the prose bound to the
// trace, no hand-typed digits, and the widget module importing without a DOM.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { alignShapes, sourceIndex, reducedCells } from '../js/widgets/tensor-widgets.js';
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

test('the shapes quoted in the check questions come from the trace', () => {
  assert.deepEqual(data.check_sum, { x_shape: [4, 5], sum0_shape: [5] });
  assert.deepEqual(data.broadcast.find((b) => b.id === 'column').b.shape, [2, 1]);
  assert.deepEqual(data.matmul.transpose.shape, [3, 2]);
  const all = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.match(all, /has shape <code>\(4, 5\)<\/code>/);
  assert.match(all, /The shape is <code>\(5\)<\/code>/);
});

test('every number in the prose is bound to the trace, with no hand-typed digits', () => {
  const parts = renderLesson(lesson, D);
  for (const { section, html } of parts) {
    assert.deepEqual(findUnboundNums(html), [], section.kind);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown, path);
  }
  const all = parts.map((p) => p.html).join('\n');
  const shown = Object.fromEntries(boundNums(all).map(({ path, shown: v }) => [path, v]));
  assert.deepEqual([shown['files.tensors.example.numel'], shown['files.tensors.example.index.5.flat'], shown['files.tensors.whole.sum']], ['6', '5', '21']);
  assert.match(all, /ShapeError: Cannot broadcast shapes \(2, 3\) and \(2\)/);
  assert.match(all, /ShapeError: matmul inner dimensions must match, got \(2, 3\) and \(2, 3\)/);
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('the widget module imports in Node without touching the DOM', async () => {
  const m = await import('../js/widgets/tensor-widgets.js');
  assert.equal(typeof m.mountBroadcastGrid, 'function');
  assert.equal(typeof m.mountAxisExplorer, 'function');
  assert.equal(typeof document, 'undefined');
});
