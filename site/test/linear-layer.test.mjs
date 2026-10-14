// Tests of the `linear-layer` lesson: the recorded TorC data, the formulas the page states (forward, backward, parameter count),
// the signal data of the init widget, the binding of every prose number, the style budget, KaTeX validity and DOM-free imports.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, runBackward } from '../js/engine.js';
import { buildD, makeT, resolvePath } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import { readJson, close, katex, lessonStats } from './helpers.mjs';
import lesson from '../js/lessons/linear-layer.js';
import { linearShapes, paramCount } from '../js/widgets/linear-shapes.js';
import { exampleFromScene, linearForward, gradW, gradB, gradX, colSums, rowsOf, neuronTex } from '../js/widgets/linear-math.js';
import { signalSeries, signalTable, growthRatios, fmtSci } from '../js/widgets/linear-signal-data.js';

const data = readJson('data/linear.json');
const D = buildD({ linear: data });
const T = makeT(D);
const scene = (id) => D.scenes[id];
const node = (s, name) => s.nodes.find((n) => n.name === name);
const sum = (a) => a.reduce((x, y) => x + y, 0);

// ---------------------------------------------------------------------------------------------- recorded data
test('engine reproduces scenes `linear` and `linear-dx` (values and gradients)', () => {
  for (const id of ['linear', 'linear-dx']) {
    const s = scene(id);
    const g = buildGraph(s), bw = runBackward(g);
    for (const n of s.nodes) {
      assert.deepEqual(g.byName[n.name].value.shape, n.value.shape, `${id}.${n.name} shape`);
      assert.ok(close(g.byName[n.name].value.v, n.value.values), `${id}.${n.name} value`);
      if (n.grad) assert.ok(close(bw.grad[n.name].v, n.grad.values), `${id}.${n.name} grad`);
      else assert.ok(!bw.grad[n.name], `${id}.${n.name} has no stored grad`);
    }
  }
});

test('parity flags: the real nn::Linear equals the hand-built graph (both scenes) and all internal checks passed', () => {
  for (const id of ['linear', 'linear-dx']) {
    const p = scene(id).parity;
    assert.equal(p.output_equal, true, id);
    assert.equal(p.grads_equal, true, id);
    assert.equal(p.loss_equal, true, id);
    assert.equal(p.max_abs_diff, 0, id);
  }
  assert.equal(data.checks.all_internal_checks_passed, true);
});

test('finite-difference checks are present and ok for every leaf of both scenes', () => {
  for (const id of ['linear', 'linear-dx']) {
    const fd = scene(id).finite_difference;
    assert.ok(fd && fd.leaves.length >= 2, `${id}: finite_difference present`);
    for (const l of fd.leaves) assert.equal(l.ok, true, `${id}.${l.name} finite difference`);
  }
});

test('hand formulas equal the library: gW = dY^T x, gb = column sums of dY, dX = column sums of W', () => {
  const s = scene('linear');
  const ex = exampleFromScene(s);
  const dY = rowsOf(node(s, 'y').grad.values, 2, 2);
  assert.ok(close(gradW(dY, ex.x).flat(), node(s, 'W').grad.values), 'gW');
  assert.ok(close(gradB(dY), node(s, 'b').grad.values), 'gb');
  assert.ok(data.example.gW_equals_dYT_x_max_abs_diff < 1e-5);
  assert.ok(data.example.gb_equals_colsum_dY_max_abs_diff < 1e-5);
  // dY is errors/2 (N = 4 outputs)
  assert.ok(close(dY.flat(), data.example.dY.values) && close(data.example.errors.values.map((e) => e / 2), data.example.dY.values));
  // loss = sum(y), x trainable
  const dx = scene('linear-dx');
  const exdx = exampleFromScene(dx);
  const ones = [[1, 1], [1, 1]];
  assert.ok(close(gradX(ones, exdx.W).flat(), node(dx, 'x').grad.values), 'dX by formula');
  const cs = colSums(exdx.W);
  for (let i = 0; i < 2; i++) assert.ok(close(node(dx, 'x').grad.values.slice(i * 3, i * 3 + 3), cs), `row ${i} of dX equals the column sums of W`);
  assert.ok(data.example.dX_equals_colsum_W_max_abs_diff < 1e-5);
  // the bias gradient is the batch size, W's rows are the column sums of x
  assert.deepEqual(node(dx, 'b').grad.values, [2, 2]);
});

test('JS forward arithmetic equals the printed y of the lesson example', () => {
  const ex = exampleFromScene(scene('linear'));
  assert.ok(close(linearForward(ex.x, ex.W, ex.b).flat(), ex.y.flat()));
  assert.ok(close(linearForward(ex.x, ex.W, ex.b).flat(), node(scene('linear'), 'y').value.values));
});

test('parameter counts: every layer and every total equals the formula', () => {
  assert.ok(data.param_counts.length >= 3);
  for (const e of data.param_counts) {
    let total = 0, tensors = 0;
    for (const L of e.layers) {
      const lin = /^Linear\((\d+), (\d+)\)$/.exec(L.layer);
      const conv = /^Conv2d\((\d+), (\d+), kernel (\d+)/.exec(L.layer);
      if (lin) assert.equal(L.count, Number(lin[1]) * Number(lin[2]) + Number(lin[2]), `${e.name}: ${L.layer} in*out+out`);
      else if (conv) assert.equal(L.count, Number(conv[2]) * Number(conv[1]) * Number(conv[3]) ** 2 + Number(conv[2]), `${e.name}: ${L.layer} out*in*k*k+out`);
      else assert.equal(L.count, 0, `${e.name}: ${L.layer} has no parameters`);
      assert.equal(L.count, sum(L.params.map((p) => p.numel)));
      for (const p of L.params) assert.equal(p.numel, p.shape.reduce((a, b) => a * b, 1));
      total += L.count; tensors += L.params.length;
    }
    assert.equal(e.total, total, `${e.name}: total = sum of layers`);
    assert.equal(e.sum_of_layers, e.total);
    assert.equal(e.tensors, tensors);
  }
  const lin = data.param_counts.find((e) => e.name === 'Linear(784, 32)');
  assert.equal(lin.total, paramCount(784, 32).total);
  assert.equal(lin.total, 25120);
  assert.equal(data.param_counts.find((e) => e.name === 'MNIST MLP').total, 26506);
});

test('signal table: six layers per init, band over other seeds, Kaiming stays near its start and the others do not', () => {
  assert.deepEqual(data.signal.map((s) => s.name), ['kaiming', 'std_0.01', 'std_1.0']);
  const series = signalSeries(data.signal);
  for (const s of series) {
    assert.equal(s.points.length, 7);
    assert.equal(s.band.length, 6);
    for (const b of s.band) assert.ok(b.min <= b.mean && b.mean <= b.max);
    for (const q of s.points) assert.ok(q.std > 0 && Number.isFinite(q.std));
  }
  const [k, small, big] = series;
  assert.ok(k.points.slice(1).every((q) => q.std > 0.6 && q.std < 1.2), 'Kaiming run stays within a factor 2 of 1');
  assert.ok(k.band.every((b) => b.min > 0.6 && b.max < 1.0), 'Kaiming across seeds stays 0.6 .. 1.0');
  assert.ok(small.points[6].std < 1e-5 && big.points[6].std > 1e5, 'too small fades, too big explodes');
  const growth = growthRatios(big), shrink = growthRatios(small);
  assert.equal(growth.length, 5);
  assert.ok(growth.every((r) => r > 10 && r < 13) && shrink.every((r) => r > 0.1 && r < 0.13), 'roughly constant factor per layer');
  const t = signalTable(series);
  assert.equal(t.rows.length, 7);
  assert.ok(t.rows.every((r) => r.length === t.columns.length));
});

// ---------------------------------------------------------------------------------------------- widgets' pure parts
test('shapes the check answers with, and the KaTeX of the grid (y[1,2] = -0.7 with the real numbers)', () => {
  assert.deepEqual(linearShapes(784, 32, 8), { x: [8, 784], W: [32, 784], Wt: [784, 32], b: [32], y: [8, 32] });
  const ex = exampleFromScene(scene('linear'));
  for (let i = 0; i < 2; i++) for (let k = 0; k < 2; k++) katex.renderToString(neuronTex(ex.x, ex.W, ex.b, i, k), { throwOnError: true, displayMode: true });
  assert.match(neuronTex(ex.x, ex.W, ex.b, 0, 1), /= -0\.7\\end\{aligned\}$/);
  // "copy the first neuron into the second" (weights and bias) makes the two output columns identical
  const W = [ex.W[0].slice(), ex.W[0].slice()], b = [ex.b[0], ex.b[0]];
  assert.ok(linearForward(ex.x, W, b).every((r) => r[0] === r[1]));
  // raising b[2] by one moves only the second column, by exactly one
  const y0 = linearForward(ex.x, ex.W, ex.b), y1 = linearForward(ex.x, ex.W, [ex.b[0], ex.b[1] + 1]);
  y0.forEach((r, i) => { assert.equal(y1[i][0], r[0]); assert.ok(Math.abs(y1[i][1] - r[1] - 1) < 1e-12); });
});

// ---------------------------------------------------------------------------------------------- lesson text
const html = renderLesson(lesson, D).map((p) => p.html).join('\n');

test('every number in the prose is bound to the trace (T() or sci()), with the values the argument rests on', () => {
  for (const { section, html: h } of renderLesson(lesson, D)) {
    assert.deepEqual(findUnboundNums(h), [], section.kind);
    for (const { path, shown } of boundNums(h)) assert.equal(T.text(path), shown, path);
  }
  const sci = [...html.matchAll(/<span class="rec" data-trace="([^"]+)" data-fmt="sci">([^<]*)<\/span>/g)];
  assert.equal(sci.length, 2);
  for (const [, path, shown] of sci) assert.equal(fmtSci(resolvePath(D, path)), shown, path);
  const shown = Object.fromEntries(boundNums(html).map(({ path, shown: v }) => [path, v]));
  const P = 'files.linear.param_counts.0';
  assert.deepEqual([shown[`${P}.layers.0.params.1.numel`], shown[`${P}.layers.0.params.0.numel`], shown[`${P}.total`]], ['25088', '32', '25120']);
  assert.deepEqual([0, 1, 2].map((i) => shown[`linear-dx.x.grad.${i}`]), colSums(exampleFromScene(scene('linear')).W).map(String), 'the predict answer: column sums of W');
  assert.match(html, /ShapeError: matmul inner dimensions must match, got \(8, 10\) and \(784, 32\)/);
});

test('the lesson keeps to the style budget (docs/authoring.md)', () => {
  const st = lessonStats(lesson, D, renderLesson, boundNums);
  assert.ok(st.words <= 800, `${st.words} words`);
  assert.ok(st.numbers <= 25, `${st.numbers} numbers`);
  assert.ok(st.widgets <= 2, `${st.widgets} widgets`);
  assert.ok(st.questions <= 3, `${st.questions} questions`);
  assert.equal(lesson.sections[1].kind, 'tldr');
  assert.equal(lesson.footer, undefined);
  assert.doesNotMatch(html, /recorded|recomputed|printed by|in your browser|show as table/i);
});

test('no hand-typed digits in the prose; every formula parses in KaTeX', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="(num|rec)"[^>]*>[^<]*<\/span>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
  for (const m of raw.matchAll(/<code>([^<]*)<\/code>/g)) if (/[\\^_{}]/.test(m[1])) katex.renderToString(m[1].replace(/&amp;/g, '&'), { throwOnError: true });
  assert.doesNotMatch(html, /katex-error/);
});

test('widget modules import in Node without touching the DOM', async () => {
  assert.equal(typeof document, 'undefined');
  assert.equal(typeof (await import('../js/widgets/matmul-grid.js')).mountMatmulGrid, 'function');
  assert.equal(typeof (await import('../js/widgets/init-signal.js')).mountInitSignal, 'function');
  assert.equal(typeof document, 'undefined');
});
