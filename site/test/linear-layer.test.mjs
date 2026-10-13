// Tests of the `linear-layer` lesson: the recorded TorC data, the pure logic of its widgets (shape explorer, grid, backward sums,
// initialisation data), the binding of every prose number, KaTeX validity of all paper work, and DOM-free imports of the widget modules.
// Run: node --test site/test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildGraph, runBackward } from '../js/engine.js';
import { buildD, makeT, resolvePath } from '../js/core/trace.js';
import { renderLesson } from '../js/core/sections.js';
import { readJson, close, katex } from './helpers.mjs';
import lesson from '../js/lessons/linear-layer.js';
import { SCENES, CALCS, calcText } from '../js/lessons/linear-layer.scenes.js';
import {
  LIMITS, fmtShape, linearShapes, paramCount, predictCall, findRecorded, recordedRuns, recordedLinearCounts, boxPx, sameShape,
} from '../js/widgets/linear-shapes.js';
import {
  exampleFromScene, linearForward, gradW, gradB, gradX, colSums, rowsOf, neuronTex, gradWTex, gradBTex, gradXTex,
} from '../js/widgets/linear-math.js';
import { histBins, normalCdf, signalSeries, signalTable, growthRatios, fmtSci, statsLabel, decadeTicks, expLabel } from '../js/widgets/linear-signal-data.js';

const data = readJson('data/linear.json');
const D = buildD({ linear: data });
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

test('init_stats: std within 5% of the expected value when n >= 1000; internal consistency of histograms', () => {
  for (const s of data.init_stats) {
    const expected = s.kaiming_default ? Math.sqrt(2 / s.in) : s.init_std;
    assert.ok(Math.abs(s.expected_std - expected) <= 1e-6 * expected, `${statsLabel(s)}: expected_std is ${expected}`);
    assert.equal(s.numel, s.in * s.out);
    assert.equal(s.bias_all_zero, true);
    const h = s.histogram;
    assert.equal(sum(h.counts) + h.under + h.over, s.numel, `${statsLabel(s)}: histogram counts add up`);
    assert.equal(h.counts.length, h.bins);
    assert.ok(Math.abs(h.hi - 4 * s.expected_std) < 1e-5 * s.expected_std && Math.abs(h.lo + 4 * s.expected_std) < 1e-5 * s.expected_std, 'histogram spans +-4 expected std');
    if (s.numel >= 1000) assert.ok(Math.abs(s.std / expected - 1) <= 0.05, `${statsLabel(s)}: std ${s.std} vs ${expected}`);
    if (s.weights) {
      const v = s.weights.values, m = sum(v) / v.length;
      assert.equal(v.length, s.numel);
      assert.ok(Math.abs(Math.sqrt(sum(v.map((x) => (x - m) ** 2)) / v.length) - s.std) < 1e-5 * s.std, `${statsLabel(s)}: shipped weights have the recorded std`);
    }
  }
  assert.ok(data.init_stats.some((s) => s.in === 784 && s.out === 32 && s.kaiming_default));
});

test('the three Linear(784, 32) cases are the same draws rescaled (identical histogram counts)', () => {
  const cases = data.init_stats.filter((s) => s.in === 784 && s.out === 32);
  assert.equal(cases.length, 3);
  for (const c of cases.slice(1)) assert.deepEqual(c.histogram.counts, cases[0].histogram.counts);
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

// ---------------------------------------------------------------------------------------------- shape logic
test('shape logic: shapes, parameter counts and limits', () => {
  assert.deepEqual(linearShapes(784, 32, 8), { x: [8, 784], W: [32, 784], Wt: [784, 32], b: [32], y: [8, 32] });
  assert.deepEqual(paramCount(3, 2), { weight: 6, bias: 2, total: 8, tensors: 2 });
  assert.equal(paramCount(784, 32).total, 25120);
  assert.equal(paramCount(32, 32).total, 1056);
  assert.equal(paramCount(32, 10).total, 330);
  assert.equal(fmtShape([2, 2]), '(2, 2)');
  assert.equal(fmtShape([3]), '(3)');
  assert.ok(LIMITS.in.min >= 1 && LIMITS.batch.max >= 8);
  assert.ok(boxPx(1) < boxPx(32) && boxPx(32) < boxPx(784) && boxPx(1024) <= 110);
  // shapes agree with what the library recorded for the lesson example
  const s = scene('linear');
  const sh = linearShapes(3, 2, 2);
  assert.deepEqual(node(s, 'x').value.shape, sh.x);
  assert.deepEqual(node(s, 'W').value.shape, sh.W);
  assert.deepEqual(node(s, 'Wt').value.shape, sh.Wt);
  assert.deepEqual(node(s, 'b').value.shape, sh.b);
  assert.deepEqual(node(s, 'y').value.shape, sh.y);
  const lin = data.param_counts[0].layers[0];
  assert.deepEqual(lin.params.find((p) => p.name === 'weight').shape, linearShapes(784, 32, 1).W);
  assert.deepEqual(lin.params.find((p) => p.name === 'bias').shape, linearShapes(784, 32, 1).b);
  assert.deepEqual(lin.output_shape, linearShapes(784, 32, 1).y);
});

test('error prediction reproduces every recorded Linear call (exception type, text, output shape)', () => {
  const widthCases = data.shape_errors.filter((r) => r.x_shape && r.x_shape.length === 2);
  assert.ok(widthCases.length >= 4);
  for (const r of data.shape_errors) {
    if (r.id === 'mse-shapes') { assert.match(r.what, /^Cannot broadcast shapes/); continue; }
    const call = predictCall(r.in, r.out, r.x_shape ?? [1, r.in]);
    if (r.exception === 'none') { assert.equal(call.ok, true, r.id); assert.deepEqual(call.y, r.y_shape, r.id); continue; }
    assert.equal(call.ok, false, r.id);
    assert.equal(call.exception, r.exception, r.id);
    assert.equal(call.what, r.what, r.id);
  }
  // the inner-dimension messages all match the documented text, the rank-1 one says rank >= 2
  for (const r of widthCases.filter((r) => r.exception === 'ShapeError')) assert.match(r.what, /inner dimensions must match/);
  assert.match(data.shape_errors.find((r) => r.id === 'linear-3x2-(3)').what, /rank >= 2/);
  // a matching width never fails; rank-3 is explicitly not modelled
  assert.equal(predictCall(5, 4, [7, 5]).ok, true);
  assert.equal(predictCall(5, 4, [2, 3, 5]).ok, null);
  assert.equal(predictCall(0, 4, [2, 3]).exception, 'TorcError');
});

test('recorded lookups: runs, exact-call matches and layer counts', () => {
  const runs = recordedRuns(data.shape_errors);
  assert.deepEqual(runs.map((r) => r.id), ['linear-3x2-(2, 2)', 'linear-784x32-(8, 10)', 'linear-3x2-(4, 4)', 'linear-3x2-(4, 3)'].map((id) => id.replace('linear-', 'linear-')));
  assert.equal(runs.filter((r) => !r.fails).length, 1);
  assert.equal(findRecorded(data.shape_errors, 3, 2, [2, 2]).exception, 'ShapeError');
  assert.equal(findRecorded(data.shape_errors, 3, 2, [2, 5]), null);
  assert.ok(sameShape([1, 2], [1, 2]) && !sameShape([1, 2], [1, 2, 3]));
  const counts = recordedLinearCounts(data.param_counts);
  assert.equal(counts.get('784,32').count, 25120);
  assert.equal(counts.get('32,32').count, 1056);
  assert.equal(counts.get('32,10').count, 330);
  // every preset the explorer offers that has a recorded count agrees with the formula
  for (const [key, v] of counts) { const [a, b] = key.split(',').map(Number); assert.equal(v.count, paramCount(a, b).total); }
});

// ---------------------------------------------------------------------------------------------- init data helpers
test('histogram helpers: expected counts add up and the normal CDF is accurate', () => {
  assert.ok(Math.abs(normalCdf(0) - 0.5) < 1e-7);
  assert.ok(Math.abs(normalCdf(1.96) - 0.9750021) < 1e-6);
  assert.ok(Math.abs(normalCdf(-1) - 0.1586553) < 1e-6);
  for (const s of data.init_stats.filter((q) => q.numel >= 1000)) {
    const bins = histBins(s.histogram, s.numel, s.expected_std);
    assert.equal(bins.length, 40);
    assert.ok(Math.abs(sum(bins.map((b) => b.expected)) / s.numel - 0.99994) < 5e-5, 'expected counts cover +-4 sigma');
    assert.equal(sum(bins.map((b) => b.count)), s.numel - s.histogram.under - s.histogram.over);
    // observed and expected agree overall: total variation distance below 5% for these sample sizes
    const tv = sum(bins.map((b) => Math.abs(b.count - b.expected))) / (2 * s.numel);
    assert.ok(tv < 0.6 * Math.sqrt(40 / s.numel), `${statsLabel(s)}: total variation ${tv} within sampling noise`);
  }
  assert.equal(fmtSci(0), '0');
  assert.equal(fmtSci(0.0123456), '0.01235');
  assert.equal(fmtSci(2.17e6), '2.17e6');
  assert.equal(fmtSci(1.59e-4), '1.59e-4');
  assert.deepEqual(decadeTicks(-5.4, 3.2, 2), [-5, -3, -1, 1, 3]);
  assert.equal(expLabel(-6), '1e-6');
});

// ---------------------------------------------------------------------------------------------- paper work
test('all paper-work strings of the widgets parse in KaTeX', () => {
  const s = scene('linear'), ex = exampleFromScene(s);
  const dY = rowsOf(node(s, 'y').grad.values, 2, 2);
  let n = 0;
  const ok = (tex) => { katex.renderToString(tex, { throwOnError: true, displayMode: true }); n++; };
  for (let i = 0; i < 2; i++) for (let k = 0; k < 2; k++) ok(neuronTex(ex.x, ex.W, ex.b, i, k));
  for (let k = 0; k < 2; k++) for (let j = 0; j < 3; j++) ok(gradWTex(dY, ex.x, k, j));
  for (let k = 0; k < 2; k++) ok(gradBTex(dY, k));
  const dx = exampleFromScene(scene('linear-dx'));
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) ok(gradXTex([[1, 1], [1, 1]], dx.W, i, j));
  assert.equal(n, 4 + 6 + 2 + 6);
  // the numbers inside are the right ones: y[1,2] = 1*1 + 2*0 + 3*(-0.5) + (-0.2) = -0.7
  assert.match(neuronTex(ex.x, ex.W, ex.b, 0, 1), /= -0\.7\\end\{aligned\}$/);
  assert.match(gradWTex(dY, ex.x, 0, 2), /= 1\.5\\end/);
  assert.match(gradBTex(dY, 1), /= -0\.45\\end/);
});

// ---------------------------------------------------------------------------------------------- graph-lab layouts
test('graph-lab layouts: every node of both scenes is positioned inside the view; teaching text carries no numerals', () => {
  assert.deepEqual(Object.keys(SCENES), ['linear', 'linear-dx']);
  for (const [id, L] of Object.entries(SCENES)) {
    const s = scene(id);
    for (const n of s.nodes) {
      const p = L.pos[n.name];
      assert.ok(p, `${id}: no position for ${n.name}`);
      assert.ok(p[0] > 48 && p[0] < L.view[0] - 48 && p[1] > 0 && p[1] < L.view[1], `${id}.${n.name} inside the view`);
    }
    assert.deepEqual(Object.keys(L.pos).sort(), s.nodes.map((n) => n.name).sort(), `${id}: no stray positions`);
    for (const key of ['goal', 'intro', 'forwardHint', 'backwardHint', 'takeaway']) {
      assert.ok(typeof L[key] === 'string' && L[key].length > 20, `${id}.${key}`);
      assert.doesNotMatch(L[key], /\d/, `${id}.${key} must not contain numerals`);
    }
    for (const a of L.above) assert.ok(L.pos[a], `${id}: above lists ${a}`);
  }
});

// ---------------------------------------------------------------------------------------------- the lesson text
test('lesson text: quoted numbers are bound to the trace and the derived ones recompute independently', () => {
  assert.equal(lesson.trace.join(), 'linear.json');
  const T = makeT(D);
  const parts = renderLesson(lesson, D);
  const html = parts.map((p) => p.html).join('\n');
  // numbers the plan and the note promise to quote (bound by T, so a change of the trace changes the page)
  const bound = (path, shown) => { assert.ok(html.includes(`data-trace="${path}">${shown}<`), `${path} should appear as ${shown}`); assert.equal(T.text(path), shown); };
  // exact integers (class "rec": the shared T() rounds to 4 significant digits, which would turn 26506 into 26510)
  const rec = [...html.matchAll(/<span class="rec" data-trace="([^"]+)"([^>]*)>([^<]*)<\/span>/g)];
  assert.ok(rec.length >= 10);
  for (const [, path, attrs, shown] of rec) {
    const v = resolvePath(D, path);
    assert.equal(shown, /data-fmt="sci"/.test(attrs) ? fmtSci(v) : String(v), `${path} is shown as recorded`);
    if (!/data-fmt="sci"/.test(attrs)) assert.ok(Number.isInteger(v), `${path} is an integer`);
  }
  for (const [path, shown] of [['files.linear.param_counts.0.total', '25120'], ['files.linear.param_counts.1.total', '26506'], ['files.linear.param_counts.0.layers.0.params.1.numel', '25088']]) {
    assert.ok(html.includes(`data-trace="${path}" style="color:var(--fwd);font-variant-numeric:tabular-nums">${shown}<`), `${path} should appear as ${shown}`);
  }
  bound('linear.y.value.0', '4.6');
  bound('linear.y.value.1', '-0.7');
  bound('linear.L.value', '0.3125');
  bound('linear.b.grad.0', '0.6');
  bound('linear.b.grad.1', '-0.45');
  bound('linear-dx.x.grad.0', '1.5');
  bound('linear-dx.x.grad.1', '-1');
  // no digit-only claim hides outside T: the lesson must not state a parameter total by hand
  assert.doesNotMatch(html.replace(/<span class="(?:num|rec)"[^>]*>[^<]*<\/span>/g, ''), /\b(25120|26506|25088|0\.3125)\b/);

  // derived (amber) numbers: recompute from the raw trace with separate code
  const sig = (name) => data.signal.find((s) => s.name === name);
  const post = (name) => sig(name).layers.slice(1).map((l) => l.post_std);
  const ratios = (name) => post(name).slice(1).map((v, i) => v / post(name)[i]);
  const expected = {
    'edit-y12': node(scene('linear'), 'y').value.values[1] + 1,
    'edit-y22': node(scene('linear'), 'y').value.values[3] + 1,
    'grow-lo': Math.min(...ratios('std_1.0')), 'grow-hi': Math.max(...ratios('std_1.0')),
    'shrink-lo': Math.min(...ratios('std_0.01')), 'shrink-hi': Math.max(...ratios('std_0.01')),
    'grow-pred': Math.sqrt(256 / 2) * 1.0, 'shrink-pred': Math.sqrt(256 / 2) * 0.01,
  };
  assert.deepEqual(Object.keys(CALCS).sort(), Object.keys(expected).sort(), 'every derived number has an independent check');
  const spans = [...html.matchAll(/<span class="calc" data-calc="([^"]+)"[^>]*>([^<]*)<\/span>/g)];
  assert.ok(spans.length >= 8, 'derived numbers are used in the lesson');
  for (const [, key, shown] of spans) assert.equal(shown, calcText(expected[key]), `derived ${key}`);
  // the human-readable claims behind them
  assert.equal(calcText(expected['edit-y12']), '0.3');
  assert.equal(calcText(expected['edit-y22']), '-1.2');
  assert.equal(calcText(expected['grow-pred']), '11.3');
  assert.ok(expected['grow-lo'] > 11 && expected['grow-hi'] < 12.5, 'growth per layer for init_std 1.0');
});

test('lesson text: every LaTeX string in the prose parses (KaTeX throwOnError)', () => {
  const real = globalThis.katex;
  globalThis.katex = { renderToString: (s, o) => real.renderToString(s, { ...o, throwOnError: true }) };
  try {
    const parts = renderLesson(lesson, D);
    const html = parts.map((p) => p.html).join('\n');
    assert.ok((html.match(/class="katex"/g) ?? []).length >= 20, 'the prose contains paper work');
    assert.doesNotMatch(html, /katex-error/);
  } finally { globalThis.katex = real; }
});

test('lesson structure: widgets, honest labels, checks', () => {
  const ids = lesson.sections.filter((s) => s.kind === 'widget').map((s) => s.id);
  assert.deepEqual(ids, ['shape-explorer', 'matmul-grid', 'linear-lab', 'linear-backward', 'init-hist', 'init-signal']);
  for (const s of lesson.sections.filter((q) => q.kind === 'widget')) assert.ok(s.label && s.className, `${s.id} has label and class`);
  assert.equal(lesson.sections.filter((s) => s.kind === 'check').length, 3);
  assert.ok(lesson.sections.some((s) => s.kind === 'gotcha'));
  assert.ok(lesson.sections.some((s) => s.kind === 'predict'));
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  // misconceptions of the plan are all addressed
  for (const needle of ['(in, out)', 'activation', 'Zero weights', 'Bigger initial weights', 'one bias per input', 'input needs a gradient']) assert.ok(html.toLowerCase().includes(needle.toLowerCase()), `misconception: ${needle}`);
  // no audit talk on a lesson page
  assert.doesNotMatch(html, /\bbug\b|audit|suspected|discrepan/i);
  assert.doesNotMatch(html, /session_|claude\.ai\/code/i);
});

// ---------------------------------------------------------------------------------------------- DOM-free imports
test('widget modules import in Node without touching the DOM', async () => {
  assert.equal(typeof document, 'undefined');
  const mods = {
    'shape-explorer': ['mountShapeExplorer'], 'matmul-grid': ['mountMatmulGrid'], 'linear-backward': ['mountLinearBackward'],
    'init-signal': ['mountWeightHistogram', 'mountInitSignal'], 'linear-ui': ['injectStyle', 'useBaseStyle', 'stepperField', 'segmented', 'matrixEl'],
  };
  for (const [file, names] of Object.entries(mods)) {
    const m = await import(new URL(`../js/widgets/${file}.js`, import.meta.url));
    for (const n of names) assert.equal(typeof m[n], 'function', `${file}.${n}`);
  }
  const ui = await import('../js/widgets/linear-ui.js');
  assert.doesNotThrow(() => ui.injectStyle('x', 'a{}'), 'injectStyle is a no-op without a document');
});
