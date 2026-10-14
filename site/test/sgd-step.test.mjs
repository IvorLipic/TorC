// Run: node --test site/test/
// The sgd-step lesson: the model against TorC's surface samples, hand formulas for transpose/mse, the widgets' pure helpers, quoted source
// lines, the claims the prose makes (each one asserted against the recorded runs), and "modules import in Node without a DOM".
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { buildGraph, runBackward } from '../js/engine.js';
import { makeModel, simulate, trainingLoop } from '../js/optim.js';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import lesson from '../js/lessons/sgd-step.js';
import { gridFor, contourSegments, niceLevels, fitView, chooseView, heatColor } from '../js/widgets/contour2d.js';
import { LR_PRESETS, MU_PRESETS, STEPS } from '../js/widgets/descent.js';
import { ITERATIONS } from '../js/widgets/loop-timeline.js';
import { yRange, fmt as sfmt } from '../js/widgets/sgd-ui.js';
import { SOURCES } from '../js/widgets/sgd-code.js';
import { readJson, SITE, lessonStats } from './helpers.mjs';

const O = readJson('data/optim.json');
const D = buildD({ optim: O });
const T = makeT(D);
const scene = O.scenes[0];
const model = makeModel(scene);
const near = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
const run = (id) => O.runs.find((r) => r.id === id);
const rowsOf = (r) => [...r.steps, r.final];

// ---- the scene and the model ---------------------------------------------------------------------------------------
test('engine: the sgd scene (transpose, matmul, add, mse) equals the real run, values and gradients', () => {
  const g = buildGraph(scene), bw = runBackward(g);
  for (const n of scene.nodes) {
    assert.ok(g.byName[n.name].value.v.every((v, i) => near(v, n.value.values[i])), `${n.name} value`);
    if (n.grad) assert.ok(bw.grad[n.name].v.every((v, i) => near(v, n.grad.values[i])), `${n.name} grad`);
  }
  assert.deepEqual(scene.nodes.filter((n) => n.role === 'op').map((n) => n.op), ['transpose', 'matmul', 'add', 'mse']);
  assert.equal(scene.parity.output_equal && scene.parity.grads_equal && scene.parity.loss_equal, true);
  assert.equal(O.checks.parity_output_and_grads_equal, true);
});

test('the mse and transpose rules equal the hand formulas and the scene gradients', () => {
  const { x, t, n } = model;
  const g = buildGraph(scene), bw = runBackward(g);
  const e = x.map((xi, i) => 0 * xi + 0 - t[i]);                       // y - t at w = b = 0
  const dy = e.map((v) => (2 * v) / n);                                // mse backward: 2 (y - t) / N
  const trace = Object.fromEntries(scene.nodes.map((q) => [q.name, q]));
  dy.forEach((v, i) => { assert.ok(near(v, trace.y.grad.values[i])); assert.ok(near(v, bw.grad.xW.v[i])); });
  const gW = dy.reduce((s, v, i) => s + v * x[i], 0), gb = dy.reduce((s, v) => s + v, 0);   // matmul: x^T dy ; add: sum over rows
  assert.ok(near(gW, trace.W.grad.values[0]) && near(gb, trace.b.grad.values[0]));
  assert.ok(near(bw.grad.Wt.v[0], bw.grad.W.v[0]), 'transpose passes the gradient on (a 1x1 swap)');
  assert.ok(near(model.eval(0, 0).loss, trace.L.value.values[0]));
  assert.ok(near(model.eval(0, 0).gw, trace.W.grad.values[0]) && near(model.eval(0, 0).gb, trace.b.grad.values[0]));
});

test('surface conformance: the engine loss equals the 25 loss values TorC printed', () => {
  assert.equal(O.surface.points.length, 25);
  for (const p of O.surface.points) assert.ok(near(model.loss(p.w, p.b), p.loss), `(${p.w}, ${p.b}): ${model.loss(p.w, p.b)} vs ${p.loss}`);
  assert.ok(near(model.loss(O.data.optimum.w, O.data.optimum.b), O.data.optimum.loss, 1e-9));
});

// ---- contour helpers ---------------------------------------------------------------------------------------------------
test('contour2d: marching squares puts every segment end on the requested level', () => {
  const grid = gridFor(model, [-1, 5], [-1.5, 3], 64, 56);
  const levels = niceLevels(grid.zmin, grid.zmax);
  assert.ok(levels.length >= 4 && levels.length <= 9, `levels ${levels}`);
  for (const lev of levels) {
    const segs = contourSegments(grid, lev);
    assert.ok(segs.length > 0, `level ${lev} has segments`);
    for (const [a, b] of segs) for (const [px, py] of [a, b]) {
      const real = model.loss(px, py);
      assert.ok(Math.abs(real - lev) <= 0.04 * lev + 0.02, `level ${lev}: loss at a segment end is ${real}`);   // linear interpolation on a quadratic
    }
  }
  assert.deepEqual(niceLevels(0, 5), [1, 2]);
});

test('contour2d: equal pixel scale, view selection and colour map', () => {
  const f = fitView([-1, 5], [-1.5, 3], 506, 384);
  assert.ok(near((506) / (f.w[1] - f.w[0]), 384 / (f.b[1] - f.b[0]), 1e-9), 'one unit of w is as long as one unit of b');
  assert.ok(f.w[0] <= -1 && f.w[1] >= 5 && f.b[0] <= -1.5 && f.b[1] >= 3);
  const base = { w: [-1, 5], b: [-1.5, 3] };
  assert.equal(chooseView(base, [{ w: 0, b: 0 }, { w: 4, b: 2 }]).zoomedOut, false);
  const wide = chooseView(base, [{ w: 0, b: 0 }, { w: -12.7, b: -5.6 }]);
  assert.ok(wide.zoomedOut && wide.w[0] < -12.7 && wide.w[1] >= 5);
  const c = (v) => heatColor(v, 100).match(/\d+/g).map(Number);
  assert.deepEqual(c(0), [20, 33, 80]);
  assert.ok(c(100).every((x, i) => x === [244, 162, 97][i]));
  assert.ok(c(1)[1] < c(50)[1] || c(1)[0] < c(50)[0], 'colour changes with the loss');
});

// ---- the claims the lesson makes about the recorded runs -------------------------------------------------------------------
test('the run order the lesson relies on, and the widget presets are recorded runs', () => {
  assert.deepEqual(O.runs.map((r) => [r.lr, r.momentum]), [[0.05, 0], [0.1, 0], [0.15, 0], [0.2, 0], [0.05, 0.5], [0.05, 0.9]]);
  for (const lr of LR_PRESETS) assert.ok(O.runs.some((r) => r.lr === lr && r.momentum === 0), `lr ${lr}`);
  for (const mu of MU_PRESETS) assert.ok(O.runs.some((r) => r.lr === 0.05 && r.momentum === mu), `mu ${mu}`);
  assert.equal(STEPS, 10); assert.equal(ITERATIONS, 10);
  assert.equal(O.runs[0].steps.length, STEPS, '"ten updates"');
  assert.equal(O.no_zero_grad.training_loop.iterations.length, ITERATIONS, '"iteration ten"');
});

test('what the prose says is true of the runs: steady, divergent, momentum overshoots, zero_grad matters', () => {
  const plain = rowsOf(O.runs[0]).map((x) => x.loss), big = rowsOf(O.runs[3]).map((x) => x.loss);
  for (let i = 1; i < plain.length; i++) assert.ok(plain[i] < plain[i - 1], 'lr 0.05 falls at every update');
  for (let i = 1; i < big.length; i++) assert.ok(big[i] > big[i - 1], 'lr 0.2 grows at every update');
  const w = (r) => rowsOf(r).map((x) => x.w), sw = (ws) => ws.slice(1).map((x, i) => Math.abs(x - ws[i]));
  const s20 = sw(w(O.runs[3]));
  for (let i = 1; i < s20.length; i++) assert.ok(s20[i] > s20[i - 1], 'every step overshoots further');
  assert.ok(O.hessian.critical_lr > 0.15 && O.hessian.critical_lr < 0.2, 'the limit sits between the converging and diverging presets');
  assert.ok(Math.abs(O.hessian.critical_lr - 2 / O.hessian.eigenvalues[0]) < 1e-12, 'two over the largest curvature');
  const m5 = O.runs[4].steps, p = O.runs[0].steps;
  for (const k of ['w', 'b', 'loss']) assert.equal(m5[1][k], p[1][k], 'the first update with momentum is the plain one');
  assert.ok(m5[2].loss < p[2].loss && m5[3].loss > m5[2].loss, 'ahead after two updates, then up');
  const m9 = O.runs[5].steps;
  assert.ok(m9[2].w - m9[1].w > p[2].w - p[1].w, 'mu 0.9: the second update moves W more than plain SGD');
  const acc = O.no_zero_grad.accumulate_only;
  acc.forEach((a, i) => assert.ok(near(a.gw, (i + 1) * acc[0].gw)));
  const noZ = O.no_zero_grad.training_loop.iterations;
  assert.ok(noZ[9].loss > 1000 * p[9].loss, 'without zero_grad the loop never settles');
  assert.ok(near(-0.05 * acc[2].gw, 3.4), 'the c-zero answer');
  O.data.x.forEach((x, i) => assert.equal(O.data.t[i], O.data.true_w * x + O.data.true_b, 'the points lie on t = 2x + 1'));
  assert.deepEqual(O.skipped_step.map((s) => [s.changed, s.threw]), [[false, false], [false, false], [false, false], [true, false]], 'step() silently skips a parameter without a gradient');
});

test('the widgets reproduce the recorded runs they show', () => {
  for (const r of O.runs) {
    const run = simulate(model, { lr: r.lr, momentum: r.momentum, steps: STEPS }), rows = [...run.steps, run.final];
    rowsOf(r).forEach((x, i) => assert.ok(near(rows[i].loss, x.loss, 1e-4) && near(rows[i].w, x.w, 1e-4), `${r.id} update ${i}`));
  }
  const noZ = trainingLoop(model, { lr: O.no_zero_grad.training_loop.lr, iterations: ITERATIONS, zeroGrad: false });
  O.no_zero_grad.training_loop.iterations.forEach((x, i) => assert.ok(near(noZ[i].loss, x.loss, 1e-4) && near(noZ[i].gw, x.gw, 1e-4), `loop ${i}`));
});

test('yRange and the number format', () => {
  assert.deepEqual(yRange([27.67, 0.001145]), [-3, 2]);
  assert.deepEqual(yRange([1, 5]), [0, 2]);
  assert.deepEqual(yRange([27.67, 1442, NaN, Infinity]), [1, 4]);
  assert.equal(sfmt(1.2345e18), '1.23e18');
  assert.equal(sfmt(27.67), '27.67');
  assert.equal(sfmt(NaN), 'not a number');
});

// ---- quoted library source ----------------------------------------------------------------------------------------------------
const SRC = new URL('../build/trace/_deps/torc-src/', SITE);
test('quoted library lines equal the library source at the pinned ref (skipped when the fetched source is absent)', (t) => {
  if (!existsSync(SRC)) { t.skip('build/trace/_deps/torc-src is not present (run tools/trace/run.sh to fetch it)'); return; }
  for (const [key, s] of Object.entries(SOURCES)) {
    const file = readFileSync(new URL(s.file, SRC), 'utf8').split('\n');
    assert.deepEqual(s.lines, file.slice(s.from - 1, s.to), `${key}: ${s.file} lines ${s.from}-${s.to}`);
  }
});

const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
const unescape = (x) => x.replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/&amp;/g, '&');

test('every inline quote in the lesson is a line of SOURCES, and they are the intended lines', () => {
  const quotes = [...html.matchAll(/<code class="quote" data-src="(\w+):(\d+)">([^<]*)<\/code>/g)];
  assert.deepEqual(quotes.map((q) => `${q[1]}:${q[2]}`), ['step:27', 'step:24', 'accumulate:83', 'step:20']);
  for (const [, key, n, text] of quotes) assert.equal(unescape(text), SOURCES[key].lines[n - SOURCES[key].from].trim());
  const line = (key, n) => SOURCES[key].lines[n - SOURCES[key].from].trim();
  assert.equal(line('step', 20), 'if (!param->has_grad()) continue;');
  assert.equal(line('step', 27), 'param->data() = param->data().sub(grad.mul(lr_));');
  assert.match(line('step', 24), /velocities_\[i\]\.mul\(momentum_\)\.add\(grad\)/);
  assert.equal(line('accumulate', 83), 'grad_.data()[i] += local_grad.data()[i];');
  assert.ok(html.includes(['loss.backward();', 'optimizer.step();', 'optimizer.zero_grad();'].join('\n')), 'the loop lines');
});

// ---- lesson text ----------------------------------------------------------------------------------------------------------------
test('every number in the lesson is bound to the trace, with the values the argument rests on', () => {
  for (const { section, html: h } of renderLesson(lesson, D)) {
    assert.deepEqual(findUnboundNums(h), [], section.kind);
    for (const { path, shown } of boundNums(h)) assert.equal(T.text(path), shown, path);
  }
  const shown = Object.fromEntries(boundNums(html).map(({ path, shown: v }) => [path, v]));
  const r = 'files.optim.runs';
  assert.deepEqual([shown['sgd.L.value'], shown['files.optim.data.true_w'], shown[`${r}.0.steps.1.w`], shown[`${r}.0.steps.1.loss`], shown[`${r}.3.final.loss`], shown[`${r}.4.steps.3.loss`]],
    ['27.67', '2', '1.133', '5.489', '1442', '2.447']);
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

test('no hand-typed digits in the prose (numbers come from the trace)', () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  let raw;
  try { raw = renderLesson(lesson, D).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<code[^>]*>[\s\S]*?<\/code>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
});

test('widget modules import in Node without a DOM', async () => {
  assert.equal(typeof document, 'undefined');
  assert.equal(typeof (await import('../js/widgets/descent.js')).mountDescent, 'function');
  assert.equal(typeof (await import('../js/widgets/loop-timeline.js')).mountLoopTimeline, 'function');
  assert.equal(typeof (await import('../js/widgets/contour2d.js')).mountContour2d, 'function');
  assert.equal(typeof document, 'undefined');
});
