// Run: node --test site/test/
// The sgd-step lesson (template 3): the model against TorC's surface samples, the hand formulas for transpose/mse, the widgets' pure
// helpers, the quoted SGD::step line, and every claim the prose and the guided tasks make, each one asserted against the recorded
// data (data/optim.json, data/why-gradients.json). The v3 rules themselves (roles, budgets, glossary) run in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { buildGraph, runBackward } from '../js/engine.js';
import { makeModel, simulate, trainingLoop } from '../js/optim.js';
import { renderLesson, boundNums } from '../js/core/sections.js';
import { fmt } from '../js/ops/texutil.js';
import lesson, { STEP_LINE } from '../js/lessons/sgd-step.js';
import { gridFor, contourSegments, niceLevels, fitView, chooseView, heatColor } from '../js/widgets/contour2d.js';
import { LR_PRESETS, STEPS, arrowsAt, r3 } from '../js/widgets/descent.js';
import { ITERATIONS } from '../js/widgets/loop-timeline.js';
import { yRange, fmt as sfmt } from '../js/widgets/sgd-ui.js';
import { SOURCES } from '../js/widgets/sgd-code.js';
import { readJson, SITE, course } from './helpers.mjs';

const O = readJson('data/optim.json');
const WG = readJson('data/why-gradients.json');
const D = buildD({ optim: O, 'why-gradients': WG }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const scene = O.scenes[0];
const model = makeModel(scene);
const near = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
const rowsOf = (r) => [...r.steps, r.final];
const BASE = { w: [-1, 5], b: [-1.5, 3] };                 // the view descent.js passes to contour2d
const [PLAIN, ZIG, BIG] = [0, 2, 3];                        // the run indices the lesson binds
const run = (i) => O.runs[i];
// coordinates of a point along the steep and the flat direction of the valley, measured from the lowest point
const [steepV, flatV] = O.hessian.eigenvectors;
const along = (p, v) => (p.w - O.data.optimum.w) * v[0] + (p.b - O.data.optimum.b) * v[1];
const rmse = (loss) => Math.sqrt(loss);

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
  const dy = x.map((xi, i) => (2 * (0 * xi + 0 - t[i])) / n);          // mse backward at w = b = 0: 2 (y - t) / N
  const trace = Object.fromEntries(scene.nodes.map((q) => [q.name, q]));
  dy.forEach((v, i) => { assert.ok(near(v, trace.y.grad.values[i])); assert.ok(near(v, bw.grad.xW.v[i])); });
  const gW = dy.reduce((s, v, i) => s + v * x[i], 0), gb = dy.reduce((s, v) => s + v, 0);
  assert.ok(near(gW, trace.W.grad.values[0]) && near(gb, trace.b.grad.values[0]));
  assert.ok(near(bw.grad.Wt.v[0], bw.grad.W.v[0]), 'transpose passes the gradient on (a 1x1 swap)');
});

test('surface conformance: the engine loss equals the 25 loss values TorC printed', () => {
  assert.equal(O.surface.points.length, 25);
  for (const p of O.surface.points) assert.ok(near(model.loss(p.w, p.b), p.loss), `(${p.w}, ${p.b}): ${model.loss(p.w, p.b)} vs ${p.loss}`);
  assert.ok(near(model.loss(O.data.optimum.w, O.data.optimum.b), O.data.optimum.loss, 1e-9));
});

// ---- contour helpers -------------------------------------------------------------------------------------------------
test('contour2d: marching squares puts every segment end on the requested level', () => {
  const grid = gridFor(model, BASE.w, BASE.b, 64, 56);
  const levels = niceLevels(grid.zmin, grid.zmax);
  assert.ok(levels.length >= 4 && levels.length <= 9, `levels ${levels}`);
  for (const lev of levels) {
    const segs = contourSegments(grid, lev);
    assert.ok(segs.length > 0, `level ${lev} has segments`);
    for (const [a, b] of segs) for (const [px, py] of [a, b]) {
      const real = model.loss(px, py);
      assert.ok(Math.abs(real - lev) <= 0.04 * lev + 0.02, `level ${lev}: loss at a segment end is ${real}`);
    }
  }
  assert.deepEqual(niceLevels(0, 5), [1, 2]);
});

test('contour2d: equal pixel scale (so the gradient arrow looks perpendicular), view selection and colour map', () => {
  const f = fitView(BASE.w, BASE.b, 506, 384);
  assert.ok(near(506 / (f.w[1] - f.w[0]), 384 / (f.b[1] - f.b[0]), 1e-9), 'one unit of w is as long as one unit of b');
  assert.ok(f.w[0] <= -1 && f.w[1] >= 5 && f.b[0] <= -1.5 && f.b[1] >= 3);
  assert.equal(chooseView(BASE, [{ w: 0, b: 0 }, { w: 4, b: 2 }]).zoomedOut, false);
  const c = (v) => heatColor(v, 100).match(/\d+/g).map(Number);
  assert.deepEqual(c(0), [20, 33, 80], 'dark is low');
  assert.ok(c(100).every((x, i) => x === [244, 162, 97][i]));
  assert.ok(c(1).reduce((s, v) => s + v) < c(50).reduce((s, v) => s + v), 'a higher loss is a lighter colour');
});

// ---- the widget against TorC's runs -------------------------------------------------------------------------------------
test('the presets are recorded plain-SGD runs, in the order the lesson binds them', () => {
  assert.deepEqual(O.runs.map((r) => [r.lr, r.momentum]).slice(0, 4), [[0.05, 0], [0.1, 0], [0.15, 0], [0.2, 0]]);
  assert.deepEqual(LR_PRESETS, [run(PLAIN).lr, run(ZIG).lr, run(BIG).lr]);
  assert.equal(STEPS, 10);
  for (const i of [PLAIN, ZIG, BIG]) assert.equal(run(i).steps.length, STEPS, '"ten steps"');
});

test('the widget reproduces the recorded runs it shows, and its arrows are the step and its mirror image', () => {
  for (const r of O.runs.filter((x) => x.momentum === 0)) {
    const sim = simulate(model, { lr: r.lr, steps: STEPS }), rows = [...sim.steps, sim.final];
    rowsOf(r).forEach((x, i) => assert.ok(near(rows[i].loss, x.loss, 1e-4) && near(rows[i].w, x.w, 1e-4) && near(rows[i].b, x.b, 1e-4), `${r.id} step ${i}`));
    rows.slice(0, -1).forEach((p, i) => {
      const { uphill, step } = arrowsAt(p, r.lr);
      assert.ok(near(step.w, rows[i + 1].w, 1e-9) && near(step.b, rows[i + 1].b, 1e-9), 'the dashed arrow ends at the next point');
      assert.ok(near(uphill.w - p.w, p.w - step.w, 1e-12) && near(uphill.b - p.b, p.b - step.b, 1e-12), 'the gradient arrow is the step turned around');
      assert.ok(model.loss(uphill.w, uphill.b) > p.loss || Math.hypot(p.gw, p.gb) < 1e-3, 'the gradient arrow points uphill');
    });
  }
});

test('map task 1: the gradient is perpendicular to the contour line, and each step is shorter than the one before (lr 0.05)', () => {
  const rows = rowsOf(run(PLAIN));
  for (const p of rows.slice(0, 4)) {
    const g = Math.hypot(p.gw, p.gb), t = [-p.gb / g, p.gw / g], h = 1e-4;   // t is perpendicular to the gradient
    const along = (model.loss(p.w + h * t[0], p.b + h * t[1]) - model.loss(p.w - h * t[0], p.b - h * t[1])) / (2 * h);
    assert.ok(Math.abs(along) < 1e-3 * g, `no change of the loss along the contour line at (${p.w}, ${p.b})`);
  }
  const len = rows.slice(1).map((q, i) => Math.hypot(q.w - rows[i].w, q.b - rows[i].b));
  for (let i = 1; i < len.length; i++) assert.ok(len[i] < len[i - 1], `step ${i + 1} is shorter than step ${i}`);
});

test('map task 2: after ten steps in the fog the trail is in the dark part of the map', () => {
  const f = run(PLAIN).final;
  assert.ok(f.loss < 0.01 * run(PLAIN).steps[0].loss);
  assert.ok(Math.hypot(f.w - O.data.optimum.w, f.b - O.data.optimum.b) < 0.1);
});

// ---- the claims of the prose ------------------------------------------------------------------------------------------
test('the three rides are the data points, and they lie exactly on y = 2x + 1 (the lowest loss is zero there)', () => {
  assert.deepEqual(O.data.points, O.data.x.map((x, i) => ({ x, t: O.data.t[i] })));
  O.data.points.forEach((p) => assert.equal(p.t, O.data.true_w * p.x + O.data.true_b));
  assert.deepEqual([O.data.optimum.w, O.data.optimum.b, O.data.optimum.loss], [O.data.true_w, O.data.true_b, 0]);
});

test('the start: w = b = 0, the loss, and the gradient that backward() gives', () => {
  const nodes = Object.fromEntries(scene.nodes.map((n) => [n.name, n]));
  assert.deepEqual([O.start.w, O.start.b], [0, 0]);
  assert.ok(near(O.start.loss, nodes.L.value.values[0]) && near(O.start.loss, O.nudge_start.L0));
  assert.ok(near(O.start.gw, nodes.W.grad.values[0]) && near(O.start.gb, nodes.b.grad.values[0]));
  assert.deepEqual(O.nudge_start.gradient, [O.start.gw, O.start.gb], 'the gradient is the list of both numbers');
  assert.ok(near(rmse(O.start.loss), 5.26, 1e-2), 'a typical miss of about 5.3 euros');
});

test('nudging one weight at a time: each ratio comes closer to the backward() gradient as the nudge shrinks', () => {
  for (const [k, g] of [['w', O.start.gw], ['b', O.start.gb]]) {
    const rows = O.nudge_start[k];
    assert.deepEqual(rows.map((r) => r.h), [1, 0.5, 0.1, 0.01, 0.001]);
    for (const r of rows) {
      assert.ok(near(r.dL, r.L_h - O.nudge_start.L0, 1e-4) && near(r.slope, r.dL / r.h, 1e-4), `${k}, h = ${r.h}`);
      assert.ok(r.dL < 0, 'the loss falls');
      assert.equal(r.backward_slope, g);
    }
    const gap = rows.map((r) => Math.abs(r.slope - g));
    for (let i = 1; i < gap.length; i++) assert.ok(gap[i] < gap[i - 1], `${k}: closer with a smaller nudge`);
  }
});

test('the predict quiz can be answered from the rounded numbers on the page', () => {
  const q = lesson.sections.find((s) => s.id === 'nudge-w');
  const r = O.nudge_start.w[2];
  assert.equal(q.answer, 'files.optim.nudge_start.w.2.slope');
  const fromPage = (Number(fmt(r.L_h)) - Number(fmt(O.nudge_start.L0))) / r.h;
  assert.ok(Math.abs(fromPage - r.slope) <= q.tolerance, `${fromPage} vs ${r.slope}`);
  const b = O.nudge_start.b[3];
  assert.ok(Math.abs((Number(fmt(b.L_h)) - Number(fmt(O.nudge_start.L0))) / b.h - O.start.gb) < 0.2, 'the b nudge on the page gives about -10');
});

test('one step for both weights: the worked w, the learner\'s b, and the loss after it', () => {
  const s1 = run(PLAIN).steps[1], lr = run(PLAIN).lr;
  assert.ok(near(s1.w, O.start.w - lr * O.start.gw) && near(s1.b, O.start.b - lr * O.start.gb));
  assert.ok(s1.w > 0 && s1.w < O.data.true_w, 'w grows toward the true price');
  const q = lesson.sections.find((s) => s.id === 'b-step');
  assert.ok(Math.abs(T.num(q.answer) - (O.start.b - lr * O.start.gb)) < 1e-6);
  assert.ok(rmse(s1.loss) < rmse(O.start.loss), 'the typical miss falls');
});

test('a learning rate four times as large: every step lands further away and the loss grows (D13: one quantity after ten steps)', () => {
  assert.ok(near(run(BIG).lr, 4 * run(PLAIN).lr, 1e-12), '"four times as large"');
  const big = rowsOf(run(BIG));
  for (let i = 1; i < big.length; i++) {
    assert.ok(big[i].loss > big[i - 1].loss, 'the loss grows at every step');
    const d = (p) => Math.hypot(p.w - O.data.optimum.w, p.b - O.data.optimum.b);
    assert.ok(d(big[i]) > d(big[i - 1]), 'each step lands further from the star');
    assert.ok(model.loss(big[i - 1].w - 1e-3 * big[i - 1].gw, big[i - 1].b - 1e-3 * big[i - 1].gb) < big[i - 1].loss, 'each step starts downhill');
  }
  assert.ok(run(BIG).final.loss > run(BIG).steps[0].loss);
  assert.deepEqual(O.after_ten_steps.with_zero_grad, { w: run(PLAIN).final.w, b: run(PLAIN).final.b, loss: run(PLAIN).final.loss });
  assert.equal(O.after_ten_steps.updates, STEPS);
  assert.equal(O.after_ten_steps.lr, run(PLAIN).lr);
});

test('the narrow valley: lr 0.15 jumps across and settles, lr 0.2 jumps higher each time, and the floor is slow', () => {
  const [lamSteep, lamFlat] = O.hessian.eigenvalues;
  assert.ok(lamSteep / lamFlat > 10, 'steep walls, an almost flat floor');
  const zig = rowsOf(run(ZIG)), big = rowsOf(run(BIG));
  for (let i = 1; i < zig.length; i++) {
    assert.ok(zig[i].loss < zig[i - 1].loss, 'lr 0.15: the loss falls at every step');
    assert.ok(along(zig[i], steepV) * along(zig[i - 1], steepV) < 0, 'lr 0.15: each step jumps to the other wall');
    assert.ok(Math.abs(along(zig[i], steepV)) < Math.abs(along(zig[i - 1], steepV)), 'lr 0.15: each jump is shorter than the last');
    assert.ok(along(big[i], steepV) * along(big[i - 1], steepV) < 0 && Math.abs(along(big[i], steepV)) > Math.abs(along(big[i - 1], steepV)), 'lr 0.2: each jump lands higher');
  }
  for (const r of [run(PLAIN), run(ZIG)]) assert.ok(Math.abs(1 - r.lr * lamFlat) > 0.95, 'along the floor the distance shrinks by less than 5 % per step');
  // rates tasks: within five steps, lr 0.15 stays in the base view and lr 0.2 makes it zoom out
  assert.equal(chooseView(BASE, zig.slice(0, 6)).zoomedOut, false);
  assert.equal(chooseView(BASE, big.slice(0, 6)).zoomedOut, true);
});

test('the cumulative quiz: four notebooks give four times the gradient at the same start and the same true price', () => {
  assert.equal(WG.steep.probes[0].grad, 4 * WG.probes[0].grad);
  assert.equal(WG.steep.probes[0].w, WG.probes[0].w);
  assert.equal(WG.steep.setup.w_min, WG.setup.w_min);
  assert.equal(D.anchors.bowl_w_min, WG.setup.w_min);
  assert.equal(WG.steep.setup.t, 2 * WG.setup.t, 'twelve euros for four notebooks, six for two');
  assert.ok(WG.steep.setup.critical_lr < WG.setup.critical_lr, 'the steeper case needs a smaller learning rate');
});

// ---- the real line ----------------------------------------------------------------------------------------------------
test('the real line is SGD::step without momentum, quoted from SOURCES', () => {
  assert.equal(STEP_LINE, 'param->data() = param->data().sub(grad.mul(lr_));');
  assert.equal(SOURCES.step.file, 'src/optim.cpp');
  assert.equal(SOURCES.step.lines[27 - SOURCES.step.from].trim(), STEP_LINE);
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  assert.ok(html.includes('param-&gt;data() = param-&gt;data().sub(grad.mul(lr_));'), 'the lesson shows the line');
});

const SRC = new URL('../build/trace/_deps/torc-src/', SITE);
test('quoted library lines equal the library source at the pinned ref (skipped when the fetched source is absent)', (t) => {
  if (!existsSync(SRC)) {
    assert.ok(!process.env.TORC_REQUIRE_SOURCE, 'TORC_REQUIRE_SOURCE is set (verify.yml), so the TorC source must be fetched (tools/trace/run.sh) first');
    t.skip('build/trace/_deps/torc-src is not present (run tools/trace/run.sh to fetch it)'); return;
  }
  for (const [key, s] of Object.entries(SOURCES)) {
    const file = readFileSync(new URL(s.file, SRC), 'utf8').split('\n');
    assert.deepEqual(s.lines, file.slice(s.from - 1, s.to), `${key}: ${s.file} lines ${s.from}-${s.to}`);
  }
});

// ---- lesson text -------------------------------------------------------------------------------------------------------
const plainHtml = () => {
  const saved = globalThis.katex;
  globalThis.katex = undefined;
  try { return renderLesson(lesson, D, { course: course() }).map((p) => p.html).join('\n'); } finally { globalThis.katex = saved; }
};

test('every bound number shows what the data gives, with the values the argument rests on', () => {
  const shown = Object.fromEntries(boundNums(plainHtml()).filter((x) => x.role !== 'human').map((x) => [x.path, x]));
  for (const b of Object.values(shown)) assert.equal(T.reformat(b), b.shown, b.path);
  assert.equal(shown['files.optim.start.loss'].shown, '27.67');
  assert.equal(shown['files.optim.start.gw'].shown, '−22.67');
  assert.equal(shown['files.optim.runs.0.steps.1.w'].shown, '1.133');
  assert.equal(shown['files.optim.runs.0.lr'].shown, '0.05');
  assert.equal(shown['files.optim.runs.3.lr'].shown, '0.2');
});

test('no hand-typed digits in the prose, and the names of decision 14: w and b, step (not update or iteration), no momentum or zero_grad', () => {
  const raw = plainHtml();
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
  const text = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
  for (const bad of [/\bupdates?\b/i, /\biterations?\b/i, /momentum/i, /zero_grad/, /\bslope\b/i, /\bknob/i, /\bW\b/, /\bbias\b/i, /curvature/i, /\bHessian\b/i]) assert.doesNotMatch(text, bad);
  assert.doesNotMatch(text, /0\.18/, 'no magic critical learning rate');
  assert.match(text, /Each colour is the loss for one pair \( w , b \)\. Dark is low\./, 'the contour map legend at first use');
});

test('the widget readout says step, w and b, and formats with 3 significant figures', () => {
  const src = readFileSync(new URL('../js/widgets/descent.js', import.meta.url), 'utf8');
  assert.doesNotMatch(src, /<b>(update|W|iteration)<\/b>/);
  assert.equal(r3(27.666666), '27.7');
  assert.equal(r3(-22.6666679), '−22.7');
  assert.equal(r3(1442.35742), '1,440');
  assert.equal(r3(0.00114498998), '0.00114');
  assert.equal(r3(NaN), 'not a number');
});

test('the loop-timeline widget (kept for the appendix) still reproduces the loop without zero_grad', () => {
  assert.equal(ITERATIONS, O.no_zero_grad.training_loop.iterations.length);
  const noZ = trainingLoop(model, { lr: O.no_zero_grad.training_loop.lr, iterations: ITERATIONS, zeroGrad: false });
  O.no_zero_grad.training_loop.iterations.forEach((x, i) => assert.ok(near(noZ[i].loss, x.loss, 1e-4) && near(noZ[i].gw, x.gw, 1e-4), `loop ${i}`));
});

test('yRange and the number format of the shared widget helpers', () => {
  assert.deepEqual(yRange([27.67, 0.001145]), [-3, 2]);
  assert.deepEqual(yRange([1, 5]), [0, 2]);
  assert.equal(sfmt(1.2345e18), '1.23e18');
  assert.equal(sfmt(27.67), '27.67');
});

test('widget modules import in Node without a DOM', async () => {
  assert.equal(typeof document, 'undefined');
  assert.equal(typeof (await import('../js/widgets/descent.js')).mountDescent, 'function');
  assert.equal(typeof (await import('../js/widgets/contour2d.js')).mountContour2d, 'function');
  assert.equal(typeof document, 'undefined');
});
