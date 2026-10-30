// Run: node --test site/test/
// The sgd-step lesson (template 3): the model against TorC's surface samples, the hand formulas for transpose/mse, the widgets' pure
// helpers, the quoted SGD::step line, and every claim the prose and the guided tasks make, each one asserted against the recorded
// data (data/optim.json, mostly its `off_axis` runs from w = 0, b = 3, and data/why-gradients.json). The v3 rules themselves
// (roles, budgets, glossary) run in lessons.test.mjs.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { buildD, makeT, anchorValues } from '../js/core/trace.js';
import { buildGraph, runBackward } from '../js/engine.js';
import { makeModel, trainingLoop } from '../js/optim.js';
import { renderLesson, boundNums } from '../js/core/sections.js';
import { fmt } from '../js/ops/texutil.js';
import lesson, { STEP_LINE } from '../js/lessons/sgd-step.js';
import { gridFor, contourSegments, niceLevels, fitView, chooseView, viewFor, heatColor } from '../js/widgets/contour2d.js';
import { LR_PRESETS, STEPS, BASE, arrowsAt, trailFor, r3 } from '../js/widgets/descent.js';
import { ITERATIONS } from '../js/widgets/loop-timeline.js';
import { yRange, fmt as sfmt } from '../js/widgets/sgd-ui.js';
import { SOURCES } from '../js/widgets/sgd-code.js';
import { readJson, SITE, course } from './helpers.mjs';

const O = readJson('data/optim.json');
const OFF = O.off_axis;
const WG = readJson('data/why-gradients.json');
const D = buildD({ optim: O, 'why-gradients': WG }, anchorValues(readJson('data/anchors.json')));
const T = makeT(D);
const scene = O.scenes[0];
const model = makeModel(scene);
const near = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
const rowsOf = (r) => [...r.steps, r.final];
const [SLOW, ZIG, BIG] = [0, 1, 2];                         // the off-axis run indices the lesson binds
const run = (i) => OFF.runs[i];
const START = [OFF.w0, OFF.b0];
const STAR = O.data.optimum;
// coordinates of a point along the steep and the flat direction of the valley, measured from the lowest point
const [steepV, flatV] = O.hessian.eigenvectors;
const along = (p, v) => (p.w - STAR.w) * v[0] + (p.b - STAR.b) * v[1];
const dist = (p) => Math.hypot(p.w - STAR.w, p.b - STAR.b);
const rmse = (loss) => Math.sqrt(loss);
const angle = (a, b) => Math.acos((a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b))) * 180 / Math.PI;

// ---- the scene and the model ---------------------------------------------------------------------------------------
test('engine: the sgd scene (transpose, matmul, add, mse) equals the real run, values and gradients', () => {
  const g = buildGraph(scene), bw = runBackward(g);
  for (const n of scene.nodes) {
    assert.ok(g.byName[n.name].value.v.every((v, i) => near(v, n.value.values[i])), `${n.name} value`);
    if (n.grad) assert.ok(bw.grad[n.name].v.every((v, i) => near(v, n.grad.values[i])), `${n.name} grad`);
  }
  assert.deepEqual(scene.nodes.filter((n) => n.role === 'op').map((n) => n.op), ['transpose', 'matmul', 'add', 'mse']);
  assert.equal(scene.parity.output_equal && scene.parity.grads_equal && scene.parity.loss_equal, true);
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

test('surface conformance: the engine loss equals the 25 loss values TorC printed, and the off-axis start', () => {
  for (const p of O.surface.points) assert.ok(near(model.loss(p.w, p.b), p.loss), `(${p.w}, ${p.b})`);
  assert.ok(near(model.loss(STAR.w, STAR.b), STAR.loss, 1e-9));
  const e = model.eval(...START);
  assert.ok(near(e.loss, OFF.start.loss) && near(e.gw, OFF.start.gw) && near(e.gb, OFF.start.gb));
});

// ---- contour helpers -------------------------------------------------------------------------------------------------
test('contour2d: marching squares puts every segment end on the requested level', () => {
  const grid = gridFor(model, BASE.w, BASE.b, 64, 56);
  const levels = niceLevels(grid.zmin, grid.zmax);
  assert.ok(levels.length >= 4 && levels.length <= 9, `levels ${levels}`);
  for (const lev of levels) {
    const segs = contourSegments(grid, lev);
    assert.ok(segs.length > 0, `level ${lev} has segments`);
    for (const [a, b] of segs) for (const [px, py] of [a, b]) assert.ok(Math.abs(model.loss(px, py) - lev) <= 0.04 * lev + 0.02, `level ${lev}`);
  }
  assert.deepEqual(niceLevels(0, 5), [1, 2]);
});

test('contour2d: equal pixel scale (so the gradient arrow looks perpendicular), view selection and colour map', () => {
  const f = fitView(BASE.w, BASE.b, 506, 384);
  assert.ok(near(506 / (f.w[1] - f.w[0]), 384 / (f.b[1] - f.b[0]), 1e-9), 'one unit of w is as long as one unit of b');
  const c = (v) => heatColor(v, 100).match(/\d+/g).map(Number);
  assert.deepEqual(c(0), [20, 33, 80], 'dark is low');
  assert.ok(c(1).reduce((s, v) => s + v) < c(50).reduce((s, v) => s + v), 'a higher loss is a lighter colour');
});

// ---- the widget against TorC's runs -------------------------------------------------------------------------------------
test('the presets are the recorded off-axis plain-SGD runs, in the order the lesson binds them', () => {
  assert.deepEqual(START, [0, 3]);
  assert.deepEqual(OFF.runs.map((r) => [r.lr, r.momentum]), [[0.05, 0], [0.15, 0], [0.2, 0]]);
  assert.deepEqual(LR_PRESETS, OFF.runs.map((r) => r.lr));
  assert.equal(STEPS, 10);
  for (const r of OFF.runs) assert.equal(r.steps.length, STEPS, '"ten steps"');
});

test('the widget reproduces the recorded runs it shows, and its arrows are the step and its mirror image', () => {
  for (const r of OFF.runs) {
    const rows = trailFor(model, r.lr, START);
    rowsOf(r).forEach((x, i) => assert.ok(near(rows[i].loss, x.loss, 1e-4) && near(rows[i].w, x.w, 1e-4) && near(rows[i].b, x.b, 1e-4), `${r.id} step ${i}`));
    rows.slice(0, -1).forEach((p, i) => {
      const { uphill, step } = arrowsAt(p, r.lr);
      assert.ok(near(step.w, rows[i + 1].w, 1e-9) && near(step.b, rows[i + 1].b, 1e-9), 'the dashed arrow ends at the next point');
      assert.ok(near(uphill.w - p.w, p.w - step.w, 1e-12) && near(uphill.b - p.b, p.b - step.b, 1e-12), 'the gradient arrow is the step turned around');
    });
  }
});

test('M1: the gradient arrow tip stays inside the view at every preset and step (its head is never clipped)', () => {
  for (const lr of LR_PRESETS) {
    const rows = trailFor(model, lr, START);
    for (let k = 0; k <= STEPS; k++) {
      const { uphill } = arrowsAt(rows[k], lr);
      const v = viewFor(BASE, { trail: rows.slice(0, k + 1), next: k < STEPS ? rows[k + 1] : null, uphill, optimum: STAR });
      assert.ok(uphill.w >= v.w[0] && uphill.w <= v.w[1] && uphill.b >= v.b[0] && uphill.b <= v.b[1], `lr ${lr}, step ${k}`);
      if (lr !== 0.2) assert.equal(v.zoomedOut, false, `lr ${lr} stays in the base view (step ${k})`);
    }
  }
});

test('map task 1: the arrows do not point at the star, the step is perpendicular to the contour line, the trail bends, both arrows shrink', () => {
  const rows = rowsOf(run(SLOW));
  const toStar = [STAR.w - START[0], STAR.b - START[1]], stepDir = [-OFF.start.gw, -OFF.start.gb];
  assert.ok(angle(toStar, stepDir) > 45, `the first step misses the star by ${angle(toStar, stepDir).toFixed(0)} degrees`);
  for (const p of rows.slice(0, 4)) {
    const g = Math.hypot(p.gw, p.gb), t = [-p.gb / g, p.gw / g], h = 1e-4;   // t is perpendicular to the gradient
    const d = (model.loss(p.w + h * t[0], p.b + h * t[1]) - model.loss(p.w - h * t[0], p.b - h * t[1])) / (2 * h);
    assert.ok(Math.abs(d) < 1e-3 * g, 'no change of the loss along the contour line');
  }
  const dir = rows.slice(1).map((q, i) => [q.w - rows[i].w, q.b - rows[i].b]);
  assert.ok(angle(dir[0], dir[4]) > 30, 'the trail bends within five steps');
  const len = dir.map((d) => Math.hypot(...d));
  for (let i = 1; i < len.length; i++) assert.ok(len[i] < len[i - 1], 'each step (and its mirror arrow) is shorter than the last');
});

test('map task 2: after ten steps the trail is on the valley floor but still far from the star', () => {
  const f = run(SLOW).final;
  for (let i = 1; i < rowsOf(run(SLOW)).length; i++) assert.ok(rowsOf(run(SLOW))[i].loss < rowsOf(run(SLOW))[i - 1].loss, 'every step went downhill');
  assert.ok(Math.abs(along(f, steepV)) < 0.1 * Math.abs(along(f, flatV)), 'on the floor: little left across the valley');
  assert.ok(dist(f) > 1, 'still far from the star');
});

// ---- the claims of the prose ------------------------------------------------------------------------------------------
test('the three rides lie on y = 2x + 1: each extra kilometre adds 2 euros, the fee is 1 euro', () => {
  assert.deepEqual(O.data.points, O.data.x.map((x, i) => ({ x, t: O.data.t[i] })));
  O.data.points.forEach((p) => assert.equal(p.t, O.data.true_w * p.x + O.data.true_b));
  assert.equal(O.data.points[1].t - O.data.points[0].t, O.data.true_w);
  assert.deepEqual([STAR.w, STAR.b, STAR.loss], [O.data.true_w, O.data.true_b, 0]);
  assert.ok(near(rmse(OFF.start.loss), 2.58, 1e-2), 'a typical miss of about 2.6 euros at the start');
});

test('nudging one weight at a time: the same nudge for both, and each ratio comes closer to the backward() gradient', () => {
  assert.equal(OFF.nudge.w[2].h, OFF.nudge.b[2].h, 'the page uses the same nudge for w and b');
  assert.ok(near(OFF.nudge.L0, OFF.start.loss));
  for (const [k, g] of [['w', OFF.start.gw], ['b', OFF.start.gb]]) {
    const rows = OFF.nudge[k];
    assert.deepEqual(rows.map((r) => r.h), [1, 0.5, 0.1, 0.01, 0.001]);
    for (const r of rows) {
      assert.ok(near(r.dL, r.L_h - OFF.nudge.L0, 1e-4) && near(r.slope, r.dL / r.h, 1e-4), `${k}, h = ${r.h}`);
      assert.ok(r.dL < 0, 'the loss falls');
      assert.equal(r.backward_slope, g);
    }
    const gap = rows.map((r) => Math.abs(r.slope - g));
    for (let i = 1; i < gap.length; i++) assert.ok(gap[i] < gap[i - 1], `${k}: closer with a smaller nudge`);
  }
});

test('the worked b nudge works from the rounded numbers on the page, and the w ratio is the same nudge', () => {
  const r = OFF.nudge.b[2];
  const fromPage = (Number(fmt(r.L_h)) - Number(fmt(OFF.nudge.L0))) / r.h;
  assert.ok(Math.abs(fromPage - r.slope) <= 0.05, `${fromPage} vs ${r.slope}`);
  assert.ok(Math.abs(Number(fmt(r.dL)) / r.h - r.slope) < 0.01, 'dL ÷ h as shown gives the ratio shown');
  assert.equal(OFF.nudge.w[2].h, r.h);
});

test('one step for both weights: b goes up, away from the best fee, so the step does not go toward the star', () => {
  const s0 = run(SLOW).steps[0], s1 = run(SLOW).steps[1], lr = run(SLOW).lr;
  assert.ok(near(s1.w, s0.w - lr * s0.gw) && near(s1.b, s0.b - lr * s0.gb));
  assert.ok(s1.b > s0.b && s0.b > STAR.b, 'b moves away from the best fee');
  const q = lesson.sections.find((s) => s.id === 'second-step'), s2 = run(SLOW).steps[2];
  assert.ok(Math.abs(T.num(q.answer) - (s1.w - lr * s1.gw)) < 1e-5, 'the second step for w');
  assert.ok(Math.abs(s1.gw) < Math.abs(s0.gw) && Math.abs(s2.w - s1.w) < Math.abs(s1.w - s0.w), 'a smaller gradient, a shorter step');
});

test('lr 0.2: four times as large; w lands near 2 but b moves away; the jumps and the loss grow at every step', () => {
  assert.ok(near(run(BIG).lr, 4 * run(SLOW).lr, 1e-12));
  const s1 = run(BIG).steps[1];
  assert.ok(Math.abs(s1.w - STAR.w) < 0.2 && s1.b > OFF.b0, 'w near the best value, b moved away');
  const big = rowsOf(run(BIG));
  for (let i = 1; i < big.length; i++) {
    assert.ok(big[i].loss > big[i - 1].loss, 'the loss grows at every step');
    const p = big[i - 1];
    assert.ok(model.loss(p.w - 1e-3 * p.gw, p.b - 1e-3 * p.gb) < p.loss, 'each step starts downhill');
  }
  assert.ok(rmse(run(BIG).final.loss) > 10 * rmse(run(SLOW).final.loss));
  assert.ok(dist(big.at(-1)) > 2 * dist(big[0]), 'the trail ends far from the star');
});

test('§5 and the rates tasks: lr 0.15 zig-zags across and creeps along the floor; lr 0.2 jumps grow; the steep direction sets the limit', () => {
  const [lamSteep, lamFlat] = O.hessian.eigenvalues;
  assert.ok(lamSteep / lamFlat > 10, 'steep walls, an almost flat floor');
  assert.ok(Math.abs(along(OFF.start, flatV)) > Math.abs(along(OFF.start, steepV)), 'the start is off the steep axis');
  const zig = rowsOf(run(ZIG)), big = rowsOf(run(BIG));
  for (let i = 1; i < zig.length; i++) {
    assert.ok(along(zig[i], steepV) * along(zig[i - 1], steepV) < 0, 'lr 0.15: each step jumps to the other wall');
    assert.ok(Math.abs(along(zig[i], steepV)) < Math.abs(along(zig[i - 1], steepV)), 'lr 0.15: the jumps shrink');
    assert.ok(Math.abs(along(zig[i], flatV)) < Math.abs(along(zig[i - 1], flatV)), 'lr 0.15: it creeps along the floor toward the star');
    assert.ok(along(big[i], steepV) * along(big[i - 1], steepV) < 0 && Math.abs(along(big[i], steepV)) > Math.abs(along(big[i - 1], steepV)), 'lr 0.2: the jumps grow');
    assert.ok(Math.abs(along(big[i], flatV)) < Math.abs(along(big[i - 1], flatV)), 'lr 0.2 is still safe along the floor');
  }
  assert.ok(Math.abs(1 - 0.2 * lamSteep) > 1 && Math.abs(1 - 0.2 * lamFlat) < 1, 'the steep direction is the one that breaks at lr 0.2');
  assert.ok(Math.abs(along(zig.at(-1), flatV)) > 0.5 * Math.abs(along(zig[0], flatV)), 'the steps along the floor are slow');
});

test('the cumulative quiz: four notebooks give four times the gradient at the same start and the same true price', () => {
  assert.equal(WG.steep.probes[0].grad, 4 * WG.probes[0].grad);
  assert.equal(WG.steep.probes[0].w, WG.probes[0].w);
  assert.equal(WG.steep.setup.w_min, WG.setup.w_min);
  assert.equal(D.anchors.bowl_w_min, WG.setup.w_min);
  assert.equal(WG.steep.setup.t, 2 * WG.setup.t);
  assert.ok(WG.steep.setup.critical_lr < WG.setup.critical_lr, 'the steeper case needs a smaller learning rate');
});

// ---- the real line ----------------------------------------------------------------------------------------------------
test('the real line is SGD::step without momentum, quoted from SOURCES', () => {
  assert.equal(STEP_LINE, 'param->data() = param->data().sub(grad.mul(lr_));');
  assert.equal(SOURCES.step.file, 'src/optim.cpp');
  assert.equal(SOURCES.step.lines[27 - SOURCES.step.from].trim(), STEP_LINE);
  assert.ok(SOURCES.step.lines.some((l) => l.includes('for (size_t i = 0; i < params_.size(); ++i)')), 'it loops over the parameter groups');
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
  const nums = boundNums(plainHtml());
  for (const b of nums) assert.equal(T.reformat(b), b.shown, b.path);
  const shown = (path, role) => nums.find((b) => b.path === path && (!role || b.role === role))?.shown;
  assert.equal(shown('files.optim.off_axis.start.gw'), '−10.67');
  assert.equal(shown('files.optim.off_axis.runs.0.steps.1.w', 'worked'), '0.5333');
  assert.equal(shown('files.optim.off_axis.runs.0.steps.1.b', 'worked'), '3.2');
  assert.equal(shown('files.optim.off_axis.runs.2.lr'), '0.2');
  assert.equal(shown('files.optim.off_axis.start.loss', 'human'), '2.6');
  assert.ok(!nums.some((b) => /loss$/.test(b.path) && !['human', 'worked', 'reveal'].includes(b.role) && !/nudge/.test(b.path)), 'no bare loss value (M5)');
});

test('no hand-typed digits in the prose, and one name for each thing (decision 14)', () => {
  const raw = plainHtml();
  const left = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<details class="refresher"[\s\S]*?<\/details>/g, '').replace(/<span class="num"[^>]*>[^<]*<\/span>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, '')
    .replace(/<svg[\s\S]*?<\/svg>/g, '').replace(/<h2[^>]*>\s*\d\s·/g, '<h2>').replace(/<[^>]+>/g, ' ').match(/[^\s]*\d[^\s]*/g) ?? [];
  assert.deepEqual(left, []);
  const text = raw.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<details class="refresher"[\s\S]*?<\/details>/g, '').replace(/<pre[\s\S]*?<\/pre>/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
  for (const bad of [/\bupdates?\b/i, /\biterations?\b/i, /momentum/i, /zero_grad/, /\bslope\b/i, /\bknob/i, /\bW\b/, /\bbias\b/i, /curvature/i, /\bHessian\b/i,
    /\bminimum\b/i, /dark part|low part/i, /your point/i, /loss\.backward/, /from above/i, /once for each weight/i, /It only uses it/, /alone showed/]) assert.doesNotMatch(text, bad);
  assert.doesNotMatch(text, /0\.18/, 'no magic critical learning rate');
  assert.match(text, /Each colour is the loss for one pair \( w , b \)\. Dark is low\./, 'the contour map legend at first use');
  assert.match(text, /The arrows do not point at the star\./, 'the task states the refutation (H1)');
  assert.match(text, /L\.backward\(\)|L.backward/);
});

test('the widget readout says step, w and b, the typical miss, and formats with 3 significant figures', () => {
  const src = readFileSync(new URL('../js/widgets/descent.js', import.meta.url), 'utf8');
  assert.doesNotMatch(src, /<b>(update|W|iteration)<\/b>/);
  assert.match(src, /<b>typical miss<\/b>/);
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

test('yRange, chooseView and the number format of the shared widget helpers', () => {
  assert.deepEqual(yRange([27.67, 0.001145]), [-3, 2]);
  assert.equal(sfmt(1.2345e18), '1.23e18');
  assert.equal(chooseView(BASE, [{ w: 0, b: 3 }, { w: 2, b: 1 }]).zoomedOut, false);
});

test('widget modules import in Node without a DOM', async () => {
  assert.equal(typeof document, 'undefined');
  assert.equal(typeof (await import('../js/widgets/descent.js')).mountDescent, 'function');
  assert.equal(typeof (await import('../js/widgets/contour2d.js')).mountContour2d, 'function');
  assert.equal(typeof document, 'undefined');
});
