// Run: node --test site/test/
// The sgd-step lesson: the model against TorC's surface samples, hand formulas for transpose/mse, the widgets' pure helpers, quoted source
// lines, the claims the prose makes (each one asserted against the recorded runs), and "modules import in Node without a DOM".
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { buildGraph, runBackward } from '../js/engine.js';
import { makeModel, simulate, trainingLoop, sgdStep, initialState } from '../js/optim.js';
import { buildD, makeT } from '../js/core/trace.js';
import { renderLesson, findUnboundNums, boundNums } from '../js/core/sections.js';
import lesson from '../js/lessons/sgd-step.js';
import { makeLayouts, CONCEPTS } from '../js/lessons/sgd-step.scenes.js';
import { gridFor, contourSegments, niceLevels, fitView, chooseView, heatColor } from '../js/widgets/contour2d.js';
import { describeRun, yRange, LR_PRESETS, MU_PRESETS, STEPS } from '../js/widgets/lr-lab.js';
import { updateWindowData } from '../js/widgets/sgd-update-window.js';
import { lifecycleReduce, START } from '../js/widgets/sgd-lifecycle.js';
import { ledger, narrate, DEFAULT_ITERATION } from '../js/widgets/loop-timeline.js';
import { SOURCES } from '../js/widgets/sgd-code.js';
import { readJson, renderStrictAll, SITE } from './helpers.mjs';

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

// ---- the claims on the lr and momentum controls ------------------------------------------------------------------------
test('presets: each tag is true of the recorded run it names', () => {
  const opt = O.data.optimum;
  const d = Object.fromEntries(O.runs.map((r) => [r.id, describeRun(rowsOf(r), opt)]));
  assert.deepEqual(LR_PRESETS.map((p) => p.lr), [0.05, 0.1, 0.15, 0.2]);
  assert.deepEqual(MU_PRESETS.map((p) => p.mu), [0, 0.5, 0.9]);
  // 0.05 steady: never overshoots, the loss falls at every update
  assert.deepEqual(d['sgd-lr0.05'].rose, []); assert.equal(d['sgd-lr0.05'].swings, 0); assert.ok(d['sgd-lr0.05'].reach < opt.w + 0.1);
  // 0.1 overshoots once: w jumps past 2 at the first update and the loss is already below the lr 0.05 value
  assert.equal(d['sgd-lr0.1'].swings, 1);
  assert.ok(run('sgd-lr0.1').steps[1].w > opt.w + 0.2 && run('sgd-lr0.1').steps[1].loss < run('sgd-lr0.05').steps[1].loss);
  // 0.15 zig-zags but the loss falls at every update
  assert.ok(d['sgd-lr0.15'].swings >= 3); assert.deepEqual(d['sgd-lr0.15'].rose, []);
  const swing = (r) => rowsOf(r).slice(1).map((x, i) => Math.abs(x.w - rowsOf(r)[i].w));   // |change of w| per update
  const s15 = swing(run('sgd-lr0.15'));
  for (let i = 1; i < s15.length; i++) assert.ok(s15[i] < s15[i - 1], 'each swing is smaller');
  // 0.2 diverges, each swing bigger
  assert.equal(d['sgd-lr0.2'].diverging, true);
  const s20 = swing(run('sgd-lr0.2'));
  for (let i = 1; i < s20.length; i++) assert.ok(s20[i] > s20[i - 1], 'each swing is bigger');
  for (let i = 1; i < 10; i++) assert.ok(run('sgd-lr0.2').steps[i].loss > run('sgd-lr0.2').steps[i - 1].loss);
  // momentum: the loss does not fall at every update; the first update is the plain one; mu 0.9 has not settled
  assert.ok(d['sgd-lr0.05-mu0.5'].rose.includes(3));
  assert.ok(run('sgd-lr0.05-mu0.5').steps[3].loss > run('sgd-lr0.05-mu0.5').steps[2].loss);
  assert.ok(run('sgd-lr0.05-mu0.5').steps[2].loss < run('sgd-lr0.05').steps[2].loss, 'momentum is ahead after two updates');
  assert.ok(d['sgd-lr0.05-mu0.9'].rose.length >= 3 && run('sgd-lr0.05-mu0.9').final.loss > 0.5);
  for (const id of ['sgd-lr0.05-mu0.5', 'sgd-lr0.05-mu0.9']) for (const k of ['w', 'b', 'loss']) assert.equal(run(id).steps[1][k], run('sgd-lr0.05').steps[1][k], `${id} first update equals the plain one`);
  assert.equal(STEPS, 10);
});

test('the lesson says the loss "falls at every update" and the numbers are near the critical rate it quotes', () => {
  assert.ok(Math.abs(O.hessian.critical_lr - 0.18) < 0.001);
  assert.ok(run('sgd-lr0.05').final.loss > 0 && run('sgd-lr0.05').final.loss < 0.002, 'ten updates are not enough for loss 0');
  const r = rowsOf(run('sgd-lr0.05')).map((x) => x.loss);
  for (let i = 1; i < r.length; i++) assert.ok(r[i] < r[i - 1]);
});

test('yRange covers all losses with at least two decades', () => {
  assert.deepEqual(yRange([27.67, 0.001145]), [-3, 2]);
  assert.deepEqual(yRange([1, 5]), [0, 2]);
  assert.deepEqual(yRange([27.67, 1442]), [1, 4]);
});

// ---- the update window --------------------------------------------------------------------------------------------------
test('updateWindow: rule and numbers parse in KaTeX, "after" equals TorC\'s next state for four updates and for momentum', () => {
  for (const [id, count] of [['sgd-lr0.05', 4], ['sgd-lr0.05-mu0.5', 3]]) {
    const r = run(id), rows = rowsOf(r);
    let st = initialState([[0], [0]]);
    for (let u = 0; u < count; u++) {
      const a = rows[u];
      const out = sgdStep(st, [[a.w], [a.b]], [[a.gw], [a.gb]], { lr: r.lr, momentum: r.momentum });
      assert.ok(near(out.params[0][0], rows[u + 1].w) && near(out.params[1][0], rows[u + 1].b), `${id} update ${u + 1}`);
      const data = updateWindowData({ lr: r.lr, momentum: r.momentum, params: [
        { name: 'W', before: a.w, grad: a.gw, vBefore: st.velocities[0][0], vAfter: out.state.velocities[0][0], after: out.params[0][0] },
        { name: 'b', before: a.b, grad: a.gb, vBefore: st.velocities[1][0], vAfter: out.state.velocities[1][0], after: out.params[1][0] }] });
      renderStrictAll(data.blocks.flatMap((b) => b.latex ?? []));
      assert.equal(data.table.rows.length, 2);
      assert.equal(data.table.columns.length, data.table.rows[0].length);
      st = out.state;
      if (r.momentum === 0) {
        assert.ok(near(data.table.rows[0][3], r.lr * a.gw), 'step = lr x grad');
        assert.ok(near(data.table.rows[0][1] - data.table.rows[0][3], data.table.rows[0][4], 1e-12), 'before - step = after');
      }
    }
  }
});

// ---- the sandbox --------------------------------------------------------------------------------------------------------
test('lifecycle: the sandbox follows TorC (silent skip, copy then add, step reads .grad and leaves it, zero_grad clears)', () => {
  const g0 = model.eval(0, 0);
  const step = (s) => lifecycleReduce(s, 'step', model, 0.05);
  const back = (s) => lifecycleReduce(s, 'backward', model, 0.05);
  const zero = (s) => lifecycleReduce(s, 'zero_grad', model, 0.05);
  let r = step(START);
  assert.deepEqual([r.state.w, r.state.b], [0, 0]); assert.deepEqual(r.highlight, [20]);
  const skipped = Object.fromEntries(O.skipped_step.map((q) => [q.name, q]));
  assert.equal(skipped.step_without_backward.changed, false); assert.equal(skipped.step_without_backward.threw, false);
  r = back(START); assert.ok(near(r.state.gw, g0.gw) && near(r.state.gb, g0.gb));
  const once = r.state;
  r = back(once); assert.ok(near(r.state.gw, 2 * g0.gw)); r = back(r.state); assert.ok(near(r.state.gw, 3 * g0.gw));
  O.no_zero_grad.accumulate_only.forEach((row, i) => assert.ok(near([once.gw, 2 * once.gw, 3 * once.gw][i], row.gw)));
  r = step(once); assert.ok(near(r.state.w, O.runs[0].steps[1].w) && near(r.state.b, O.runs[0].steps[1].b));
  assert.equal(r.state.gw, once.gw, 'step() does not clear .grad');
  assert.deepEqual([r.state.gw, r.state.gb].map((v) => v == null), [false, false]);
  r = zero(r.state); assert.equal(r.state.gw, null);
  r = step(r.state); assert.ok(near(r.state.w, O.runs[0].steps[1].w), 'step() after zero_grad changes nothing');
  assert.equal(skipped.step_after_zero_grad.changed, false); assert.equal(skipped.step_after_backward.changed, true);
  assert.ok(near(skipped.step_after_backward.after.w, O.runs[0].steps[1].w));
  assert.deepEqual(START.calls, [], 'the shared start state is never modified');
  assert.throws(() => lifecycleReduce(START, 'nope', model));
});

// ---- the loop timeline --------------------------------------------------------------------------------------------------------
test('loop timeline: ledger and narration agree with the recorded loop (what the prose says about iteration 3)', () => {
  const rec = O.no_zero_grad.training_loop.iterations;
  const noZ = trainingLoop(model, { lr: 0.05, iterations: 11, zeroGrad: false });
  const withZ = trainingLoop(model, { lr: 0.05, iterations: 11, zeroGrad: true });
  assert.equal(DEFAULT_ITERATION, 3);
  const k = DEFAULT_ITERATION;
  const L = ledger(noZ, k, false, 0.05);
  assert.ok(near(L.leftover, rec[1].gw) && near(L.thisPass, rec[2].gw_this_pass) && near(L.read, rec[2].gw));
  assert.ok(near(L.leftover + L.thisPass, L.read, 1e-9), 'leftover + this pass = what step() reads');
  assert.ok(near(L.wBefore, rec[2].w) && near(L.wAfter, rec[3].w));
  assert.ok(L.wBefore > O.data.optimum.w && L.thisPass > 0 && L.leftover < 0 && L.read < 0 && L.wAfter > L.wBefore, 'W is past the best value, the pass says "down", the sum says "up"');
  assert.match(narrate(L, false, k), /decrease/);
  assert.ok(rec[3].loss > rec[2].loss, 'the loss goes up');
  const Z = ledger(withZ, 3, true, 0.05);
  assert.equal(Z.leftover, null); assert.ok(near(Z.read, Z.thisPass, 1e-12));
  assert.equal(Z.cells.afterZeroGrad.grad, null); assert.equal(L.cells.afterZeroGrad.grad, L.read, 'without the call, .grad is kept');
  assert.equal(ledger(noZ, 1, false, 0.05).leftover, null);
  assert.doesNotMatch(narrate(Z, true, 3), /decrease|increase/);
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

test('every inline quote in the lesson text is a line of SOURCES, and the highlighted line numbers in the sandbox point at the intended lines', () => {
  const T2 = makeT(D);
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  const quotes = [...html.matchAll(/<code class="quote" data-src="(\w+):(\d+)">([^<]*)<\/code>/g)];
  assert.ok(quotes.length >= 6);
  for (const [, key, n, text] of quotes) {
    const s = SOURCES[key];
    assert.equal(text.replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/&amp;/g, '&'), s.lines[n - s.from].trim());
  }
  const line = (key, n) => SOURCES[key].lines[n - SOURCES[key].from].trim();
  assert.equal(line('step', 20), 'if (!param->has_grad()) continue;');
  assert.equal(line('step', 27), 'param->data() = param->data().sub(grad.mul(lr_));');
  assert.match(line('step', 24), /velocities_\[i\]\.mul\(momentum_\)\.add\(grad\)/);
  assert.match(line('step', 25), /sub\(velocities_\[i\]\.mul\(lr_\)\)/);
  assert.equal(line('accumulate', 83), 'grad_.data()[i] += local_grad.data()[i];');
  assert.equal(line('accumulate', 80), 'has_grad_ = true;');
  assert.equal(line('zero', 34), 'param->zero_grad();');
  assert.equal(line('varzero', 111), 'has_grad_ = false;');
  assert.deepEqual([66, 67, 68].map((n) => line('loop', n)), ['loss.backward();', 'optimizer.step();', 'optimizer.zero_grad();']);
  void T2;
});

// ---- lesson text ----------------------------------------------------------------------------------------------------------------
test('the graph lab text and the concept cards bind every number to the trace', () => {
  const L = makeLayouts(T).sgd;
  const texts = [L.goal, L.intro, L.forwardHint, L.backwardHint, L.takeaway, ...CONCEPTS(T).flatMap((c) => [c.h, c.p])];
  for (const html of texts) {
    assert.deepEqual(findUnboundNums(html), []);
    for (const { path, shown } of boundNums(html)) assert.equal(T.text(path), shown);
  }
  assert.ok(boundNums(L.takeaway).length >= 4);
  // every node the layout positions exists in the scene, and every scene node has a position
  assert.deepEqual(Object.keys(L.pos).sort(), scene.nodes.map((n) => n.name).sort());
  assert.deepEqual(L.above.every((n) => L.pos[n]), true);
});

test('the lesson text has every storyboard part, predict and check sections, and a gotcha', () => {
  const kinds = lesson.sections.map((s) => s.kind);
  for (const k of ['predict', 'check', 'gotcha', 'widget']) assert.ok(kinds.includes(k), k);
  const widgetIds = lesson.sections.filter((s) => s.kind === 'widget').map((s) => s.id);
  assert.deepEqual(widgetIds, ['lab', 'lifecycle', 'update-window', 'landscape', 'lr-lab', 'loop', 'momentum']);
  const html = renderLesson(lesson, D).map((p) => p.html).join('\n');
  for (const id of ['setup', 'graph', 'who', 'paper', 'land', 'lr', 'zg', 'mom', 'ideas']) assert.match(html, new RegExp(`id="${id}"`), `section ${id}`);
  assert.match(html, /never computes a gradient/);
  assert.match(html, /zero_grad\(\)<\/code> is not optional/);
  assert.match(html, /parameters are updated, inputs are not/);
  assert.doesNotMatch(html, new RegExp(['Claude', 'Session'].join('-') + '|claude\\.ai/code'), 'no session trailer or URL on a lesson page');
  assert.ok(boundNums(html).length > 80);
});

// ---- Node import hygiene ----------------------------------------------------------------------------------------------------------
test('widget modules import in Node without a DOM and expose their mount functions', async () => {
  assert.equal(typeof document, 'undefined');
  const names = { 'contour2d.js': 'mountContour2d', 'lr-lab.js': 'mountDescent', 'loop-timeline.js': 'mountLoopTimeline', 'sgd-lifecycle.js': 'mountLifecycle', 'sgd-update-window.js': 'mountUpdateWindow', 'sgd-code.js': 'codePanel', 'sgd-ui.js': 'sliderField' };
  for (const [file, fn] of Object.entries(names)) {
    const m = await import(new URL(`../js/widgets/${file}`, import.meta.url));
    assert.equal(typeof m[fn], 'function', `${file} exports ${fn}`);
  }
  assert.equal(typeof document, 'undefined');
});
