// Run: node --test site/test/
// optim.js (the JS port of optim::SGD) against every run TorC recorded in data/optim.json, plus the derived mathematics.
import test from 'node:test';
import assert from 'node:assert/strict';
import { sgdStep, initialState, validateSGD, makeModel, simulate, trainingLoop, accumulatePasses, hessian, eigenvalues2, analyze } from '../js/optim.js';
import { readJson } from './helpers.mjs';

const O = readJson('data/optim.json');
const model = makeModel(O.scenes[0]);
// float32 (TorC) vs float64 (this port): mixed tolerance measured in docs/lessons/sgd-step.md (max 2.3e-6)
const TOL = 1e-5;
const near = (a, b, tol = TOL) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
const KEYS = ['w', 'b', 'loss', 'gw', 'gb'];

test('sgdStep: plain rule p = p - lr*grad, inputs are not modified', () => {
  const params = [[1, 2], [3]], grads = [[0.5, -1], [2]];
  const out = sgdStep(null, params, grads, { lr: 0.1 });
  assert.deepEqual(out.params, [[1 - 0.05, 2 + 0.1], [3 - 0.2]]);
  assert.deepEqual(params, [[1, 2], [3]]);
  assert.deepEqual(grads, [[0.5, -1], [2]]);
  assert.deepEqual(out.applied, [true, true]);
  assert.deepEqual(out.state.velocities, [[0, 0], [0]], 'the velocity is untouched without momentum');
});

test('sgdStep: momentum v = mu*v + g, p = p - lr*v; the first step equals the plain step', () => {
  const lr = 0.05, mu = 0.9;
  let st = initialState([[1]]);
  let r = sgdStep(st, [[1]], [[2]], { lr, momentum: mu });
  assert.equal(r.state.velocities[0][0], 2);                       // 0.9*0 + 2
  assert.equal(r.params[0][0], 1 - lr * 2);                        // identical to the plain step
  assert.equal(r.params[0][0], sgdStep(null, [[1]], [[2]], { lr }).params[0][0]);
  r = sgdStep(r.state, r.params, [[-1]], { lr, momentum: mu });
  assert.ok(near(r.state.velocities[0][0], 0.9 * 2 - 1, 1e-12));   // 0.8
  assert.ok(near(r.params[0][0], (1 - lr * 2) - lr * 0.8, 1e-12));
});

test('sgdStep: a parameter without a gradient is skipped silently (has_grad() false), the others are updated', () => {
  const out = sgdStep(null, [[1], [2]], [null, [1]], { lr: 0.5, momentum: 0.9 });
  assert.deepEqual(out.applied, [false, true]);
  assert.deepEqual(out.params, [[1], [1.5]]);
  assert.deepEqual(out.state.velocities, [[0], [1]]);
  // the recorded cases: step() with nothing to read leaves (w, b) alone, also with momentum, also repeated
  let st = initialState([[0], [0]]), p = [[0], [0]];
  for (let i = 0; i < 2; i++) ({ params: p, state: st } = sgdStep(st, p, [undefined, undefined], { lr: 0.05, momentum: 0.9 }));
  assert.deepEqual(p, [[0], [0]]);
  assert.deepEqual(st.velocities, [[0], [0]]);
});

test('validateSGD: same rules and message text as the SGD constructor (data/optim.json ctor_errors)', () => {
  const msg = (f) => { try { f(); return null; } catch (e) { return { name: e.name, what: e.message }; } };
  const byCall = Object.fromEntries(O.ctor_errors.map((e) => [e.call, e]));
  for (const [lr, mu, call] of [[0, 0, 'SGD(params, lr = 0)'], [-0.1, 0, 'SGD(params, lr = -0.1)'], [0.1, -0.5, 'SGD(params, lr = 0.1, momentum = -0.5)']]) {
    const m = msg(() => validateSGD(lr, mu));
    assert.equal(m.what, byCall[call].what, call);
    assert.equal(m.name, byCall[call].exception);
    assert.throws(() => sgdStep(null, [[0]], [[1]], { lr, momentum: mu }));
  }
  assert.equal(msg(() => validateSGD(0.01, 0)), null);
});

test('simulate reproduces every row of every recorded run (steps and final) within 1e-5 (1 + |b|)', () => {
  assert.equal(O.runs.length, 6);
  for (const r of O.runs) {
    assert.equal(r.steps.length, 10, `${r.id} has ten recorded updates (the lesson text says "ten")`);
    const s = simulate(model, { lr: r.lr, momentum: r.momentum, steps: 10 });
    const mine = [...s.steps, s.final], rec = [...r.steps, r.final];
    rec.forEach((row, i) => { for (const k of KEYS) assert.ok(near(mine[i][k], row[k]), `${r.id} row ${i} ${k}: ${mine[i][k]} vs ${row[k]}`); });
    assert.equal(s.steps[0].w, 0); assert.equal(s.steps[0].b, 0);
  }
});

test('trainingLoop: with zero_grad it is the lr 0.05 run; without it, the recorded no_zero_grad loop', () => {
  const lr = O.no_zero_grad.training_loop.lr;
  const withZ = trainingLoop(model, { lr, iterations: 10, zeroGrad: true });
  O.runs.find((r) => r.id === 'sgd-lr0.05').steps.forEach((row, i) => { for (const k of KEYS) assert.ok(near(withZ[i][k], row[k]), `with zero_grad, iteration ${i + 1} ${k}`); });
  const noZ = trainingLoop(model, { lr, iterations: 10, zeroGrad: false });
  O.no_zero_grad.training_loop.iterations.forEach((row, i) => {
    assert.equal(noZ[i].iteration, row.iteration);
    for (const k of [...KEYS, 'gw_this_pass', 'gb_this_pass']) assert.ok(near(noZ[i][k], row[k], 2e-5), `without zero_grad, iteration ${i + 1} ${k}: ${noZ[i][k]} vs ${row[k]}`);
  });
});

test('accumulate: three passes at the same parameters give k times the gradient (recorded: -22.67, -45.33, -68.00)', () => {
  const acc = accumulatePasses(model, { passes: 3 });
  O.no_zero_grad.accumulate_only.forEach((row, i) => { for (const k of ['gw', 'gb', 'loss', 'w', 'b']) assert.ok(near(acc[i][k], row[k]), `pass ${i + 1} ${k}`); });
  const g = model.eval(0, 0);
  acc.forEach((row, i) => { assert.ok(near(row.gw, (i + 1) * g.gw, 1e-12)); assert.ok(near(row.gb, (i + 1) * g.gb, 1e-12)); });
});

test('a loop that never calls zero_grad() is exactly SGD with momentum 1 (the lesson says so)', () => {
  const lr = 0.05;
  const noZ = trainingLoop(model, { lr, iterations: 10, zeroGrad: false });
  const mom1 = simulate(model, { lr, momentum: 1, steps: 10 });
  mom1.steps.forEach((row, i) => { for (const k of ['w', 'b', 'loss']) assert.ok(near(row[k], noZ[i][k], 1e-9), `iteration ${i + 1} ${k}`); });
});

test('the Hessian, its eigenvalues and the critical learning rate equal the derived values in the data', () => {
  const H = hessian(model.x);
  O.hessian.matrix.forEach((row, i) => row.forEach((v, j) => assert.ok(near(H[i][j], v, 1e-12))));
  const lam = eigenvalues2(H);
  assert.ok(near(lam[0], O.hessian.eigenvalues[0], 1e-12) && near(lam[1], O.hessian.eigenvalues[1], 1e-12));
  assert.ok(near(2 / lam[0], O.hessian.critical_lr, 1e-12));
  assert.deepEqual(model.x, O.data.x);
});

test('analyze matches every recorded analysis, and the stability predicate agrees with the observed divergence', () => {
  const lam = eigenvalues2(hessian(model.x));
  for (const r of O.runs) {
    const a = analyze(r.lr, r.momentum, lam);
    assert.ok(near(a.critical_lr, r.analysis.critical_lr, 1e-12), `${r.id} critical`);
    assert.ok(near(a.spectral_radius, r.analysis.spectral_radius, 1e-9), `${r.id} spectral radius`);
    assert.equal(a.stable_predicted, r.analysis.stable_predicted, r.id);
    assert.equal(a.stable_predicted, !r.observed_diverged, `${r.id}: lr < 2(1+mu)/lambda_max agrees with what TorC did`);
    assert.equal(a.stable_predicted, a.spectral_radius < 1, `${r.id}: radius < 1 is the same condition`);
  }
  assert.deepEqual(O.runs.filter((r) => r.observed_diverged).map((r) => r.id), ['sgd-lr0.2']);
});

test('the predicate also separates a sweep of learning rates (100 updates), with and without momentum', () => {
  const lam = eigenvalues2(hessian(model.x));
  for (const mu of [0, 0.5, 0.9]) {
    const crit = analyze(0.1, mu, lam).critical_lr;
    for (const f of [0.15, 0.4, 0.7, 0.9, 1.15, 1.4, 2]) {
      const lr = f * crit;
      const s = simulate(model, { lr, momentum: mu, steps: 200 });
      const a = analyze(lr, mu, lam);
      assert.equal(a.stable_predicted, f < 1);
      if (f <= 0.9) assert.ok(s.final.loss < s.steps[0].loss, `mu ${mu} lr ${lr.toFixed(3)} should converge`);
      if (f >= 1.4) assert.ok(s.final.loss > s.steps[0].loss, `mu ${mu} lr ${lr.toFixed(3)} should diverge`);
    }
  }
});
