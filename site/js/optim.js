// optim.js — a JavaScript port of what optim::SGD does, plus the small least-squares model of the "sgd-step" lesson.
//
// Source of truth: TorC (src/optim.cpp, SGD::step and SGD::zero_grad). This file only re-implements the same formulas so a
// learner can change the learning rate and watch the result recompute. site/test/optim.test.mjs checks it against every
// run recorded from the real library in data/optim.json.
//
//   no momentum:   p = p - lr * grad
//   momentum:      v = momentum * v + grad;   p = p - lr * v        (v starts at 0, so the first step is the plain step)
//   a parameter without a gradient (has_grad() == false) is skipped silently: p and v stay as they were.
//
// Pure functions, no DOM. Parameters and gradients are plain arrays of numbers (one array per parameter tensor).
import { buildGraph, runBackward } from './engine.js';

// ---- the optimizer -------------------------------------------------------------------------------------------
// Same checks and same message text as the SGD constructor (the C++ compares floats; NaN is not special-cased here either).
export function validateSGD(lr, momentum = 0) {
  if (lr <= 0) throw Object.assign(new Error('SGD: lr must be positive'), { name: 'TorcError' });
  if (momentum < 0) throw Object.assign(new Error('SGD: momentum must be non-negative'), { name: 'TorcError' });
}

// The optimizer's own state: one velocity tensor per parameter, zero at the start.
export const initialState = (params) => ({ velocities: params.map((p) => p.map(() => 0)) });

// sgdStep(state, params, grads, {lr, momentum}) -> {params, state, applied}
//   params  [[w], [b], ...]            tensors flattened, one array per parameter
//   grads   [[gw], null, ...]          null/undefined = this parameter has no gradient (has_grad() is false) -> skipped
//   applied [true, false, ...]         whether the update ran for each parameter
// Inputs are never modified; new arrays are returned.
export function sgdStep(state, params, grads, { lr, momentum = 0 }) {
  validateSGD(lr, momentum);
  const vel = state?.velocities ?? initialState(params).velocities;
  const outP = [], outV = [], applied = [];
  params.forEach((p, i) => {
    const g = grads[i];
    if (g == null) { outP.push(p.slice()); outV.push(vel[i].slice()); applied.push(false); return; }   // `if (!param->has_grad()) continue;`
    if (g.length !== p.length) throw new Error(`gradient has ${g.length} elements but the parameter has ${p.length}`);
    if (momentum !== 0) {
      const v = vel[i].map((vk, k) => momentum * vk + g[k]);          // velocities_[i] = velocities_[i].mul(momentum_).add(grad)
      outV.push(v);
      outP.push(p.map((pk, k) => pk - lr * v[k]));                    // param = param - velocity * lr
    } else {
      outV.push(vel[i].slice());                                      // the velocity is not touched without momentum
      outP.push(p.map((pk, k) => pk - lr * g[k]));                    // param = param - grad * lr
    }
    applied.push(true);
  });
  return { params: outP, state: { velocities: outV }, applied };
}

// ---- the model of the lesson: y = w x + b, loss = mean((y - t)^2), evaluated with the ENGINE (the same op rules that
// reproduce TorC's graph) so the loss landscape and every gradient come from one place -------------------------------
export function makeModel(scene) {
  const leaf = (n) => scene.nodes.find((q) => q.name === n).value.values;
  const x = leaf('x').slice(), t = leaf('t').slice();
  const run = (w, b, withGrad) => {
    const g = buildGraph(scene, { W: [w], b: [b] });
    const loss = g.byName[scene.root].value.v[0];
    if (!withGrad) return { loss };
    const bw = runBackward(g);
    return { loss, gw: bw.grad.W.v[0], gb: bw.grad.b.v[0] };
  };
  return {
    x, t, n: x.length,
    loss: (w, b) => run(w, b, false).loss,                // forward pass only
    eval: (w, b) => run(w, b, true),                      // forward + backward: {loss, gw, gb}
  };
}

// ---- running the optimizer on the model --------------------------------------------------------------------------
// simulate(model, {lr, momentum, steps, start}) -> {steps[], final}
//   steps[i] = {w, b, loss, gw, gb, vw, vb} is the state BEFORE update i+1 (same convention as data/optim.json runs[]):
//   the parameters, the loss there, and the gradients step() is about to read; vw, vb is the direction step() applies
//   (the new velocity with momentum, the gradient itself without).  `final` is the same record after the last update.
export function simulate(model, { lr, momentum = 0, steps = 10, start = [0, 0] } = {}) {
  validateSGD(lr, momentum);
  let params = [[start[0]], [start[1]]];
  let state = initialState(params);
  const rec = [];
  for (let i = 0; i <= steps; i++) {
    const [[w], [b]] = params;
    const e = model.eval(w, b);
    const row = { w, b, loss: e.loss, gw: e.gw, gb: e.gb, vw: e.gw, vb: e.gb };
    if (i === steps) return { steps: rec, final: row };
    const next = sgdStep(state, params, [[e.gw], [e.gb]], { lr, momentum });
    if (momentum !== 0) { row.vw = next.state.velocities[0][0]; row.vb = next.state.velocities[1][0]; }
    rec.push(row);
    params = next.params; state = next.state;
  }
  return null;
}

// The lesson's training loop with or without optimizer.zero_grad(): every iteration is forward + backward + step() (+ zero_grad()).
// Without zero_grad() the gradient step() reads is the running sum of every pass so far.
// iterations[i] = {iteration, w, b, loss, gw, gb, gw_this_pass, gb_this_pass} like data/optim.json no_zero_grad.training_loop.
export function trainingLoop(model, { lr, momentum = 0, iterations = 10, zeroGrad = true, start = [0, 0] } = {}) {
  validateSGD(lr, momentum);
  let params = [[start[0]], [start[1]]];
  let state = initialState(params);
  let grad = null;                                               // null = has_grad() is false
  const out = [];
  for (let k = 1; k <= iterations; k++) {
    const [[w], [b]] = params;
    const e = model.eval(w, b);                                  // forward pass + loss.backward() at the current parameters
    grad = grad ? [[grad[0][0] + e.gw], [grad[1][0] + e.gb]] : [[e.gw], [e.gb]];   // accumulate_grad: copy the first time, add after
    out.push({ iteration: k, w, b, loss: e.loss, gw: grad[0][0], gb: grad[1][0], gw_this_pass: e.gw, gb_this_pass: e.gb });
    ({ params, state } = sgdStep(state, params, grad, { lr, momentum }));
    if (zeroGrad) grad = null;                                   // optimizer.zero_grad()
  }
  return out;
}

// k forward+backward passes at the same parameters with no step() and no zero_grad(): the gradient is k times one pass.
export function accumulatePasses(model, { passes = 3, start = [0, 0] } = {}) {
  const e = model.eval(start[0], start[1]);
  const out = [];
  let gw = 0, gb = 0;
  for (let k = 1; k <= passes; k++) { gw += e.gw; gb += e.gb; out.push({ iteration: k, w: start[0], b: start[1], loss: e.loss, gw, gb, gw_this_pass: e.gw, gb_this_pass: e.gb }); }
  return out;
}

// ---- derived mathematics (NOT library output): the Hessian of this loss and the critical learning rate ----------------
// L(w, b) = (1/N) sum (w x_i + b - t_i)^2 is exactly quadratic, so its Hessian is the constant matrix
//   H = (2/N) [[sum x^2, sum x], [sum x, N]].
export function hessian(x) {
  const n = x.length, sxx = x.reduce((s, v) => s + v * v, 0), sx = x.reduce((s, v) => s + v, 0);
  return [[(2 / n) * sxx, (2 / n) * sx], [(2 / n) * sx, 2]];
}

// Eigenvalues of a symmetric 2x2 matrix, largest first.
export function eigenvalues2([[a, c], [, d]]) {
  const m = (a + d) / 2, r = Math.hypot((a - d) / 2, c);
  return [m + r, m - r];
}

// Plain gradient descent on a quadratic multiplies the error along an eigenvector by 1 - lr*lambda; with momentum the error obeys
// r^2 - (1 + mu - lr*lambda) r + mu = 0 (largest root modulus = spectral radius). Stable iff the radius is below 1, i.e. iff
// 0 <= mu < 1 AND lr < 2 (1 + mu) / lambda_max (the product of the two roots is mu, so mu >= 1 can never be stable: at mu = 1 the
// velocity never decays, which is what a loop without zero_grad() amounts to).
export function analyze(lr, momentum, lambdas) {
  const radius = (lam) => {
    const tr = 1 + momentum - lr * lam, disc = tr * tr - 4 * momentum;
    return disc >= 0 ? Math.max(Math.abs(tr + Math.sqrt(disc)), Math.abs(tr - Math.sqrt(disc))) / 2 : Math.sqrt(momentum);
  };
  const radii = lambdas.map(radius);
  const spectral = Math.max(...radii);
  const critical = (2 * (1 + momentum)) / Math.max(...lambdas);
  return {
    critical_lr: critical,
    spectral_radius: spectral,
    stable_predicted: momentum >= 0 && momentum < 1 && lr < critical,
    factors: lambdas.map((lam) => 1 - lr * lam),                 // per-step error factor along each eigenvector (no momentum)
    radii,
  };
}
