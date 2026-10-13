// widgets/landscape-model.js — the maths behind the one-weight lesson "why-gradients", with NO DOM and no plotting.
//
// Everything is computed by the same code the graph lab uses: engine.buildGraph() builds the graph of trace scene `why`
// (w -> p = w*x -> e = p - t -> L = e*e) with the weight overridden, and engine.runBackward() gives dL/dw.
// So the curve, the secants, the steps and the graph shown on the page cannot disagree. The tests compare all of it with the
// numbers TorC recorded (data/why-gradients.json: grid, probes, secants, nudge, trajectories).
//
//   const m = makeModel(D.scenes.why);
//   m.at(1.2)              -> { w, p, e, L, grad }                 forward + backward at that weight
//   m.secant(0, 0.5)       -> { h, L0, Lh, slope, gap, ... }       (L(w+h) - L(w)) / h
//   m.iterate(0, 0.05, 8)  -> states 0..8 of w <- w - lr * grad
//   m.nudge(0, 0.01)       -> change of every stage when w moves by delta, next to its local slope
//   m.factor(0.05)         -> 1 - curvature*lr    ("distance to the minimum is multiplied by this each step")
import { buildGraph, runBackward, fill } from '../engine.js';
import { getOp } from '../ops/index.js';

export const CHAIN = ['w', 'p', 'e', 'L'];                 // the stages the weight reaches the loss through

export function makeModel(scene) {
  const leaf = (name) => {
    const n = scene.nodes.find((q) => q.name === name);
    if (!n) throw new Error(`scene ${scene.id}: no node "${name}"`);
    return n.value.values[0];
  };
  const x = leaf('x'), t = leaf('t'), w0 = leaf('w');
  // L(w) = (x w - t)^2.  Derived (Expected mathematics, not library output): L'' = 2 x^2, minimum at t/x.
  const curvature = 2 * x * x;
  const wMin = t / x;

  function graphAt(w) {
    const g = buildGraph(scene, { w: [w] });
    return { g, bw: runBackward(g) };
  }
  const val = (g, name) => g.byName[name].value.v[0];

  function at(w) {
    const { g, bw } = graphAt(w);
    return { w, p: val(g, 'p'), e: val(g, 'e'), L: val(g, 'L'), grad: bw.grad.w.v[0] };
  }

  const lossAt = (w) => val(buildGraph(scene, { w: [w] }), 'L');

  // Local slope of each stage with respect to the stage before it: the op's own backward rule with a seed of 1 (so it is
  // "d stage / d previous", independent of anything downstream). An input used twice (e in L = e*e) sums both uses.
  function localSlopes(w) {
    const { g } = graphAt(w);
    return CHAIN.slice(1).map((name, i) => {
      const n = g.byName[name], from = CHAIN[i];
      const raws = getOp(n.op).backward(fill(n.value.shape, 1), n.inputs.map((q) => g.byName[q].value), n.value);
      const slope = n.inputs.reduce((s, q, k) => s + (q === from ? raws[k].v[0] : 0), 0);
      return { name, from, slope };
    });
  }

  function secant(w, h) {
    const here = at(w);
    const Lh = lossAt(w + h);
    const slope = (Lh - here.L) / h;
    return { w, h, L0: here.L, Lh, dL: Lh - here.L, slope, grad: here.grad, gap: slope - here.grad };
  }

  // states[k] = forward/backward at the weight after k steps; wNext = w - lr*grad; promise = the tangent line's value of L at wNext
  function iterate(start, lr, steps) {
    const states = [];
    let w = start;
    for (let k = 0; k <= steps; k++) {
      const s = at(w);
      const wNext = w - lr * s.grad;
      states.push({ k, ...s, wNext, promise: s.L + s.grad * (wNext - w) });
      w = wNext;
    }
    return states;
  }

  function nudge(w, delta) {
    const before = at(w), after = at(w + delta);
    const slopes = localSlopes(w);
    let prevChange = delta;
    const stages = slopes.map(({ name, from, slope }) => {
      const change = after[name] - before[name];
      const predicted = slope * prevChange;
      prevChange = change;
      return { name, from, before: before[name], after: after[name], change, slope, predicted };
    });
    const product = slopes.reduce((q, s) => q * s.slope, 1);
    return {
      w, delta, stages, product, backwardSlope: before.grad,
      predictedDL: product * delta, actualDL: after.L - before.L,
      secondOrder: (after.L - before.L) - product * delta,       // = (curvature/2) delta^2 for this bowl
    };
  }

  const factor = (lr) => 1 - curvature * lr;
  const criticalLr = 2 / curvature, newtonLr = 1 / curvature;
  // Expected mathematics: the iteration is stable iff |1 - curvature*lr| < 1.
  const stable = (lr) => Math.abs(factor(lr)) < 1;

  function verdict(lr) {
    const f = factor(lr), eps = 1e-9;
    if (Math.abs(f) < eps) return { id: 'exact', factor: f, text: 'lands exactly on the minimum in one step' };
    if (Math.abs(f + 1) < eps) return { id: 'bounce', factor: f, text: 'bounces between two points forever' };
    if (Math.abs(f) < 1) return f > 0
      ? { id: 'smooth', factor: f, text: 'creeps toward the minimum from one side' }
      : { id: 'zigzag', factor: f, text: 'jumps over the minimum each step, landing closer every time' };
    return { id: 'diverge', factor: f, text: 'overshoots by more each step and runs away' };
  }

  const curve = (a, b, n) => Array.from({ length: n + 1 }, (_, i) => at(a + ((b - a) * i) / n));

  return { x, t, w0, wMin, curvature, criticalLr, newtonLr, at, lossAt, localSlopes, secant, iterate, nudge, factor, stable, verdict, curve };
}

// ---- comparison with the recorded TorC run (used by the widgets' badges and by the tests) ------------------------------
export const sameNumber = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol * (1 + Math.abs(b));
export const near = (a, b) => Math.abs(a - b) < 1e-6;   // float32 grid values such as 0.100000001 equal the slider's 0.1

export function recordedPoint(files, w) {
  return [...files.grid, ...files.probes].find((q) => near(q.w, w)) ?? null;
}
export function recordedTrajectory(files, lr) {
  return files.trajectories.find((q) => near(q.lr, lr)) ?? null;
}
export function recordedSecant(files, h) {
  return files.secants.find((q) => near(q.h, h)) ?? null;
}
