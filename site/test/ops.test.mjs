// The op registry (site/js/ops/): completeness, ground-truth coverage, and numeric rules checked against finite differences.
import test from 'node:test';
import assert from 'node:assert/strict';
import { REGISTRY, OP_NAMES, getOp } from '../js/ops/index.js';
import { buildGraph, runBackward } from '../js/engine.js';
import { forwardWindow, backwardWindow } from '../js/explain.js';
import { T } from '../js/ops/tensor.js';
import { allScenes, renderStrictAll, rng, close } from './helpers.mjs';

// Ops that are registered and tested here but do not yet appear in any committed trace scene. The "every registered op
// appears in a trace scene" rule below skips exactly these. KEEP THIS EMPTY: add an op here only while its trace scene is
// still being written, and delete the entry in the same change that commits the scene.
export const OPS_AWAITING_TRACE = [];

const tracedOps = () => new Set(allScenes().flatMap(({ scene }) => scene.nodes.map((n) => n.op)).filter((op) => op !== 'leaf'));

test('registry: every op has the full interface', () => {
  assert.deepEqual([...OP_NAMES].sort(), ['add', 'matmul', 'mean', 'mse', 'mul', 'relu', 'sub', 'sum', 'transpose']);
  for (const [key, op] of Object.entries(REGISTRY)) {
    assert.equal(op.name, key, `${key}: name must equal the registry key`);
    assert.ok([1, 2].includes(op.arity), `${key}: arity`);
    for (const f of ['forward', 'backward', 'texForward', 'texBackward', 'forwardNumbers', 'backwardNumbers']) assert.equal(typeof op[f], 'function', `${key}.${f}`);
    assert.ok(typeof op.title === 'string' && op.title.length > 1, `${key}.title`);
    for (const s of ['whyForward', 'whyBackward']) assert.ok(typeof op[s] === 'string' && op[s].length > 10, `${key}.${s}`);
    assert.equal(op.texBackward('n', ['a', 'b'].slice(0, op.arity)).length, op.arity, `${key}: one backward rule per input`);
  }
  assert.throws(() => getOp('conv2d'), /unknown op "conv2d"/);
  assert.throws(() => getOp('toString'), /unknown op/, 'prototype keys are not ops');
});

test('registry: every registered op appears in some trace scene (ground truth), except OPS_AWAITING_TRACE', (t) => {
  const traced = tracedOps();
  for (const name of OPS_AWAITING_TRACE) assert.ok(OP_NAMES.includes(name), `OPS_AWAITING_TRACE lists unknown op "${name}"`);
  const missing = OP_NAMES.filter((n) => !traced.has(n) && !OPS_AWAITING_TRACE.includes(n));
  assert.deepEqual(missing, [], `ops with no trace scene (add a scene in tools/trace/): ${missing.join(', ')}`);
  const stale = OPS_AWAITING_TRACE.filter((n) => traced.has(n));
  if (stale.length) t.diagnostic(`OPS_AWAITING_TRACE is stale: ${stale.join(', ')} now has a trace scene; remove it from the list`);
});

test('every op named in a trace scene is registered', () => {
  for (const op of tracedOps()) assert.ok(OP_NAMES.includes(op), `trace uses op "${op}" which has no module in site/js/ops/`);
});

// ---- synthetic scenes for numeric checks ----------------------------------------------------------------
const leaf = (name, shape, v, grad = true) => ({ name, op: 'leaf', role: 'leaf', requires_grad: grad, inputs: [], value: { shape, values: v } });
const op = (name, o, inputs) => ({ name, op: o, role: 'op', inputs });
const randomVals = (n, r, lo = -2, hi = 2) => Array.from({ length: n }, () => lo + (hi - lo) * r());

// d(root)/d(leaf[k]) by central differences on the JS forward pass
function numericGrad(scene, leafName, k, h = 1e-5) {
  const l = scene.nodes.find((n) => n.name === leafName);
  const f = (d) => { const v = l.value.values.slice(); v[k] += d; return buildGraph(scene, { [leafName]: v }).byName[scene.root].value.v[0]; };
  return (f(h) - f(-h)) / (2 * h);
}
function checkAgainstFiniteDifferences(scene, names) {
  const bw = runBackward(buildGraph(scene));
  for (const name of names) {
    const analytic = bw.grad[name].v;
    analytic.forEach((a, k) => assert.ok(Math.abs(numericGrad(scene, name, k) - a) < 1e-6 * (1 + Math.abs(a)), `d/d${name}[${k}]: numeric ${numericGrad(scene, name, k)} vs analytic ${a}`));
  }
  return bw;
}

test('transpose: forward swaps entries; backward = transpose of the incoming gradient (finite differences, random tensors)', () => {
  const r = rng(7);
  for (const [rows, cols] of [[3, 2], [1, 4], [4, 1], [2, 2]]) {
    const a = randomVals(rows * cols, r), w = randomVals(rows * cols, r);
    const scene = { id: 't', root: 's', nodes: [leaf('a', [rows, cols], a), leaf('w', [cols, rows], w, false), op('at', 'transpose', ['a']), op('m', 'mul', ['at', 'w']), op('s', 'sum', ['m'])] };
    const g = buildGraph(scene);
    const at = g.byName.at.value;
    assert.deepEqual(at.shape, [cols, rows]);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) assert.equal(at.v[j * rows + i], a[i * cols + j]);
    const bw = checkAgainstFiniteDifferences(scene, ['a']);
    // exact rule: s = sum(a^T ⊙ w) so ds/da = w^T
    const wT = T([cols, rows], w); const expected = getOp('transpose').forward(wT);
    assert.deepEqual(bw.grad.a.v, expected.v);
    assert.deepEqual(bw.grad.a.shape, [rows, cols]);
  }
  // rank 1 is the identity (as in the library), rank 3 is outside this port
  assert.deepEqual(getOp('transpose').forward(T([3], [1, 2, 3])).v, [1, 2, 3]);
  assert.throws(() => getOp('transpose').forward(T([1, 2, 2], [1, 2, 3, 4])), /rank 1 or 2/);
});

test('mse: forward = mean((p-t)^2); backward = 2(p-t)/N * g for the prediction and the negative for the target', () => {
  const r = rng(99);
  for (const shape of [[1], [5], [2, 3], [4, 1]]) {
    const n = shape.reduce((a, b) => a * b, 1);
    const p = randomVals(n, r), t = randomVals(n, r), c = 3.7;
    const scene = { id: 'm', root: 'y', nodes: [leaf('p', shape, p), leaf('t', shape, t), leaf('c', [1], [c], false), op('L', 'mse', ['p', 't']), op('y', 'mul', ['L', 'c'])] };
    const g = buildGraph(scene);
    const want = p.reduce((s, x, i) => s + (x - t[i]) ** 2, 0) / n;
    assert.ok(Math.abs(g.byName.L.value.v[0] - want) < 1e-12 * (1 + want));
    assert.deepEqual(g.byName.L.value.shape, [1]);
    const bw = checkAgainstFiniteDifferences(scene, ['p', 't']);
    // upstream gradient reaching mse is c, so the rule gives 2 (p - t) / N * c
    p.forEach((x, i) => {
      const gp = (2 * (x - t[i]) / n) * c;
      assert.ok(Math.abs(bw.grad.p.v[i] - gp) < 1e-12 * (1 + Math.abs(gp)), `grad p[${i}]`);
      assert.ok(Math.abs(bw.grad.t.v[i] + gp) < 1e-12 * (1 + Math.abs(gp)), `grad t[${i}] is the negative`);
    });
  }
  // hand-computed: p=[1,2], t=[0,4] -> (1 + 4)/2 = 2.5; grad p = [1,-2]; grad t = [-1,2]
  const hand = { id: 'h', root: 'L', nodes: [leaf('p', [2], [1, 2]), leaf('t', [2], [0, 4]), op('L', 'mse', ['p', 't'])] };
  const hg = buildGraph(hand), hb = runBackward(hg);
  assert.deepEqual(hg.byName.L.value.v, [2.5]);
  assert.deepEqual(hb.grad.p.v, [1, -2]);
  assert.deepEqual(hb.grad.t.v, [-1, 2]);
  assert.throws(() => getOp('mse').forward(T([2], [1, 2]), T([3], [1, 2, 3])), /same shape/);
});

test('mse with a constant target (as in training): the target gets no gradient', () => {
  const scene = { id: 'm', root: 'L', nodes: [leaf('p', [2], [1, 2]), leaf('t', [2], [0, 4], false), op('L', 'mse', ['p', 't'])] };
  const bw = runBackward(buildGraph(scene));
  assert.deepEqual(bw.grad.p.v, [1, -2]);
  assert.equal(bw.grad.t, undefined);
});

test('explanations of transpose and mse render in KaTeX and show the numbers', () => {
  const scene = { id: 'x', root: 'L', nodes: [leaf('x', [2, 3], [1, 2, 3, -1, 0, 2]), leaf('W', [2, 3], [0.5, -1, 2, 1, 0, -0.5]), leaf('t', [2, 2], [4, 0, 3, -2], false),
    op('Wt', 'transpose', ['W']), op('y', 'matmul', ['x', 'Wt']), op('L', 'mse', ['y', 't'])] };
  const g = buildGraph(scene), bw = runBackward(g);
  for (const name of ['Wt', 'L']) {
    for (const w of [forwardWindow(g, name), backwardWindow(g, bw, name)]) w.blocks.forEach((b) => renderStrictAll(b.latex ?? []));
  }
  const text = (w) => w.blocks.map((b) => (b.latex ?? []).join(' ')).join(' ');
  assert.match(text(forwardWindow(g, 'Wt')), /\\top/);
  assert.match(text(backwardWindow(g, bw, 'L')), /\\dfrac\{2\}\{4\}/, 'N = 4 elements appears in the numeric substitution');
  assert.match(text(backwardWindow(g, bw, 'Wt')), /\\top/);
  // the scalar case (N = 1) also renders
  const one = { id: 'o', root: 'L', nodes: [leaf('p', [1], [3]), leaf('t', [1], [1]), op('L', 'mse', ['p', 't'])] };
  const g1 = buildGraph(one), b1 = runBackward(g1);
  for (const w of [forwardWindow(g1, 'L'), backwardWindow(g1, b1, 'L')]) w.blocks.forEach((b) => renderStrictAll(b.latex ?? []));
  assert.deepEqual(g1.byName.L.value.v, [4]);
});

test('add/sub/mul/relu/sum/mean/matmul still give the documented values', () => {
  const a = T([2, 2], [1, -2, 3, 4]), b = T([2], [10, 20]);
  assert.deepEqual(getOp('add').forward(a, b).v, [11, 18, 13, 24]);          // broadcast over rows
  assert.deepEqual(getOp('sub').forward(a, b).v, [-9, -22, -7, -16]);
  assert.deepEqual(getOp('mul').forward(a, b).v, [10, -40, 30, 80]);
  assert.deepEqual(getOp('relu').forward(a).v, [1, 0, 3, 4]);
  assert.deepEqual(getOp('sum').forward(a).v, [6]);
  assert.deepEqual(getOp('mean').forward(a).v, [1.5]);
  assert.deepEqual(getOp('matmul').forward(a, a).v, [-5, -10, 15, 10]);
  assert.ok(close([1], [1]));
});
