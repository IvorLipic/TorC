// engine.js — a small JavaScript port of the autograd rules in TorC (src/autograd.cpp),
// used so that learners can edit inputs and watch values and gradients recompute.
//
// It is NOT the source of truth: site/test/engine.test.mjs checks that, for the preset
// inputs, this engine reproduces every value and gradient that the real C++ library
// produced (data/autograd_trace.json, written by visualizations/examples/autograd_trace.cpp).
//
// Tensors are {shape: number[], v: number[]} (row-major). Only the ops used in the scenes
// exist here: add, sub, mul, relu, sum, mean, matmul.

export const numel = (shape) => shape.reduce((a, b) => a * b, 1);
export const T = (shape, v) => ({ shape: shape.slice(), v: v.slice() });
export const fill = (shape, x) => T(shape, new Array(numel(shape)).fill(x));
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

// ---- broadcasting (NumPy rules, as in Tensor::broadcast_shape) -----------------
export function broadcastShape(a, b) {
  const n = Math.max(a.length, b.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    const da = a[a.length - 1 - i] ?? 1, db = b[b.length - 1 - i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) throw new Error(`shapes ${a} and ${b} do not broadcast`);
    out.unshift(Math.max(da, db));
  }
  return out;
}

// For each flat index of `outShape`, the flat index into a tensor of `shape` that is broadcast to it.
function indexMap(shape, outShape) {
  const n = numel(outShape), rank = outShape.length, pad = rank - shape.length;
  const strides = new Array(rank).fill(0);
  let s = 1;
  for (let d = shape.length - 1; d >= 0; d--) { strides[d + pad] = shape[d] === 1 ? 0 : s; s *= shape[d]; }
  const map = new Array(n);
  for (let i = 0; i < n; i++) {
    let rem = i, idx = 0;
    for (let d = rank - 1; d >= 0; d--) { idx += (rem % outShape[d]) * strides[d]; rem = Math.floor(rem / outShape[d]); }
    map[i] = idx;
  }
  return map;
}

export function expandTo(t, outShape) {
  if (same(t.shape, outShape)) return T(t.shape, t.v);
  const m = indexMap(t.shape, outShape);
  return T(outShape, m.map((j) => t.v[j]));
}

// reduce_sum_to_shape: sum a gradient of broadcast shape back down to `target`.
export function reduceSumToShape(g, target) {
  if (same(g.shape, target)) return T(g.shape, g.v);
  const m = indexMap(target, g.shape);
  const out = new Array(numel(target)).fill(0);
  g.v.forEach((x, i) => { out[m[i]] += x; });
  return T(target, out);
}

const zip = (a, b, f) => {
  const shape = broadcastShape(a.shape, b.shape);
  const A = expandTo(a, shape), B = expandTo(b, shape);
  return T(shape, A.v.map((x, i) => f(x, B.v[i])));
};
const map = (a, f) => T(a.shape, a.v.map(f));

export function matmul(a, b) {
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0])
    throw new Error(`matmul shape mismatch ${a.shape} x ${b.shape}`);
  const [n, k] = a.shape, m = b.shape[1];
  const out = new Array(n * m).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    let s = 0;
    for (let p = 0; p < k; p++) s += a.v[i * k + p] * b.v[p * m + j];
    out[i * m + j] = s;
  }
  return T([n, m], out);
}
export function transpose(a) {
  const [n, m] = a.shape;
  const out = new Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j * n + i] = a.v[i * m + j];
  return T([m, n], out);
}

// ---- forward ----------------------------------------------------------------
const FORWARD = {
  add: (a, b) => zip(a, b, (x, y) => x + y),
  sub: (a, b) => zip(a, b, (x, y) => x - y),
  mul: (a, b) => zip(a, b, (x, y) => x * y),
  relu: (a) => map(a, (x) => (x > 0 ? x : 0)),
  sum: (a) => T([1], [a.v.reduce((s, x) => s + x, 0)]),
  mean: (a) => T([1], [a.v.reduce((s, x) => s + x, 0) / a.v.length]),
  matmul,
};

// ---- local backward rules: (grad of output, input values, output value) -> raw input grads ---
// "Raw" = shape of the broadcast output; the caller reduces to each input's own shape.
const BACKWARD = {
  add: (g) => [g, g],
  sub: (g) => [g, map(g, (x) => -x)],
  mul: (g, [a, b]) => [zip(g, b, (x, y) => x * y), zip(g, a, (x, y) => x * y)],
  relu: (g, [x]) => [T(g.shape, g.v.map((gi, i) => (x.v[i] > 0 ? gi : 0)))],
  sum: (g, [x]) => [fill(x.shape, g.v[0])],
  mean: (g, [x]) => [fill(x.shape, g.v[0] / x.v.length)],
  matmul: (g, [a, b]) => [matmul(g, transpose(b)), matmul(transpose(a), g)],
};
export const OPS = Object.keys(FORWARD);

// ---- graph ------------------------------------------------------------------
// `scene.nodes` is the list from the real library trace: {name, op, role, requires_grad, inputs, value:{shape,values}}
// in construction order (which is a valid topological order). `overrides` maps leaf name -> number[].
export function buildGraph(scene, overrides = {}) {
  const nodes = [], byName = {};
  for (const n of scene.nodes) {
    const node = { name: n.name, op: n.role === 'leaf' ? 'leaf' : n.op, role: n.role, inputs: n.inputs.slice(), index: nodes.length };
    if (n.role === 'leaf') {
      node.requires_grad = n.requires_grad;
      node.value = T(n.value.shape, overrides[n.name] ?? n.value.values);
    } else {
      node.requires_grad = n.inputs.some((i) => byName[i].requires_grad);
      node.value = FORWARD[n.op](...n.inputs.map((i) => byName[i].value));
    }
    nodes.push(node);
    byName[n.name] = node;
  }
  return { nodes, byName, root: scene.root };
}

// Follows Variable::backward_with_grad: the library visits nodes in reverse DFS post-order, this visits them in reverse
// construction order. For the scenes here the order of the operations is identical (only where leaves are visited can
// differ, and leaves do no work). Visit nodes in reverse construction order; each node's
// gradient is the SUM of everything its consumers sent; each op's rule then sends raw gradients to
// its inputs, which are reduced to the input shape and accumulated.
export function runBackward(graph) {
  const { nodes, byName, root } = graph;
  const received = {};  // name -> [{from, arg, tensor, running}]
  const steps = [];
  const grad = {};
  for (const n of nodes) received[n.name] = [];
  const sumTensors = (list, shape) => list.reduce((acc, c) => T(shape, acc.v.map((x, i) => x + c.tensor.v[i])), fill(shape, 0));

  for (let k = nodes.length - 1; k >= 0; k--) {
    const n = nodes[k];
    if (!n.requires_grad) continue;
    const isRoot = n.name === root;
    const incoming = received[n.name];
    if (!isRoot && incoming.length === 0) continue;          // not reachable from the root
    const G = isRoot ? fill(n.value.shape, 1) : sumTensors(incoming, n.value.shape);
    grad[n.name] = isRoot ? null : G;                        // the library never stores the root's own grad
    const step = { node: n.name, isRoot, incoming: incoming.slice(), G, sends: [] };
    if (n.role === 'op') {
      const ins = n.inputs.map((i) => byName[i].value);
      const raws = BACKWARD[n.op](G, ins, n.value);
      n.inputs.forEach((iname, arg) => {
        const inp = byName[iname];
        const raw = raws[arg];
        const tracked = inp.requires_grad;
        const send = { to: iname, arg, tracked, raw, tensor: null, reduced: false, running: null };
        if (tracked) {
          send.reduced = !same(raw.shape, inp.value.shape);
          send.tensor = reduceSumToShape(raw, inp.value.shape);
          received[iname].push({ from: n.name, arg, tensor: send.tensor });
          send.running = sumTensors(received[iname], inp.value.shape);
        }
        step.sends.push(send);
      });
    }
    steps.push(step);
  }
  return { steps, grad };
}

// The ordered list of everything the learner steps through.
export function timeline(graph, bw) {
  const out = [{ kind: 'inputs' }];
  for (const n of graph.nodes) if (n.role === 'op') out.push({ kind: 'forward', node: n.name });
  for (const s of bw.steps) out.push({ kind: 'backward', node: s.node });
  return out;
}
