// ops/tensor.js — the minimal tensor helpers the op rules are written with.
// Tensors are {shape: number[], v: number[]} (row-major). Broadcasting follows NumPy / Tensor::broadcast_shape.

export const numel = (shape) => shape.reduce((a, b) => a * b, 1);
export const T = (shape, v) => ({ shape: shape.slice(), v: v.slice() });
export const fill = (shape, x) => T(shape, new Array(numel(shape)).fill(x));
export const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

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

export const zip = (a, b, f) => {
  const shape = broadcastShape(a.shape, b.shape);
  const A = expandTo(a, shape), B = expandTo(b, shape);
  return T(shape, A.v.map((x, i) => f(x, B.v[i])));
};
export const map = (a, f) => T(a.shape, a.v.map(f));

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

// Reverses the axes like Variable transpose(a) does; ranks 1 (identity) and 2 are supported.
export function transpose(a) {
  if (a.shape.length === 1) return T(a.shape, a.v);
  if (a.shape.length !== 2) throw new Error(`transpose supports rank 1 or 2 here, got shape ${a.shape}`);
  const [n, m] = a.shape;
  const out = new Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j * n + i] = a.v[i * m + j];
  return T([m, n], out);
}
