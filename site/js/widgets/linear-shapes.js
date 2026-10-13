// widgets/linear-shapes.js — pure logic of the `linear-layer` shape explorer (no DOM, no globals, importable in Node).
//
// Rules, as TorC implements them (src/nn/linear.cpp, src/tensor.cpp):
//   Linear(in, out):  weight (out, in), bias (out), forward(x) = add(matmul(x, transpose(weight)), bias)
//   x (B, in) -> y (B, out);  parameters = out*in + out (two tensors).
//   A wrong input width fails inside matmul with   "matmul inner dimensions must match, got (B, w) and (in, out)"
//   (the second shape is the TRANSPOSED weight); a rank-1 input fails with "matmul requires rank >= 2 operands, ...".
// Rank-3 inputs are deliberately not modelled (the library accepts them; their gradients are not verified for the site).
// The predicted message text is checked, in site/test/linear-layer.test.mjs, against every message TorC recorded in linear.json.

export const LIMITS = { in: { min: 1, max: 1024 }, out: { min: 1, max: 1024 }, batch: { min: 1, max: 128 } };

// TorC prints shapes like "(2, 2)" and "(3)".
export const fmtShape = (s) => '(' + s.join(', ') + ')';
export const numel = (s) => s.reduce((a, b) => a * b, 1);
export const sameShape = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i]);

export function linearShapes(inF, outF, B) {
  return { x: [B, inF], W: [outF, inF], Wt: [inF, outF], b: [outF], y: [B, outF] };
}

export function paramCount(inF, outF) {
  const weight = inF * outF, bias = outF;
  return { weight, bias, total: weight + bias, tensors: 2 };
}

// What happens when Linear(inF, outF) is applied to an input of shape xShape.
//   {ok:true, y:[B,out]}                       the call works
//   {ok:false, exception, what}                the exception type and its what() text
//   {ok:null, unverified:true}                 rank >= 3: not modelled
export function predictCall(inF, outF, xShape) {
  if (!(inF > 0) || !(outF > 0)) return { ok: false, exception: 'TorcError', what: 'Linear features must be positive' };
  if (xShape.length < 2) return { ok: false, exception: 'ShapeError', what: `matmul requires rank >= 2 operands, got ${fmtShape(xShape)} and ${fmtShape([inF, outF])}` };
  if (xShape.length > 2) return { ok: null, unverified: true };
  if (xShape[1] !== inF) return { ok: false, exception: 'ShapeError', what: `matmul inner dimensions must match, got ${fmtShape(xShape)} and ${fmtShape([inF, outF])}` };
  return { ok: true, y: [xShape[0], outF] };
}

// The recorded real call (from linear.json `shape_errors`) for exactly this layer and input shape, or null.
export function findRecorded(shapeErrors, inF, outF, xShape) {
  return shapeErrors.find((r) => r.in === inF && r.out === outF && sameShape(r.x_shape, xShape)) ?? null;
}

// Recorded runs the explorer can reproduce with its steppers: Linear(in,out) on a rank-2 input.
export function recordedRuns(shapeErrors) {
  return shapeErrors.filter((r) => Number.isInteger(r.in) && Number.isInteger(r.out) && Array.isArray(r.x_shape) && r.x_shape.length === 2)
    .map((r) => ({ id: r.id, in: r.in, out: r.out, B: r.x_shape[0], width: r.x_shape[1], fails: r.exception !== 'none', what: r.what, call: r.call, yShape: r.y_shape ?? null }));
}

// Layer parameter counts that TorC measured for real Linear layers (linear.json `param_counts`): Map "in,out" -> {count, source}.
export function recordedLinearCounts(paramCounts) {
  const m = new Map();
  for (const entry of paramCounts) for (const L of entry.layers) {
    const g = /^Linear\((\d+), (\d+)\)$/.exec(L.layer);
    if (g && !m.has(`${g[1]},${g[2]}`)) m.set(`${g[1]},${g[2]}`, { count: L.count, source: entry.name });
  }
  return m;
}

// Box size in pixels for a dimension: grows with log(n) so that 1 .. 1024 stay on screen together.
export const boxPx = (n) => Math.round(22 + 80 * Math.log(Math.max(1, n)) / Math.log(1024));
