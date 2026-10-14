// widgets/linear-math.js — pure arithmetic and paper-work (LaTeX) of the `linear-layer` widgets. No DOM; importable in Node.
// Matrices here are arrays of rows (number[][]); indices in the LaTeX are 1-based, as in the rest of the site's paper work.
//   forward :  y[i][k]  = sum_j x[i][j] * W[k][j] + b[k]          (x (B,in), W (out,in), b (out), y (B,out))
//   backward:  gW[k][j] = sum_i dY[i][k] * x[i][j]                 (= dY^T x)
//              gb[k]    = sum_i dY[i][k]                           (column sums: b was copied to every row)
//              dX[i][j] = sum_k dY[i][k] * W[k][j]                 (= dY W)
import { tn, par } from '../ops/texutil.js';

export const round7 = (v) => Number(v.toPrecision(7));         // float32 values are stored with 9 digits; show 7 in input boxes
export const rowsOf = (flat, r, c) => Array.from({ length: r }, (_, i) => flat.slice(i * c, (i + 1) * c));
export const clone2 = (M) => M.map((r) => r.slice());
export const colSums = (M) => M[0].map((_, j) => M.reduce((s, r) => s + r[j], 0));

const node = (scene, name) => scene.nodes.find((n) => n.name === name);
const mat = (scene, name, round = true) => {
  const n = node(scene, name);
  return rowsOf(round ? n.value.values.map(round7) : n.value.values, n.value.shape[0], n.value.shape[1] ?? 1);
};
const vec = (scene, name) => node(scene, name).value.values.map(round7);

// The preset inputs and the printed output of a trace scene shaped like `linear` (x, W, b, y).
export function exampleFromScene(scene) {
  return { x: mat(scene, 'x'), W: mat(scene, 'W'), b: vec(scene, 'b'), y: mat(scene, 'y', false) };
}

export function linearForward(x, W, b) {
  return x.map((row) => W.map((w, k) => row.reduce((s, v, j) => s + v * w[j], 0) + b[k]));
}

export const gradW = (dY, x) => dY[0].map((_, k) => x[0].map((_, j) => dY.reduce((s, r, i) => s + r[k] * x[i][j], 0)));
export const gradB = (dY) => colSums(dY);
export const gradX = (dY, W) => dY.map((r) => W[0].map((_, j) => r.reduce((s, g, k) => s + g * W[k][j], 0)));

// ---- paper work -------------------------------------------------------------------------------------
const joinPlus = (list) => list.join(' + ');
const aligned = (rows) => `\\begin{aligned}${rows.join(' \\\\ ')}\\end{aligned}`;

// y_{ik}: dot product of row i of x with row k of W, plus b_k.
export function neuronTex(x, W, b, i, k) {
  const xr = x[i], wr = W[k];
  const y = xr.reduce((s, v, j) => s + v * wr[j], 0) + b[k];
  const n = xr.length, I = i + 1, K = k + 1;
  return aligned([
    `y_{${I}${K}} &= \\sum_{j=1}^{${n}} x_{${I}j}\\,W_{${K}j} + b_{${K}}`,
    `&= ${joinPlus(xr.map((v, j) => `${par(v)}\\cdot ${par(wr[j])}`))} + ${par(b[k])}`,
    `&= ${joinPlus(xr.map((v, j) => par(v * wr[j])))} + ${par(b[k])} = ${tn(y)}`,
  ]);
}



