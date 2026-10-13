// ops/texutil.js — number/tensor formatting and LaTeX helpers used by every op's explanation text.
import { numel } from './tensor.js';

export function fmt(x) {
  if (Object.is(x, -0) || Math.abs(x) < 1e-9) return '0';
  const s = Number(x.toPrecision(4)).toString();
  return s.includes('e') ? Number(x.toPrecision(3)).toExponential().replace('e+', 'e') : s;
}
export const tn = (x) => fmt(x);                                          // number as TeX
export const par = (x) => (x < 0 ? `(${tn(x)})` : tn(x));                 // factor: parenthesise negatives
export const isScalar = (t) => numel(t.shape) === 1;

export function tex(name) {                                               // W1 -> W_1, xW1 -> xW_1
  const m = /^([A-Za-z]+)(\d+)$/.exec(name);
  const base = name.length === 1 ? `\\mathit{${name}}` : `\\texttt{${name}}`;   // multi-letter node names are code names, not a product of variables
  return m ? (m[1].length === 1 ? `{\\mathit{${m[1]}}_{${m[2]}}}` : `{\\texttt{${m[1]}}_{${m[2]}}}`) : base;
}
export const bar = (name) => `\\overline{${tex(name)}}`;                  // gradient notation: d out / d name
export const txt = (s) => `\\text{${s}}`;

export function mat(t) {                                                  // tensor -> TeX
  if (isScalar(t)) return tn(t.v[0]);
  if (t.shape.length === 2 && t.shape[0] > 1) {
    const [r, c] = t.shape;
    const rows = [];
    for (let i = 0; i < r; i++) rows.push(t.v.slice(i * c, (i + 1) * c).map(tn).join(' & '));
    return `\\begin{bmatrix}${rows.join(' \\\\ ')}\\end{bmatrix}`;
  }
  return `\\begin{bmatrix}${t.v.map(tn).join(' & ')}\\end{bmatrix}`;
}
export const shapeStr = (t) => (isScalar(t) ? 'a single number' : `shape (${t.shape.join(', ')})`);
export const plus = (list) => list.map((s, i) => (i === 0 ? s : s.startsWith('-') ? ` - ${s.slice(1)}` : ` + ${s}`)).join('');
// rows written as "n &= ..." where the leading n stands for the node name
export const aligned = (rows, name) => `\\begin{aligned}${rows.map((r) => r.replace(/^n/, tex(name))).join(' \\\\ ')}\\end{aligned}`;
export const negate = (t) => ({ shape: t.shape, v: t.v.map((x) => -x) });
