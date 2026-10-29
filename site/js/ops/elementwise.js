// ops/elementwise.js — the "paper work" shared by add, sub and mul (two inputs broadcast to a common shape).
import { expandTo } from './tensor.js';
import { tex, mat, aligned } from './texutil.js';

// sym: TeX operator between the two matrices; entry(a, b, out): "a + b = out" for one element;
// alwaysElementwise: also list the element-by-element lines for a single number (add/sub do, mul does not).
export function elementwiseForwardNumbers(sym, entry, alwaysElementwise) {
  return ({ name, node, inputs: [a, b], out }) => {
    const shape = out.shape;
    const A = expandTo(a, shape), B = expandTo(b, shape);
    const latex = [];
    let note = '';
    const expanded = (a.shape.join() !== shape.join() ? [`${tex(node.inputs[0])} \\to ${mat(A)}`] : [])
      .concat(b.shape.join() !== shape.join() ? [`${tex(node.inputs[1])} \\to ${mat(B)}`] : []);
    if (expanded.length) { latex.push(expanded.join(',\\quad ')); note = 'The smaller input is copied (broadcast) so both have the same shape:'; }
    latex.push(`${mat(A)} ${sym} ${mat(B)} = ${mat(out)}`);
    if (out.v.length > 1 || alwaysElementwise) {
      const rows = out.v.map((_, i) => `${out.v.length > 1 ? `n_{${i}}` : 'n'} &= ${entry(A.v[i], B.v[i], out.v[i])}`);
      latex.push(aligned(rows, name));
    }
    return { note, latex };
  };
}
