// ops/relu.js — max(0, x) elementwise. TorC's slope at exactly 0 is 0.
import { map, T } from './tensor.js';
import { tex, bar, mat, tn, aligned } from './texutil.js';

const slopeOf = (x) => ({ shape: x.shape, v: x.v.map((z) => (z > 0 ? 1 : 0)) });

export default {
  name: 'relu', title: 'ReLU', arity: 1,
  forward: (a) => map(a, (x) => (x > 0 ? x : 0)),
  backward: (g, [x]) => [T(g.shape, g.v.map((gi, i) => (x.v[i] > 0 ? gi : 0)))],
  texForward: (n, [a]) => `${tex(n)}_i = \\max(0,\\, ${tex(a)}_i)`,
  whyForward: 'Keep positive numbers, replace negative ones by 0.',
  forwardNumbers: ({ name, inputs: [a], out }) => {
    const rows = out.v.map((_, i) => `${out.v.length > 1 ? `n_{${i}}` : 'n'} &= \\max(0,\\, ${tn(a.v[i])}) = ${tn(out.v[i])}`);
    return { note: '', latex: [aligned(rows, name)] };
  },
  texBackward: (n, [a]) => [`${bar(a)}_i = \\begin{cases}${bar(n)}_i & ${tex(a)}_i > 0\\\\ 0 & ${tex(a)}_i \\le 0\\end{cases}`],
  whyBackward: 'ReLU is a straight line of slope 1 for positive inputs and flat (slope 0) for negative ones, so the gradient passes only where the input was positive. At exactly 0 TorC uses slope 0.',
  backwardNumbers: ({ name, node, graph, s, G }) => {
    const x = graph.byName[node.inputs[0]].value;
    const slope = mat(slopeOf(x));
    return [
      `\\text{slope: } ${slope}\\quad(1 \\text{ where } ${tex(node.inputs[0])} > 0)`,
      `${bar(s.to)} = ${bar(name)} \\odot \\text{slope} = ${mat(G)} \\odot ${slope} = ${mat(s.raw)}`,
    ];
  },
};
