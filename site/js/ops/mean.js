// ops/mean.js — mean of all elements (one number).
import { T, fill } from './tensor.js';
import { tex, bar, mat, tn, plus } from './texutil.js';

export default {
  name: 'mean', title: 'mean', arity: 1,
  forward: (a) => T([1], [a.v.reduce((s, x) => s + x, 0) / a.v.length]),
  backward: (g, [x]) => [fill(x.shape, g.v[0] / x.v.length)],
  texForward: (n, [a]) => `${tex(n)} = \\tfrac{1}{N}\\sum_i ${tex(a)}_i`,
  whyForward: 'Add all the elements and divide by how many there are (N).',
  forwardNumbers: ({ name, inputs: [a], out }) => ({ note: '', latex: [`${tex(name)} = \\tfrac{1}{${a.v.length}}\\left(${plus(a.v.map(tn))}\\right) = ${tn(out.v[0])}`] }),
  texBackward: (n, [a]) => [`${bar(a)}_i = \\dfrac{${bar(n)}}{N}\\quad\\text{for every } i`],
  whyBackward: 'Every element is added once and then divided by N, so each has slope 1/N.',
  backwardNumbers: ({ graph, s, G }) => [`${bar(s.to)}_i = \\dfrac{${tn(G.v[0])}}{${graph.byName[s.to].value.v.length}} = ${tn(s.raw.v[0])}\\ \\Rightarrow\\ ${bar(s.to)} = ${mat(s.raw)}`],
};
