// ops/sum.js — sum of all elements (one number).
import { T, fill } from './tensor.js';
import { tex, bar, mat, tn, plus } from './texutil.js';

export default {
  name: 'sum', title: 'sum', arity: 1,
  forward: (a) => T([1], [a.v.reduce((s, x) => s + x, 0)]),
  backward: (g, [x]) => [fill(x.shape, g.v[0])],
  texForward: (n, [a]) => `${tex(n)} = \\sum_i ${tex(a)}_i`,
  whyForward: 'Add all the elements into one number.',
  forwardNumbers: ({ name, inputs: [a], out }) => ({ note: '', latex: [`${tex(name)} = ${plus(a.v.map(tn))} = ${tn(out.v[0])}`] }),
  texBackward: (n, [a]) => [`${bar(a)}_i = ${bar(n)}\\quad\\text{for every } i`],
  whyBackward: 'Every element is added exactly once, so each has slope 1: all elements receive the same gradient.',
  backwardNumbers: ({ s, G }) => [`${bar(s.to)}_i = ${tn(G.v[0])}\\ \\Rightarrow\\ ${bar(s.to)} = ${mat(s.raw)}`],
};
