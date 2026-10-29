// ops/matmul.js — matrix product of two rank-2 tensors. Backward: dA = G Bᵀ, dB = Aᵀ G.
import { matmul, transpose } from './tensor.js';
import { tex, bar, mat, tn, par, aligned } from './texutil.js';

export default {
  name: 'matmul', title: 'matrix multiplication', arity: 2,
  forward: matmul,
  backward: (g, [a, b]) => [matmul(g, transpose(b)), matmul(transpose(a), g)],
  texForward: (n, [a, b]) => `${tex(n)}_{ij} = \\sum_k ${tex(a)}_{ik}\\,${tex(b)}_{kj}`,
  whyForward: 'Each output entry is the dot product of a row of the first matrix with a column of the second.',
  forwardNumbers: ({ name, inputs: [a, b], out }) => {
    const latex = [`${mat(a)}\\,${mat(b)} = ${mat(out)}`];
    const [r, k] = a.shape, c = b.shape[1];
    const rows = [];
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
      const terms = [];
      for (let p = 0; p < k; p++) terms.push(`${par(a.v[i * k + p])}\\cdot ${par(b.v[p * c + j])}`);
      rows.push(`n_{${i}${j}} &= ${terms.join(' + ')} = ${tn(out.v[i * c + j])}`);
    }
    if (rows.length <= 8) latex.push(aligned(rows, name));
    return { note: '', latex };
  },
  texBackward: (n, [a, b]) => [`${bar(a)} = ${bar(n)}\\, ${tex(b)}^{\\top}`, `${bar(b)} = ${tex(a)}^{\\top}\\, ${bar(n)}`],
  whyBackward: 'Each output entry is a dot product, so entry (i,k) of the first input feeds every output in row i, weighted by the second matrix. Collecting those slopes gives the two matrix products shown.',
  backwardNumbers: ({ name, s, G, other }) => (s.arg === 0
    ? [`${bar(s.to)} = ${bar(name)}\\,${tex(other.name)}^{\\top} = ${mat(G)}\\,${mat(transpose(other.value))} = ${mat(s.raw)}`]
    : [`${bar(s.to)} = ${tex(other.name)}^{\\top}\\,${bar(name)} = ${mat(transpose(other.value))}\\,${mat(G)} = ${mat(s.raw)}`]),
};
