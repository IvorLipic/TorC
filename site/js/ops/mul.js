// ops/mul.js — elementwise (Hadamard) product a ⊙ b with broadcasting. Not a matrix product.
import { zip, expandTo } from './tensor.js';
import { tex, bar, mat, tn, par } from './texutil.js';
import { elementwiseForwardNumbers } from './elementwise.js';

export default {
  name: 'mul', title: 'multiplication', arity: 2,
  forward: (a, b) => zip(a, b, (x, y) => x * y),
  backward: (g, [a, b]) => [zip(g, b, (x, y) => x * y), zip(g, a, (x, y) => x * y)],
  texForward: (n, [a, b]) => `${tex(n)} = ${tex(a)} \\odot ${tex(b)}\\quad(\\text{element by element})`,
  whyForward: 'Multiply the matching elements (not a matrix product).',
  forwardNumbers: elementwiseForwardNumbers('\\odot', (a, b, o) => `${par(a)} \\cdot ${par(b)} = ${tn(o)}`, false),
  texBackward: (n, [a, b]) => [`${bar(a)} = ${bar(n)} \\odot ${tex(b)}`, `${bar(b)} = ${bar(n)} \\odot ${tex(a)}`],
  whyBackward: 'If one factor grows by δ, the product grows by (the other factor)·δ. So the slope with respect to one input is the <em>other</em> input, and the chain rule multiplies it by the gradient that arrived.',
  backwardNumbers: ({ name, s, G, other }) => {
    const O = expandTo(other.value, G.shape);
    return [`${bar(s.to)} = ${bar(name)} \\odot ${tex(other.name)} = ${mat(G)} \\odot ${mat(O)} = ${mat(s.raw)}`];
  },
};
