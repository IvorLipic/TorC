// ops/add.js — elementwise a + b with broadcasting (Tensor::add; autograd add()).
import { zip } from './tensor.js';
import { tex, bar, mat, tn, par } from './texutil.js';
import { elementwiseForwardNumbers } from './elementwise.js';

export default {
  name: 'add', title: 'addition', arity: 2,
  forward: (a, b) => zip(a, b, (x, y) => x + y),
  backward: (g) => [g, g],                                   // raw gradients; the engine sums them back to each input's shape
  texForward: (n, [a, b]) => `${tex(n)} = ${tex(a)} + ${tex(b)}`,
  whyForward: 'Add the matching elements. If one input is smaller it is first copied (broadcast) to the other’s shape.',
  forwardNumbers: elementwiseForwardNumbers('+', (a, b, o) => `${tn(a)} + ${par(b)} = ${tn(o)}`, true),
  texBackward: (n, [a, b]) => [`${bar(a)} = ${bar(n)}`, `${bar(b)} = ${bar(n)}`],
  whyBackward: 'Nudge either input by a tiny amount δ and the sum moves by exactly δ. The slope is 1, so the chain rule hands the incoming gradient on unchanged.',
  backwardNumbers: ({ name, s, G }) => [`${bar(s.to)} = ${bar(name)} = ${mat(G)}`],
};
